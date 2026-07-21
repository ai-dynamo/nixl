/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "proxy_worker.h"
#include "proxy_runtime.h"
#include "backend_adapter.h"
#include "nixl_log.h"
#include <chrono>
#include <cuda_runtime.h>

ProxyWorker::ProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                         const nixlProxyMemViewRegistry *proxy_memview_registry,
                         uint32_t *shutdown_word,
                         nixlProxyChannelState *channels,
                         std::atomic<nixl_proxy_channel_lifecycle_t> *channel_lifecycle,
                         uint32_t peer_capacity,
                         uint32_t channel_count,
                         uint32_t worker_index,
                         uint32_t worker_count,
                         uint64_t pthr_delay_us) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      channels_(channels),
      channel_lifecycle_(channel_lifecycle),
      peer_capacity_(peer_capacity),
      channel_count_(channel_count),
      worker_index_(worker_index),
      worker_count_(worker_count),
      pthr_delay_us_(pthr_delay_us) {}

ProxyWorker::~ProxyWorker() {
    join();
}

void
ProxyWorker::start() {
    thread_ = std::thread([this]() {
        NIXL_INFO << "ProxyWorker thread " << worker_index_ << " started";
        while (__atomic_load_n(shutdown_word_, __ATOMIC_ACQUIRE) ==
               static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING)) {
            runOnce();
            if (pthr_delay_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pthr_delay_us_));
            }
        }
        NIXL_INFO << "ProxyWorker thread " << worker_index_ << " exiting";
    });
}

void
ProxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
ProxyWorker::handleOwnedChannels(bool publish_completions) {
    // A logical channel has one CPU owner across every peer.
    for (uint32_t peer = 0; peer < peer_capacity_; ++peer) {
        const size_t first = static_cast<size_t>(peer) * channel_count_;
        for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
             channel_id += worker_count_) {
            const size_t slot = first + channel_id;
            nixlProxyChannelState &channel = channels_[slot];
            const nixl_proxy_channel_lifecycle_t lifecycle =
                channel_lifecycle_[slot].load(std::memory_order_acquire);

            if (lifecycle == nixl_proxy_channel_lifecycle_t::RESET_PENDING) {
                channel.resetLocalState();
                channel_lifecycle_[slot].store(nixl_proxy_channel_lifecycle_t::INACTIVE,
                                               std::memory_order_release);
                continue;
            }

            if (lifecycle != nixl_proxy_channel_lifecycle_t::ACTIVE || !channel.allocated()) {
                continue;
            }

            if (publish_completions) {
                publishCompletions(channel);
            } else {
                submitReady(channel);
            }
        }
    }
}

void
ProxyWorker::runOnce() {
    handleOwnedChannels(/*publish_completions=*/false);
    driveBackendProgress();
    handleOwnedChannels(/*publish_completions=*/true);
}

void
ProxyWorker::submitReady(nixlProxyChannelState &channel) {
    for (;;) {
        const uint64_t consumer_idx = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
        const uint64_t submit_idx = channel.submit_idx_;
        if (submit_idx - consumer_idx >= channel.ring_depth_) {
            break;
        }

        const uint32_t slot = static_cast<uint32_t>(submit_idx % channel.ring_depth_);
        // The GPU's system-scope release publication pairs with this acquire.
        const uint64_t op_idx =
            __atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE);
        if (op_idx == 0) {
            break;
        }

        nixlProxySubmission submission = channel.records_host_[slot];
        submission.op_idx = op_idx;
        __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
        submitToBackend(channel, slot, submission);
        channel.submit_idx_ = submit_idx + 1;
        NIXL_DEBUG << "ProxyWorker::submitReady: channel=" << channel.device_view.channel_id
                   << " submit=" << submit_idx << " opcode=" << static_cast<int>(submission.opcode)
                   << " op_idx=" << submission.op_idx << " size=" << submission.size;
    }
}

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                             uint32_t slot,
                             const nixlProxySubmission &submission) {
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx << " status=" << status;
        inflight.status = status;
    } else {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
                   << " opcode=" << static_cast<int>(submission.opcode)
                   << " channel=" << submission.channel_id << " local_addr=0x" << std::hex
                   << prepared_submission.local.desc.addr << " remote_addr=0x"
                   << prepared_submission.remote.desc.addr << std::dec
                   << " size=" << submission.size << " remote_agent='"
                   << prepared_submission.remote_agent << "'";

        uint64_t request_token = 0;
        status = backend_->submit(prepared_submission, request_token);
        inflight.backend_req_token = request_token;
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "ProxyWorker::submitToBackend: backend submit failed"
                       << " status=" << status << " op_idx=" << submission.op_idx
                       << " request_token=" << request_token;
            inflight.status = status;
        }
    }
    channel.inflight_slots_[slot] = inflight;
}

void
ProxyWorker::driveBackendProgress() {
    backend_->progress();
}

void
ProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    for (;;) {
        const uint64_t consumer_idx = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
        if (consumer_idx == channel.submit_idx_) {
            break;
        }

        const uint32_t slot = static_cast<uint32_t>(consumer_idx % channel.ring_depth_);
        nixlProxyRequestState &front = channel.inflight_slots_[slot];
        if (front.op_idx == 0) {
            break;
        }
        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backend_req_token);
            if (st == NIXL_IN_PROG) {
                break;
            }
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: channel=" << channel.device_view.channel_id
                   << " op_idx=" << front.op_idx << " status=" << st
                   << " token=" << front.backend_req_token;

        // Keep the first error visible, while continuing to retire work that
        // was accepted before the error became visible to the GPU.
        if (!channel.error_latched) {
            channel.completion_slot_host_->next_status = st;
            __atomic_store_n(
                &channel.completion_slot_host_->completed_idx, front.op_idx, __ATOMIC_RELEASE);
        }
        front = nixlProxyRequestState{};
        __atomic_store_n(channel.consumer_idx_host_, consumer_idx + 1, __ATOMIC_RELEASE);
        if (!channel.error_latched && st != NIXL_SUCCESS) {
            channel.error_latched = true;
        }
    }
}
