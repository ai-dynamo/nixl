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
#include "proxy_backend_ops.h"
#include "nixl_log.h"
#include <chrono>

ProxyWorker::ProxyWorker(const nixlProxyBackendOps *backend_ops,
                         const nixlProxyMemViewRegistry *proxy_memview_registry,
                         std::atomic<uint64_t> *shutdown_state,
                         nixlProxyChannelState *channels,
                         uint32_t max_peers,
                         uint32_t channel_count,
                         uint32_t worker_index,
                         uint32_t worker_count,
                         uint64_t pthr_delay_us) noexcept
    : backend_ops_(backend_ops),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_state_(shutdown_state),
      channels_(channels),
      max_peers_(max_peers),
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
        while (shutdown_state_->load(std::memory_order_acquire) ==
               static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)) {
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

nixlProxyChannelState *
ProxyWorker::getChannelState(uint32_t peer, uint32_t channel_id) {
    return &channels_[static_cast<size_t>(channel_id) * max_peers_ + peer];
}

template<typename Fn>
void
ProxyWorker::forEachOwnedChannel(Fn &&fn) {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            fn(*getChannelState(peer, channel_id), channel_id, peer);
        }
    }
}

void
ProxyWorker::publishOwnedChannels() {
    forEachOwnedChannel([this](nixlProxyChannelState &channel, uint32_t, uint32_t) {
        publishCompletions(channel);
    });
}

void
ProxyWorker::submitOwnedChannels() {
    forEachOwnedChannel([this](nixlProxyChannelState &channel, uint32_t, uint32_t peer) {
        submitReady(channel, peer);
    });
}

void
ProxyWorker::runOnce() {
    submitOwnedChannels();
    driveBackendProgress();
    publishOwnedChannels();
}

void
ProxyWorker::submitReady(nixlProxyChannelState &channel, uint32_t peer) {
    const uint64_t consumer_idx = channel.consumer_idx_shadow_;
    const uint64_t submit_idx = channel.submit_idx_;

    if (submit_idx - consumer_idx >= channel.ring_depth_) {
        return;
    }

    const uint32_t slot = static_cast<uint32_t>(submit_idx % channel.ring_depth_);
    const uint64_t op_idx = __atomic_load_n(&channel.recordsHost()[slot].op_idx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return;
    }

    nixlProxySubmission submission = channel.recordsHost()[slot];
    submission.op_idx = op_idx;

    __atomic_store_n(&channel.recordsHost()[slot].op_idx, 0, __ATOMIC_RELAXED);
    channel.submit_idx_ = submit_idx + 1;

    NIXL_DEBUG << "ProxyWorker::submitReady: channel=" << submission.channel_id
               << " submit=" << submit_idx << " opcode=" << static_cast<int>(submission.opcode)
               << " op_idx=" << submission.op_idx << " size=" << submission.size;
    submitToBackend(channel, peer, slot, submission);
}

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                             uint32_t peer,
                             uint32_t slot,
                             const nixlProxySubmission &submission) {
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;

    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    prepared_submission.peer_index = peer;
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx << " status=" << status;
        inflight.status = status;
        channel.inflight_slots_[slot] = inflight;
        return;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id << " local_addr=0x" << std::hex
               << prepared_submission.local.desc.addr << " remote_addr=0x"
               << prepared_submission.remote.desc.addr << std::dec << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remote_agent << "'";

    status = backend_ops_->submit(prepared_submission, inflight.backend_request);
    inflight.status = status;
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_ERROR << "ProxyWorker::submitToBackend: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.op_idx
                   << " request_token=" << inflight.backend_request.token
                   << " request_context=" << inflight.backend_request.context;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: submitted op_idx=" << submission.op_idx
               << " request_token=" << inflight.backend_request.token
               << " request_context=" << inflight.backend_request.context << " status=" << status;
    channel.inflight_slots_[slot] = inflight;
}

void
ProxyWorker::driveBackendProgress() {
    forEachOwnedChannel([this](nixlProxyChannelState &, uint32_t channel_id, uint32_t peer) {
        backend_ops_->progress(channel_id, peer);
    });
}

void
ProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    for (;;) {
        const uint64_t consumer_idx = channel.consumer_idx_shadow_;
        if (consumer_idx == channel.submit_idx_) {
            break;
        }

        const uint32_t slot = static_cast<uint32_t>(consumer_idx % channel.ring_depth_);
        nixlProxyRequestState &front = channel.inflight_slots_[slot];

        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_ops_->check_completion(front.backend_request);
            if (st == NIXL_IN_PROG) {
                break;
            }
            front.status = st;
        }
        NIXL_DEBUG << "ProxyWorker::publishCompletions: op_idx=" << front.op_idx << " status=" << st
                   << " token=" << front.backend_request.token
                   << " context=" << front.backend_request.context;

        if (channel.completionSlotHost()->next_status >= 0) {
            channel.completionSlotHost()->next_status = st;
            __atomic_store_n(
                &channel.completionSlotHost()->completed_idx, front.op_idx, __ATOMIC_RELEASE);
        }
        if (channel.publishConsumerIdx(consumer_idx + 1) != NIXL_SUCCESS) {
            NIXL_ERROR << "ProxyWorker::publishCompletions: failed to publish CI"
                       << " consumer_idx=" << consumer_idx + 1;
            break;
        }
        front = nixlProxyRequestState{};
    }
}
