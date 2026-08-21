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

nixlProxyWorker::nixlProxyWorker(nixlDeviceProxyBackendAdapter *backend,
                                 const nixlProxyMemViewRegistry *proxy_memview_registry,
                                 std::atomic<uint64_t> *shutdown_state,
                                 nixlProxyChannelState *channels,
                                 uint32_t max_peers,
                                 uint32_t channel_count,
                                 uint32_t worker_index,
                                 uint32_t worker_count,
                                 uint64_t pthr_delay_us) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_state_(shutdown_state),
      channels_(channels),
      max_peers_(max_peers),
      channel_count_(channel_count),
      worker_index_(worker_index),
      worker_count_(worker_count),
      pthr_delay_us_(pthr_delay_us) {}

nixlProxyWorker::~nixlProxyWorker() {
    join();
}

void
nixlProxyWorker::start() {
    thread_ = std::thread([this]() {
        NIXL_INFO << "nixlProxyWorker thread " << worker_index_ << " started";
        while (shutdown_state_->load(std::memory_order_acquire) ==
               static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)) {
            runOnce();
            if (pthr_delay_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pthr_delay_us_));
            }
        }
        NIXL_INFO << "nixlProxyWorker thread " << worker_index_ << " exiting";
    });
}

void
nixlProxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

nixlProxyChannelState *
nixlProxyWorker::getChannelState(uint32_t peer, uint32_t channel_id) {
    return &channels_[static_cast<size_t>(channel_id) * max_peers_ + peer];
}

void
nixlProxyWorker::publishOwnedChannels() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            nixlProxyChannelState *channel = getChannelState(peer, channel_id);
            publishCompletions(*channel);
        }
    }
}

void
nixlProxyWorker::submitOwnedChannels() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            nixlProxyChannelState *channel = getChannelState(peer, channel_id);
            submitReady(*channel, peer);
        }
    }
}

void
nixlProxyWorker::runOnce() {
    submitOwnedChannels();
    driveBackendProgress();
    publishOwnedChannels();
}

void
nixlProxyWorker::submitReady(nixlProxyChannelState &channel, uint32_t peer) {
    const uint64_t consumer_idx = channel.consumerIdxShadow;
    const uint64_t submit_idx = channel.submitIdx;

    if (submit_idx - consumer_idx >= channel.ringDepth) {
        return;
    }

    const uint32_t slot = static_cast<uint32_t>(submit_idx % channel.ringDepth);
    const uint64_t op_idx = __atomic_load_n(&channel.recordsHost[slot].opIdx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return;
    }

    nixlProxySubmission submission = channel.recordsHost[slot];
    submission.opIdx = op_idx;

    __atomic_store_n(&channel.recordsHost[slot].opIdx, 0, __ATOMIC_RELAXED);
    channel.submitIdx = submit_idx + 1;

    NIXL_DEBUG << "nixlProxyWorker::submitReady: channel=" << submission.channelId
               << " submit=" << submit_idx << " opcode=" << static_cast<int>(submission.opcode)
               << " op_idx=" << submission.opIdx << " size=" << submission.size;
    submitToBackend(channel, peer, slot, submission);
}

void
nixlProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                                 uint32_t peer,
                                 uint32_t slot,
                                 const nixlProxySubmission &submission) {
    nixlProxyRequestState inflight{};
    inflight.opIdx = submission.opIdx;

    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    prepared_submission.peerIndex = peer;
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.opIdx << " status=" << status;
        inflight.status = status;
        channel.inflightSlots[slot] = inflight;
        return;
    }

    NIXL_DEBUG << "nixlProxyWorker::submitToBackend: op_idx=" << submission.opIdx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channelId << " local_addr=0x" << std::hex
               << prepared_submission.local.desc.addr << " remote_addr=0x"
               << prepared_submission.remote.desc.addr << std::dec << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remoteAgent << "'";

    status = backend_->submit(prepared_submission, inflight.backendRequest);
    inflight.status = status;
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_ERROR << "nixlProxyWorker::submitToBackend: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.opIdx
                   << " request_token=" << inflight.backendRequest.token
                   << " request_context=" << inflight.backendRequest.context;
    }

    NIXL_DEBUG << "nixlProxyWorker::submitToBackend: submitted op_idx=" << submission.opIdx
               << " request_token=" << inflight.backendRequest.token
               << " request_context=" << inflight.backendRequest.context << " status=" << status;
    channel.inflightSlots[slot] = inflight;
}

void
nixlProxyWorker::driveBackendProgress() {
    for (uint32_t channel_id = worker_index_; channel_id < channel_count_;
         channel_id += worker_count_) {
        for (uint32_t peer = 0; peer < max_peers_; ++peer) {
            backend_->progress(channel_id, peer);
        }
    }
}

void
nixlProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    for (;;) {
        const uint64_t consumer_idx = channel.consumerIdxShadow;
        if (consumer_idx == channel.submitIdx) {
            break;
        }

        const uint32_t slot = static_cast<uint32_t>(consumer_idx % channel.ringDepth);
        nixlProxyRequestState &front = channel.inflightSlots[slot];

        nixl_status_t st;
        if (front.status != NIXL_IN_PROG) {
            st = front.status;
        } else {
            st = backend_->checkCompletion(front.backendRequest);
            if (st == NIXL_IN_PROG) {
                break;
            }
            front.status = st;
        }
        NIXL_DEBUG << "nixlProxyWorker::publishCompletions: op_idx=" << front.opIdx
                   << " status=" << st << " token=" << front.backendRequest.token
                   << " context=" << front.backendRequest.context;

        if (channel.completionSlotHost->nextStatus >= 0) {
            channel.completionSlotHost->nextStatus = st;
            __atomic_store_n(
                &channel.completionSlotHost->completedIdx, front.opIdx, __ATOMIC_RELEASE);
        }
        if (channel.publishConsumerIdx(consumer_idx + 1) != NIXL_SUCCESS) {
            NIXL_ERROR << "nixlProxyWorker::publishCompletions: failed to publish CI"
                       << " consumer_idx=" << consumer_idx + 1;
            break;
        }
        front = nixlProxyRequestState{};
    }
}
