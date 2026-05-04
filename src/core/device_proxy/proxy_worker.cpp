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
                         nixlProxyChannelState *assigned_channels,
                         uint32_t assigned_channel_count,
                         uint64_t pthr_delay_us) noexcept
    : backend_(backend),
      proxy_memview_registry_(proxy_memview_registry),
      shutdown_word_(shutdown_word),
      assigned_channels_(assigned_channels),
      assigned_channel_count_(assigned_channel_count),
      pthr_delay_us_(pthr_delay_us) {}

ProxyWorker::~ProxyWorker() {
    join();
}

void
ProxyWorker::start(uint32_t worker_idx) {
    thread_ = std::thread([this, worker_idx]() {
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " started";
        while (__atomic_load_n(shutdown_word_, __ATOMIC_ACQUIRE) ==
               static_cast<uint32_t>(nixl_proxy_control_state_t::RUNNING)) {
            runOnce();
            if (pthr_delay_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(pthr_delay_us_));
            }
        }
        NIXL_INFO << "ProxyWorker thread " << worker_idx << " exiting";
    });
}

void
ProxyWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
ProxyWorker::runOnce() {
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        nixlProxySubmission submission;
        while (tryDequeue(channel, submission)) {
            submitToBackend(channel, submission);
        }
    }
    driveBackendProgress();
    for (uint32_t i = 0; i < assigned_channel_count_; i++) {
        nixlProxyChannelState &channel = assigned_channels_[i];
        publishCompletions(channel);
    }
}

bool
ProxyWorker::tryDequeue(nixlProxyChannelState &channel, nixlProxySubmission &submission) {
    // Sole writer of consumer_idx on host — relaxed load is sufficient.
    uint64_t local_consumer_idx = __atomic_load_n(channel.consumer_idx_host_, __ATOMIC_RELAXED);
    uint32_t slot = static_cast<uint32_t>(local_consumer_idx % channel.ring_depth_);
    // op_idx is the GPU-to-CPU signal that the record is written
    // (pairs with release store in device enqueue).  No producer index
    // read on host — it is GPU-internal for slot allocation.
    const uint64_t op_idx = __atomic_load_n(&channel.records_host_[slot].op_idx, __ATOMIC_ACQUIRE);
    if (op_idx == 0) {
        return false;
    }
    submission = channel.records_host_[slot];
    submission.op_idx = op_idx;
    __atomic_store_n(&channel.records_host_[slot].op_idx, 0, __ATOMIC_RELAXED);
    __atomic_store_n(channel.consumer_idx_host_, local_consumer_idx + 1, __ATOMIC_RELEASE);
    NIXL_DEBUG << "ProxyWorker::tryDequeue: channel=" << channel.device_view.channel_id
               << " consumer=" << local_consumer_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " op_idx=" << submission.op_idx << " size=" << submission.size;
    return true;
}

void
ProxyWorker::submitToBackend(nixlProxyChannelState &channel,
                             const nixlProxySubmission &submission) {
    nixlBackendProxySubmission prepared_submission;
    nixl_status_t status =
        proxy_memview_registry_->prepareSubmission(submission, prepared_submission);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "ProxyWorker::submitToBackend: submission preparation failed"
                   << " op_idx=" << submission.op_idx << " status=" << status;
        channel.inflight_requests.push_back({submission.op_idx, 0, status});
        // The terminal error is queued for publishCompletions(); the worker handled it.
        return;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: op_idx=" << submission.op_idx
               << " opcode=" << static_cast<int>(submission.opcode)
               << " channel=" << submission.channel_id << " local_addr=0x" << std::hex
               << prepared_submission.local.desc.addr << " remote_addr=0x"
               << prepared_submission.remote.desc.addr << std::dec << " size=" << submission.size
               << " remote_agent='" << prepared_submission.remote_agent << "'";

    uint64_t request_token = 0;
    nixlProxyRequestState inflight{};
    inflight.op_idx = submission.op_idx;
    status = backend_->submit(prepared_submission, request_token);
    inflight.backend_req_token = request_token;
    if (status != NIXL_SUCCESS) {
        // backend submit failed, so status is already terminal and can be
        // published without polling the backend.
        NIXL_ERROR << "ProxyWorker::submitToBackend: backend submit failed"
                   << " status=" << status << " op_idx=" << submission.op_idx
                   << " request_token=" << request_token;
        inflight.status = status;
    }

    NIXL_DEBUG << "ProxyWorker::submitToBackend: submitted op_idx=" << submission.op_idx
               << " request_token=" << request_token << " status=" << status;
    channel.inflight_requests.push_back(inflight);
}

void
ProxyWorker::driveBackendProgress() {
    backend_->progress();
}

void
ProxyWorker::publishCompletions(nixlProxyChannelState &channel) {
    if (channel.error_latched) {
        return;
    }
    while (!channel.inflight_requests.empty()) {
        nixlProxyRequestState &front = channel.inflight_requests.front();
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
        channel.completion_slot_host_->next_status = st;
        __atomic_store_n(
            &channel.completion_slot_host_->completed_idx, front.op_idx, __ATOMIC_RELEASE);
        channel.inflight_requests.pop_front();
        if (st != NIXL_SUCCESS) {
            channel.error_latched = true;
            break;
        }
    }
}
