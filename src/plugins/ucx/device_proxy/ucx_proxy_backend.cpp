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
#include "ucx_proxy_backend.h"
#include "../ucx_backend.h"
#include "nixl_log.h"
#include "nixl_types.h"

static_assert(sizeof(nixlUcxReq) <= sizeof(uint64_t),
              "UCX request pointer must fit in the opaque request token");

nixl_status_t
nixlUcxProxyBackendAdapter::init(uint32_t, uint32_t channel_count, uint32_t peer_capacity) {
    if (engine_ == nullptr || channel_count == 0 || peer_capacity == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const size_t worker_count = engine_->getSharedWorkersSize();
    const size_t expected_workers = static_cast<size_t>(channel_count) * peer_capacity;
    if (worker_count != expected_workers) {
        NIXL_ERROR << "UCX proxy requires one UCX worker per (channel, peer): workers="
                   << worker_count << " channels=" << channel_count
                   << " peer_capacity=" << peer_capacity << " expected=" << expected_workers;
        return NIXL_ERR_INVALID_PARAM;
    }
    peer_capacity_ = peer_capacity;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submit(const nixlBackendProxySubmission &submission,
                                   nixlBackendProxyRequest &request) {
    request = {};
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        return submitPut(submission, request);
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        return submitAtomicAdd(submission, request);
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

size_t
nixlUcxProxyBackendAdapter::workerIdFor(uint32_t channel_id, uint32_t peer_index) const {
    // Channel-major layout: each (channel, peer) owns a dedicated UCX worker/EP/QP so a
    // channel's put and its follow-up atomic to a given peer share one QP (ordering), while
    // distinct peers of a channel land on independent workers (per-peer progress isolation).
    return static_cast<size_t>(channel_id) * peer_capacity_ + peer_index;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      nixlBackendProxyRequest &request) {
    const size_t worker_id = workerIdFor(submission.channel_id, submission.peer_index);

    nixlUcxReq req = nullptr;
    const nixl_status_t status = engine_->submitProxyRmaWrite(
        submission.local.desc, submission.remote.desc, submission.size, worker_id, req);
    if (status == NIXL_IN_PROG) {
        request.token = reinterpret_cast<uint64_t>(req);
        request.context = worker_id;
    } else if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: submitProxyRmaWrite failed "
                      "status="
                   << status;
        return status;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex << submission.local.desc.addr << std::dec
               << " dst_addr=0x" << std::hex << submission.remote.desc.addr << std::dec
               << " size=" << submission.size << " token=" << request.token
               << " context=" << request.context;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitAtomicAdd(const nixlBackendProxySubmission &submission,
                                            nixlBackendProxyRequest &request) {
    // Same (channel, peer) -> worker mapping as submitPut so a channel's put and its follow-up
    // atomic flag travel the same worker/EP/QP, preserving IB write-before-atomic order.
    const size_t worker_id = workerIdFor(submission.channel_id, submission.peer_index);

    nixlUcxReq req = nullptr;
    const nixl_status_t status =
        engine_->submitProxyAtomicAdd(submission.remote.desc, submission.value, worker_id, req);
    if (status == NIXL_IN_PROG) {
        request.token = reinterpret_cast<uint64_t>(req);
        request.context = worker_id;
    } else if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: submitProxyAtomicAdd "
                      "failed status="
                   << status;
        return status;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: posted RDMA atomic add"
               << " dst_addr=0x" << std::hex << submission.remote.desc.addr << std::dec
               << " size=" << submission.size << " value=" << submission.value
               << " token=" << request.token << " context=" << request.context;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::checkCompletion(const nixlBackendProxyRequest &request) {
    if (engine_ == nullptr || !request) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlUcxReq req = reinterpret_cast<nixlUcxReq>(request.token);
    const nixl_status_t status = engine_->checkProxyRequest(req);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::checkCompletion: token=" << request.token
               << " context=" << request.context << " status=" << status;
    engine_->releaseProxyRequest(request.context, req);
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::releaseRequest(const nixlBackendProxyRequest &request) {
    if (engine_ == nullptr || !request) {
        return NIXL_ERR_INVALID_PARAM;
    }
    engine_->releaseProxyRequest(request.context, reinterpret_cast<nixlUcxReq>(request.token));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress() {
    if (engine_ != nullptr && !progress_thread_enabled_) {
        engine_->progress();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress(uint32_t channel_id, uint32_t peer_index) {
    if (engine_ != nullptr && !progress_thread_enabled_) {
        engine_->progress(workerIdFor(channel_id, peer_index));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::shutdown() {
    // Outstanding request tokens are owned by the runtime's inflight slots
    // and released there before adapter shutdown.
    return NIXL_SUCCESS;
}
