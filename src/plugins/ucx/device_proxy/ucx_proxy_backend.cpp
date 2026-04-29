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

namespace {
constexpr uint64_t kInvalidToken = 0;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submit(const nixlBackendProxySubmission &submission,
                                   uint64_t &request_token) {
    request_token = kInvalidToken;
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    switch (submission.opcode) {
    case nixl_proxy_opcode_t::PUT:
        return submitPut(submission, request_token);
    case nixl_proxy_opcode_t::ATOMIC_ADD:
        return submitAtomicAdd(submission, request_token);
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      uint64_t &request_token) {
    nixl_meta_dlist_t local_list(submission.local.mem_type);
    local_list.addDesc(submission.local.desc);

    nixl_meta_dlist_t remote_list(submission.remote.mem_type);
    remote_list.addDesc(submission.remote.desc);

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine_->prepXfer(
        NIXL_WRITE, local_list, remote_list, submission.remote_agent, handle);
    if (status != NIXL_SUCCESS) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: prepXfer failed status=" << status;
        return status;
    }

    status = engine_->postXfer(
        NIXL_WRITE, local_list, remote_list, submission.remote_agent, handle);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: postXfer failed status=" << status;
        engine_->releaseReqH(handle);
        return status;
    }

    request_token = trackRequest(handle);
    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex
               << submission.local.desc.addr << std::dec
               << " dst_addr=0x" << std::hex
               << submission.remote.desc.addr << std::dec
               << " size=" << submission.size
               << " remote_agent='" << submission.remote_agent << "'"
               << " token=" << request_token;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitAtomicAdd(const nixlBackendProxySubmission &,
                                            uint64_t &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlUcxProxyBackendAdapter::checkCompletion(uint64_t request_token) {
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lock(request_mutex_);
    const auto it = tracked_requests_.find(request_token);
    if (it == tracked_requests_.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlBackendReqH *handle = it->second;
    const nixl_status_t status = engine_->checkXfer(handle);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::checkCompletion: token=" << request_token
               << " status=" << status;
    engine_->releaseReqH(handle);
    tracked_requests_.erase(it);
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress() {
    if (engine_ != nullptr && !progress_thread_enabled_) {
        engine_->progress();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::shutdown() {
    NIXL_INFO << "nixlUcxProxyBackendAdapter::shutdown: releasing "
              << tracked_requests_.size() << " tracked request(s)";
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        if (engine_ != nullptr) {
            for (auto &entry : tracked_requests_) {
                engine_->releaseReqH(entry.second);
            }
        }
        tracked_requests_.clear();
    }
    return NIXL_SUCCESS;
}

uint64_t
nixlUcxProxyBackendAdapter::trackRequest(nixlBackendReqH *handle) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    const uint64_t token = next_request_token_++;
    tracked_requests_.emplace(token, handle);
    return token;
}
