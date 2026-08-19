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
static_assert(sizeof(nixlUcxReq) <= sizeof(uint64_t),
              "UCX proxy requests must fit in the opaque token field");

nixlUcxReq
requestFromToken(const nixlBackendProxyRequest &request) {
    return reinterpret_cast<nixlUcxReq>(request.token);
}

uint64_t
tokenFromRequest(nixlUcxReq req) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req));
}
} // namespace

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
nixlUcxProxyBackendAdapter::resolveDirectPointers(const nixl_remote_meta_dlist_t &dlist,
                                                  std::vector<void *> &direct_ptrs) {

    direct_ptrs.assign(dlist.descCount(), nullptr);
    const size_t worker_id = engine_->getSharedWorkerId();

    size_t index = 0;
    for (const auto &desc : dlist) {
        if (desc.remoteAgent == nixl_null_agent) {
            ++index;
            continue;
        }

        const auto *metadata = static_cast<const nixlUcxPublicMetadata *>(desc.metadataP);

        void *direct_ptr = nullptr;
        const ucs_status_t status = ucp_rkey_ptr(
            metadata->getRkey(worker_id).get(), static_cast<uint64_t>(desc.addr), &direct_ptr);
        if (status == UCS_OK) {
            direct_ptrs[index] = direct_ptr;
        } else {
            NIXL_DEBUG << "nixlUcxProxyBackendAdapter::resolveDirectPointers: "
                          "direct access unavailable for descriptor "
                       << index << ": " << ucs_status_string(status);
        }
        ++index;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submit(const nixlBackendProxySubmission &submission,
                                   nixlBackendProxyRequest &request) {
    request = nixlBackendProxyRequest{};
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
nixlUcxProxyBackendAdapter::getSharedWorkerIdForChannelPeer(uint32_t channel_id,
                                                            uint32_t peer_index) const {
    return static_cast<size_t>(channel_id) * peer_capacity_ + peer_index;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitPut(const nixlBackendProxySubmission &submission,
                                      nixlBackendProxyRequest &request) {
    const size_t worker_id =
        getSharedWorkerIdForChannelPeer(submission.channel_id, submission.peer_index);

    nixlUcxReq req = nullptr;
    nixl_status_t status = engine_->submitProxyRmaWrite(
        submission.local.desc, submission.remote.desc, submission.size, worker_id, req);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: submitProxyRmaWrite failed "
                      "status="
                   << status;
        return status;
    }

    if (status == NIXL_IN_PROG) {
        request = nixlBackendProxyRequest{tokenFromRequest(req), worker_id};
    }
    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitPut: posted RDMA write"
               << " src_addr=0x" << std::hex << submission.local.desc.addr << std::dec
               << " dst_addr=0x" << std::hex << submission.remote.desc.addr << std::dec
               << " size=" << submission.size << " remote_agent='" << submission.remote_agent << "'"
               << " token=" << request.token << " context=" << request.context
               << " status=" << status;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::submitAtomicAdd(const nixlBackendProxySubmission &submission,
                                            nixlBackendProxyRequest &request) {
    const size_t worker_id =
        getSharedWorkerIdForChannelPeer(submission.channel_id, submission.peer_index);

    nixlUcxReq req = nullptr;
    nixl_status_t status =
        engine_->submitProxyAtomicAdd(submission.remote.desc, submission.value, worker_id, req);
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: submitProxyAtomicAdd "
                      "failed status="
                   << status;
        return status;
    }

    if (status == NIXL_IN_PROG) {
        request = nixlBackendProxyRequest{tokenFromRequest(req), worker_id};
    }
    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::submitAtomicAdd: posted RDMA atomic add"
               << " dst_addr=0x" << std::hex << submission.remote.desc.addr << std::dec
               << " size=" << submission.size << " value=" << submission.value << " remote_agent='"
               << submission.remote_agent << "'"
               << " token=" << request.token << " context=" << request.context
               << " status=" << status;
    return status;
}

nixl_status_t
nixlUcxProxyBackendAdapter::checkCompletion(const nixlBackendProxyRequest &request) {
    if (engine_ == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!request) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlUcxReq req = requestFromToken(request);
    const nixl_status_t status = engine_->checkProxyRequest(req);
    if (status == NIXL_IN_PROG) {
        return NIXL_IN_PROG;
    }

    NIXL_DEBUG << "nixlUcxProxyBackendAdapter::checkCompletion: token=" << request.token
               << " context=" << request.context << " status=" << status;
    engine_->releaseProxyRequest(request.context, req, false);
    return status;
}

void
nixlUcxProxyBackendAdapter::releaseRequest(const nixlBackendProxyRequest &request) {
    if (engine_ == nullptr || !request) {
        return;
    }

    engine_->releaseProxyRequest(request.context, requestFromToken(request), true);
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress() {
    if (engine_ != nullptr) {
        engine_->progress();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::progress(uint32_t channel_id, uint32_t peer_index) {
    if (engine_ != nullptr) {
        engine_->progress(getSharedWorkerIdForChannelPeer(channel_id, peer_index));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxProxyBackendAdapter::shutdown() {
    return NIXL_SUCCESS;
}
