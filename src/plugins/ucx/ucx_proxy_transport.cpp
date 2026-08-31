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

#include "ucx_backend.h"

#include <cstdint>
#include <vector>

#include "common/nixl_log.h"
#include "device/proxy/proxy_transport.h"

static_assert(sizeof(nixlUcxReq) <= sizeof(uint64_t),
              "UCX proxy requests must fit in the opaque token field");

/**
 * The device proxy's transport over the engine's shared UCX workers, one worker per
 * (channel, peer) ring. Nested in the engine, it reads the workers and the descriptor
 * metadata with the engine's own access.
 */
class nixlUcxEngine::proxyTransportImpl final : public nixl::proxyTransport {
public:
    explicit proxyTransportImpl(worker_span_t workers) noexcept : workers_(workers) {}

    nixl_status_t
    init(const nixl::proxyConfig &config) override {
        if (workers_.size() != config.ringCount()) {
            NIXL_ERROR << "UCX proxy requires one UCX worker per (channel, peer): "
                       << workers_.size() << " worker(s) for " << config.ringCount() << " ring(s)";
            return NIXL_ERR_INVALID_PARAM;
        }
        max_peers_ = config.max_peers;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &submission,
           nixl::proxyBackendRequest &request) override {
        request = nixl::proxyBackendRequest{};
        const size_t worker_id = workerId(submission.channel_id, submission.peer_index);

        nixlUcxReq req = nullptr;
        nixl_status_t status;
        switch (submission.opcode) {
        case nixl_proxy_opcode_t::PUT:
            status = postPut(submission, worker_id, req);
            break;
        case nixl_proxy_opcode_t::ATOMIC_ADD:
            status = postAtomicAdd(submission, worker_id, req);
            break;
        default:
            return NIXL_ERR_NOT_SUPPORTED;
        }

        // Only puts ever yield a request: a post-mode atomic completes inside
        // ucp_atomic_op_nbx and never hands back a handle, so there is nothing
        // to track, poll or release for one.
        if (status == NIXL_IN_PROG) {
            request = nixl::proxyBackendRequest{tokenFrom(req)};
        }
        NIXL_DEBUG << "device proxy submit: opcode=" << static_cast<int>(submission.opcode)
                   << " src_addr=0x" << std::hex << submission.local.addr << " dst_addr=0x"
                   << submission.remote.addr << std::dec << " size=" << submission.size
                   << " token=" << request.token << " status=" << status;
        return status;
    }

    nixl_status_t
    checkCompletion(uint32_t channel,
                    uint32_t peer,
                    const nixl::proxyBackendRequest &request) override {
        if (!request) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const nixlUcxReq req = reqFrom(request);
        const nixl_status_t status = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (status == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }

        NIXL_DEBUG << "device proxy completion: token=" << request.token << " status=" << status;
        release(workerId(channel, peer), req);
        return status;
    }

    void
    progress(uint32_t channel, uint32_t peer) noexcept override {
        workers_[workerId(channel, peer)]->progress();
    }

    nixl_status_t
    quiesce(uint32_t channel, uint32_t peer) override {
        const auto &worker = workers_[workerId(channel, peer)];
        const ucp_request_param_t params{};
        auto *request = ucp_worker_flush_nbx(worker->get(), &params);
        if (UCS_PTR_IS_ERR(request)) {
            return nixl::ucx::ucsToNixlStatus(UCS_PTR_STATUS(request));
        }
        if (request == nullptr) {
            return NIXL_SUCCESS;
        }
        ucs_status_t status;
        do {
            worker->progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        worker->reqRelease(request);
        return nixl::ucx::ucsToNixlStatus(status);
    }

    nixl_status_t
    shutdown() override {
        return NIXL_SUCCESS;
    }

    /** Any worker's rkey resolves the same address; worker 0 always exists. */
    nixl_status_t
    resolveDirectPtrs(const nixl_remote_meta_dlist_t &dlist,
                      std::vector<void *> &direct_ptrs) override {
        direct_ptrs.assign(dlist.descCount(), nullptr);

        size_t index = 0;
        for (const auto &desc : dlist) {
            // A hole, which the registry marks unusable: nothing to resolve.
            if (desc.remoteAgent == nixl_null_agent || desc.metadataP == nullptr) {
                ++index;
                continue;
            }

            const auto *metadata = static_cast<const nixlUcxPublicMetadata *>(desc.metadataP);
            void *direct_ptr = nullptr;
            const ucs_status_t status = ucp_rkey_ptr(
                metadata->getRkey(0).get(), static_cast<uint64_t>(desc.addr), &direct_ptr);
            if (status == UCS_OK) {
                direct_ptrs[index] = direct_ptr;
            } else {
                NIXL_DEBUG << "device proxy: direct access unavailable for descriptor " << index
                           << ": " << ucs_status_string(status);
            }
            ++index;
        }

        return NIXL_SUCCESS;
    }

private:
    /** The one place the (channel, peer) -> shared worker layout is spelled out. */
    [[nodiscard]] size_t
    workerId(uint32_t channel, uint32_t peer) const noexcept {
        return static_cast<size_t>(channel) * max_peers_ + peer;
    }

    /** Post one put on the endpoint that reaches the submission's peer. */
    nixl_status_t
    postPut(const nixl::proxyBackendSubmission &submission,
            size_t worker_id,
            nixlUcxReq &req) const {
        req = nullptr;
        const nixlMetaDesc &local = submission.local;
        const nixlMetaDesc &remote = submission.remote;

        if (local.len != submission.size || remote.len != submission.size) {
            return NIXL_ERR_INVALID_PARAM;
        }

        if (worker_id >= workers_.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        auto *lmd = static_cast<nixlUcxPrivateMetadata *>(local.metadataP);
        auto *rmd = static_cast<nixlUcxPublicMetadata *>(remote.metadataP);
        if (lmd == nullptr || rmd == nullptr || rmd->conn == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        auto &ep = rmd->conn->getEp(worker_id);
        return ep->write(reinterpret_cast<void *>(local.addr),
                         lmd->mem,
                         static_cast<uint64_t>(remote.addr),
                         rmd->getRkey(worker_id),
                         submission.size,
                         req);
    }

    nixl_status_t
    postAtomicAdd(const nixl::proxyBackendSubmission &submission,
                  size_t worker_id,
                  nixlUcxReq &req) const {
        req = nullptr;
        const nixlMetaDesc &remote = submission.remote;

        if (remote.len != sizeof(uint64_t)) {
            return NIXL_ERR_INVALID_PARAM;
        }

        if (worker_id >= workers_.size()) {
            return NIXL_ERR_INVALID_PARAM;
        }

        auto *rmd = static_cast<nixlUcxPublicMetadata *>(remote.metadataP);
        if (rmd == nullptr || rmd->conn == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        // Order the counter update after the puts already posted on this worker:
        // the receiver treats the counter as the signal that the data has landed.
        const auto status =
            nixl::ucx::ucsToNixlStatus(ucp_worker_fence(workers_[worker_id]->get()));
        if (status != NIXL_SUCCESS) {
            return status;
        }

        auto &ep = rmd->conn->getEp(worker_id);
        return ep->atomicAdd(
            submission.value, static_cast<uint64_t>(remote.addr), rmd->getRkey(worker_id), req);
    }

    /**
     * Drop the proxy's bookkeeping for a request. This does NOT abort the
     * operation: under err-mode none, which the proxy requires, UCX offers no
     * way to abort an in-flight put. The NIC may keep reading the source
     * buffer and land the write remotely until the transport reports the
     * failure - so the memory behind a released operation must stay mapped
     * until the endpoint is torn down.
     */
    void
    release(size_t worker_id, nixlUcxReq req) const {
        if (req == nullptr) {
            return;
        }
        if (worker_id >= workers_.size()) {
            NIXL_WARN << "UCX proxy transport: invalid worker_id=" << worker_id;
            return;
        }

        // The caller has observed terminal completion.
        workers_[worker_id]->reqRelease(req);
    }

    [[nodiscard]] static uint64_t
    tokenFrom(nixlUcxReq req) noexcept {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req));
    }

    [[nodiscard]] static nixlUcxReq
    reqFrom(const nixl::proxyBackendRequest &request) noexcept {
        return reinterpret_cast<nixlUcxReq>(request.token);
    }

    const worker_span_t workers_;
    /** Set by init(), before any worker thread exists. */
    uint32_t max_peers_ = 0;
};
