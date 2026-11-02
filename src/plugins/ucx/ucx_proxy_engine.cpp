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

#include "ucx_proxy_engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/backend.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "device/device_memview.h"
#include "device/device_ops.h"
#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_runtime.h"
#include "device/proxy/proxy_transport.h"

static_assert(sizeof(nixlUcxReq) <= sizeof(uint64_t),
              "UCX proxy requests must fit in the opaque token field");

namespace nixl::ucx {
/**
 * The device proxy's transport over the engine's shared UCX workers, one worker per
 * (channel, peer) ring. Nested in the engine, it reads the workers with the engine's own
 * access.
 */
class proxyEngine::transport final : public nixl::proxyTransport {
public:
    transport(worker_span_t workers, const nixl::proxyConfig &config) noexcept
        : workers_(workers),
          max_peers_(config.max_peers) {}

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &submission,
           nixl::proxyBackendRequest &request) override {
        request = nixl::proxyBackendRequest{};
        const size_t worker_id = workerId(submission.channel_id, submission.peer_index);

        nixlUcxReq req = nullptr;
        nixl_status_t status = NIXL_ERR_NOT_SUPPORTED;
        switch (submission.opcode) {
        case nixl_proxy_opcode_t::PUT:
            status = postPut(submission, worker_id, req);
            break;
        case nixl_proxy_opcode_t::ATOMIC_ADD:
            status = postAtomicAdd(submission, worker_id, req);
            break;
        }

        if (status == NIXL_IN_PROG) {
            request = nixl::proxyBackendRequest{tokenFrom(req)};
        }
        if (submission.opcode == nixl_proxy_opcode_t::PUT) {
            NIXL_DEBUG << "device proxy submit: PUT src_addr=" << std::hex << submission.local.addr
                       << " dst_addr=" << submission.remote.addr << std::dec
                       << " size=" << submission.size << " token=" << request.token
                       << " status=" << status;
        } else {
            NIXL_DEBUG << "device proxy submit: ATOMIC_ADD dst_addr=" << std::hex
                       << submission.remote.addr << std::dec << " value=" << submission.value
                       << " token=" << request.token << " status=" << status;
        }
        return status;
    }

    nixl_status_t
    checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &request) override {
        if (!request) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const nixlUcxReq req = reqFrom(request);
        const nixl_status_t status = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (status == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }

        NIXL_DEBUG << "device proxy completion: token=" << request.token << " status=" << status;
        ucp_request_free(req);
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

    /** Any worker's rkey resolves the same address; worker 0 always exists. */
    nixl_status_t
    resolveDirectPtrs(const nixl_remote_meta_dlist_t &dlist,
                      std::vector<void *> &direct_ptrs) override {
        direct_ptrs.assign(dlist.descCount(), nullptr);

        size_t index = 0;
        for (const auto &desc : dlist) {
            // A hole, which the memory-view manager marks unusable: nothing to resolve.
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
                         lmd->getMem(),
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

    [[nodiscard]] static uint64_t
    tokenFrom(nixlUcxReq req) noexcept {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req));
    }

    [[nodiscard]] static nixlUcxReq
    reqFrom(const nixl::proxyBackendRequest &request) noexcept {
        return reinterpret_cast<nixlUcxReq>(request.token);
    }

    const worker_span_t workers_;
    const uint32_t max_peers_;
};
} // namespace nixl::ucx

namespace {
[[nodiscard]] nixl::deviceOps &
requireDeviceOps() {
    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        nixl::throwRuntimeError(
            "device proxy needs a device operations implementation; none is loaded");
    }
    return *ops;
}

/** One shared worker per (channel, peer) ring, RW sync and error mode none; `storage` holds the
 *  custom params. */
[[nodiscard]] nixlBackendInitParams
proxyInitParams(const nixlBackendInitParams &init_params,
                const nixl::proxyConfig &config,
                nixl_b_params_t &&storage) {
    const size_t ring_workers = config.ringCount();
    const auto num_workers =
        nixl::getBackendParamOptional<size_t>(init_params.customParams, "num_workers");
    if (num_workers.value_or(0) > ring_workers) {
        nixl::throwRuntimeError(
            "num_workers=", *num_workers, " exceeds the device proxy's ", ring_workers, " rings");
    }

    // Under peer, UCX disables every AMO protocol and cancels the proxy's atomics.
    const auto err_mode = nixl::getBackendParamOptional<std::string>(
        init_params.customParams, std::string(nixl_ucx_err_handling_param_name));
    if (err_mode.has_value() && ucx_err_mode_from_string(*err_mode) != UCP_ERR_HANDLING_MODE_NONE) {
        nixl::throwRuntimeError(nixl_ucx_err_handling_param_name,
                                "=",
                                *err_mode,
                                " is not supported with device_proxy=true; use none");
    }

    storage = init_params.customParams ? *init_params.customParams : nixl_b_params_t{};
    storage["num_workers"] = std::to_string(ring_workers);
    storage[std::string(nixl_ucx_err_handling_param_name)] = "none";
    nixlBackendInitParams params = init_params;
    params.customParams = &storage;
    // Proxy progress can overlap metadata operations under any agent lock mode.
    params.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    return params;
}
} // namespace

namespace nixl::ucx {
proxyEngine::proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config)
    // The temporary params outlive the base constructor, the only place they are read.
    : nixlUcxEngine(proxyInitParams(init_params, config, nixl_b_params_t{})),
      ops_(requireDeviceOps()),
      runtime_(nixl::proxyRuntime::create(std::make_unique<transport>(getSharedWorkers(), config),
                                          config,
                                          ops_)) {
    NIXL_INFO << "Engine-owned device proxy enabled: " << config.channel_count << " channel(s), "
              << config.thread_count << " thread(s), max_peers=" << config.max_peers
              << ", ring_depth=" << config.ring_depth;
}

proxyEngine::~proxyEngine() = default;

nixl_status_t
proxyEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Every shared worker is a ring worker that its proxy thread progresses.
    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

void
proxyEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

template<typename DlistT>
nixl_status_t
proxyEngine::prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const {
    nixlMemViewH proxy_mvh = nullptr;
    nixl_status_t status = runtime_->prepMemView(dlist, &proxy_mvh);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = nixl::deviceMemViewAllocate(ops_, nixl_device_exec_mode_t::PROXY, proxy_mvh, mvh);
    if (status != NIXL_SUCCESS) {
        runtime_->releaseMemView(proxy_mvh);
    }
    return status;
}

nixl_status_t
proxyEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

nixl_status_t
proxyEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

void
proxyEngine::releaseMemView(nixlMemViewH mem_view) const {
    if (mem_view == nullptr) {
        return;
    }

    nixlMemViewH proxy_mvh = nullptr;
    if (nixl::deviceMemViewGetBackend(ops_, mem_view, proxy_mvh) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to read proxy memory view before retirement";
    }

    runtime_->releaseMemView(proxy_mvh);
    nixl::deviceMemViewFree(ops_, mem_view);
}
} // namespace nixl::ucx
