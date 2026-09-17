/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H
#define NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H

#include <vector>
#include <span>
#include <cstring>
#include <memory>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <optional>
#include <mutex>
#include <unordered_map>

#include "backend/backend_engine.h"

#include "mem_list.h"
#include "rkey.h"
#include "ucx_utils.h"

class nixlUcxConnection : public nixlBackendConnMD {
    private:
        std::vector<std::unique_ptr<nixlUcxEp>> eps;

    public:
        [[nodiscard]] const std::unique_ptr<nixlUcxEp>& getEp(size_t ep_id) const noexcept {
            return eps[ep_id];
        }

    friend class nixlUcxEngine;
};

using ucx_connection_ptr_t = std::shared_ptr<nixlUcxConnection>;

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
    private:
        nixlUcxMem mem;
        nixl_blob_t rkeyStr;

    public:
        nixlUcxPrivateMetadata() : nixlBackendMD(true) {
        }

        [[nodiscard]] const std::string& get() const noexcept {
            return rkeyStr;
        }

        [[nodiscard]] const nixlUcxMem &
        getMem() const noexcept {
            return mem;
        }

    friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {
public:
    nixlUcxPublicMetadata() = delete;
    nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn, std::vector<nixl::ucx::rkey> &&rkeys);

    [[nodiscard]] const nixl::ucx::rkey &
    getRkey(const size_t id) const {
        return rkeys_[id];
    }

    const ucx_connection_ptr_t conn;

private:
    const std::vector<nixl::ucx::rkey> rkeys_;
};

class nixlUcxEngine : public nixlBackendEngine {
public:
    static std::unique_ptr<nixlUcxEngine>
    create(const nixlBackendInitParams &init_params);

    ~nixlUcxEngine();

    bool
    supportsRemote() const override {
        return true;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override;

    /* Object management */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;
    nixl_status_t
    getConnInfo(std::string &str) const override;
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent,
                       const std::string &remote_conn_info) override;

    nixl_status_t
    connect(const std::string &remote_agent) override;
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    // Data transfer
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    estimateXferCost(const nixl_xfer_op_t &operation,
                     const nixl_meta_dlist_t &local,
                     const nixl_meta_dlist_t &remote,
                     const std::string &remote_agent,
                     nixlBackendReqH *const &handle,
                     std::chrono::microseconds &duration,
                     std::chrono::microseconds &err_margin,
                     nixl_cost_t &method,
                     const nixl_opt_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    unsigned
    progress();

    void
    progressLoop();

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &,
                nixlMemViewH &,
                const nixl_opt_b_args_t * = nullptr) const override;

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &,
                nixlMemViewH &,
                const nixl_opt_b_args_t * = nullptr) const override;

    void releaseMemView(nixlMemViewH) const override;

protected:
    using worker_span_t = std::span<const std::unique_ptr<nixlUcxWorker>>;

    [[nodiscard]] worker_span_t
    getSharedWorkers() const {
        return {workers_.data(), numSharedWorkers_};
    }

    [[nodiscard]] worker_span_t
    getDedicatedWorkers() const {
        return {workers_.data() + numSharedWorkers_, workers_.size() - numSharedWorkers_};
    }

    [[nodiscard]] const std::unique_ptr<nixlUcxWorker> &
    getSharedWorker(size_t worker_id) const {
        if (worker_id >= numSharedWorkers_) [[unlikely]] {
            throw std::out_of_range("Worker ID out of range");
        }
        return workers_[worker_id];
    }

    [[nodiscard]] size_t
    getSharedWorkerId(const nixl_opt_b_args_t *opt_args = nullptr) const noexcept;

    [[nodiscard]] size_t
    getSharedWorkersSize() const {
        return numSharedWorkers_;
    }

    virtual void
    appendNotif(std::string &&remote_name, std::string &&msg);

    virtual nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const;

    nixlUcxEngine(const nixlBackendInitParams &init_params, size_t num_dedicated_workers = 0);

    notif_list_t notifList_;

private:
    // Memory management helpers
    nixl_status_t
    internalMDHelper(const nixl_blob_t &blob, const std::string &agent, nixlBackendMD *&output);

    // Notifications
    static ucs_status_t
    notifAmCb(void *arg,
              const void *header,
              size_t header_length,
              void *data,
              size_t length,
              const ucp_am_recv_param_t *param);

    [[nodiscard]] std::unique_ptr<std::string>
    buildNotif(const std::string &msg) const;

    [[nodiscard]] static nixl_status_t
    sendNotif(std::unique_ptr<std::string> &&msg, const nixlUcxEp &ep, nixlUcxReq *req);

    nixl_status_t
    notifSendPriv(const std::string &remote_agent,
                  const std::string &msg,
                  const nixlUcxEp &ep,
                  nixlUcxReq *req = nullptr) const;

    ucx_connection_ptr_t
    getConnection(const std::string &remote_agent) const;

    /* connection_mode=sockaddr helpers */

    /* Creates the local ucp_listener and returns the connection info blob to be
     * advertised to peers. Throws on misconfiguration. */
    std::string
    initSockaddrListener(nixl_b_params_t *custom_params);

    /* Called from the listener callback, in progress context of worker 0.
     * Returns true when the connection request has been accepted. */
    bool
    onConnRequest(ucp_conn_request_h conn_request);

    nixl_status_t
    connectSockaddrPeer(const std::string &remote_agent,
                        const std::string &remote_conn_info,
                        const ucx_connection_ptr_t &conn);

    /* Waits until the client/server wireup of every endpoint of conn has
     * completed, progressing the local workers (which also accepts the peer's
     * incoming connection requests, so two agents connecting to each other at
     * the same time make progress). Needed because ucp_ep_rkey_unpack() - and
     * hence loadRemoteMD() - requires a fully connected endpoint. */
    nixl_status_t
    waitConnected(const ucx_connection_ptr_t &conn, const std::string &remote_agent) const;

    void
    releaseRequests(std::vector<nixlUcxReq> &reqs, size_t from) const;

#ifdef HAVE_UCX_SGL_API
    nixl_status_t
    prepXferSgl(const nixl_meta_dlist_t &local,
                const nixl_meta_dlist_t &remote,
                nixlBackendReqH *handle) const;

    nixl_status_t
    sendXferSgl(nixlBackendReqH *handle) const;
#endif

    /**
     * Get the worker ID from the optional arguments.
     * Returns std::nullopt if the 'worker_id' option extraction fails.
     */
    [[nodiscard]] std::optional<size_t>
    getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept;

    /* UCX data */
    std::unique_ptr<nixlUcxContext> uc;
    std::vector<std::unique_ptr<nixlUcxWorker>> workers_;
    size_t numSharedWorkers_;
    std::string workerAddr;
    mutable std::atomic<size_t> sharedWorkerIndex_;
    const bool sglEnabled_;

    // Map of agent name to saved nixlUcxConnection info
    std::unordered_map<std::string, ucx_connection_ptr_t> remoteConnMap;

    /* Connection establishment mode and the blob returned by getConnInfo():
     * the local worker address in worker_address mode, the serialized listener
     * address in sockaddr mode. */
    nixl::ucx::conn_mode_t connMode_{nixl::ucx::conn_mode_t::WORKER_ADDRESS};
    std::string connInfo_;
    std::chrono::milliseconds connectTimeout_{30000};

    /* Server-side endpoints created from incoming connection requests.
     *
     * NIXL never sends on these endpoints - notifications and RMA always go
     * through the local client endpoints held in remoteConnMap - they only
     * exist to complete the UCX client/server wireup and to receive. They are
     * therefore not mapped to a remote agent name: they are released when the
     * last remote connection is dropped, and at engine destruction.
     *
     * Mutated from the listener callback, which runs in progress context (and
     * hence on the progress thread for the threaded engines), so it needs its
     * own lock rather than relying on the nixlAgent lock.
     *
     * Declared after workers_ so that they are destroyed before the workers.
     */
    mutable std::mutex acceptedEpsMutex_;
    std::vector<std::unique_ptr<nixlUcxEp>> acceptedEps_;

    /* Declared last: destroyed first, so no new connection request can arrive
     * while the accepted endpoints or the workers are being torn down. */
    std::unique_ptr<nixlUcxListener> listener_;
};

#endif
