/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <optional>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "ucx/rkey.h"
#include "ucx/ucx_utils.h"

enum ucx_cb_op_t { NOTIF_STR };

class nixlUcxConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
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

    friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {
public:
    nixlUcxPublicMetadata() : nixlBackendMD(false) {}

    [[nodiscard]] const nixl::ucx::rkey &
    getRkey(size_t id) const {
        return *rkeys_[id];
    }

    void
    addRkey(const nixlUcxEp &ep, const void *rkey_buffer) {
        rkeys_.emplace_back(std::make_unique<nixl::ucx::rkey>(ep, rkey_buffer));
    }

    ucx_connection_ptr_t conn;

private:
    std::vector<std::unique_ptr<nixl::ucx::rkey>> rkeys_;
};

// Forward declaration of CUDA context
// It is only visible in ucx_backend.cpp to ensure that
// HAVE_CUDA works properly
// Once we will introduce static config (i.e. config.h) that
// will be part of NIXL installation - we can have
// HAVE_CUDA in h-files
class nixlUcxCudaCtx;
class nixlUcxCudaDevicePrimaryCtx;
using nixlUcxCudaDevicePrimaryCtxPtr = std::shared_ptr<nixlUcxCudaDevicePrimaryCtx>;

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

    nixl_status_t
    createGpuXferReq(const nixlBackendReqH &req_hndl,
                     const nixl_meta_dlist_t &local_descs,
                     const nixl_meta_dlist_t &remote_descs,
                     nixlGpuXferReqH &gpu_req_hndl) const override;

    void
    releaseGpuXferReq(nixlGpuXferReqH gpu_req_hndl) const override;

    nixl_status_t
    getGpuSignalSize(size_t &signal_size) const override;

    nixl_status_t
    prepGpuSignal(const nixlBackendMD &meta,
                  void *signal,
                  const nixl_opt_b_args_t *opt_args = nullptr) const override;

    int
    progress();

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

    // public function for UCX worker to mark connections as connected
    nixl_status_t
    checkConn(const std::string &remote_agent);

private:
    // Helper to extract worker_id from opt_args->customParam or nullopt if not found
    [[nodiscard]] std::optional<size_t>
    getWorkerIdFromOptArgs(const nixl_opt_b_args_t *opt_args) const noexcept;

protected:
    const std::vector<std::unique_ptr<nixlUcxWorker>> &
    getWorkers() const {
        return uws;
    }

    const std::unique_ptr<nixlUcxWorker> &
    getWorker(size_t worker_id) const {
        return uws[worker_id];
    }

    size_t
    getWorkerId() const;

    virtual size_t
    getSharedWorkersSize() const {
        return uws.size();
    }

    void
    getNotifsImpl(notif_list_t &notif_list);

    virtual int
    vramApplyCtx();

    virtual void
    appendNotif(std::string remote_name, std::string msg);

    virtual nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const;

    nixlUcxEngine(const nixlBackendInitParams &init_params);

private:
    void
    vramInitCtx();
    void
    vramFiniCtx();
    int
    vramUpdateCtx(void *address, uint64_t dev_id, bool &restart_reqd);

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

    nixl_status_t
    notifSendPriv(const std::string &remote_agent,
                  const std::string &msg,
                  nixlUcxReq &req,
                  const std::unique_ptr<nixlUcxEp> &ep) const;

    ucx_connection_ptr_t
    getConnection(const std::string &remote_agent) const;

    /* UCX data */
    std::unique_ptr<nixlUcxContext> uc;
    std::vector<std::unique_ptr<nixlUcxWorker>> uws;
    std::string workerAddr;
    mutable std::atomic<size_t> sharedWorkerIndex_;

    /* CUDA data*/
    std::unique_ptr<nixlUcxCudaCtx> cudaCtx; // Context matching specific device
    bool cuda_addr_wa;
    mutable std::optional<size_t> gpuSignalSize_;

    // Context to use when current context is missing
    nixlUcxCudaDevicePrimaryCtxPtr m_cudaPrimaryCtx;

    /* Notifications */
    notif_list_t notifMainList;

    // Map of agent name to saved nixlUcxConnection info
    std::unordered_map<std::string, ucx_connection_ptr_t, std::hash<std::string>, strEqual>
        remoteConnMap;
};

class nixlUcxThread;

/**
 * Represents an engine with a single progress thread for all shared workers
 */
class nixlUcxThreadEngine : public nixlUcxEngine {
public:
    nixlUcxThreadEngine(const nixlBackendInitParams &init_params);
    ~nixlUcxThreadEngine();

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

protected:
    int
    vramApplyCtx() override;

    void
    appendNotif(std::string remote_name, std::string msg) override;

private:
    std::unique_ptr<nixlUcxThread> thread_;
    std::mutex notifMtx_;
    notif_list_t notifPthr_;
};

namespace asio {
class io_context;
}

class nixlUcxThreadPoolEngine : public nixlUcxEngine {
public:
    nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params);
    ~nixlUcxThreadPoolEngine();

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    size_t
    getSharedWorkersSize() const override {
        return numSharedWorkers_;
    }

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

protected:
    int
    vramApplyCtx() override;

    void
    appendNotif(std::string remote_name, std::string msg) override;

    nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const override;

private:
    std::unique_ptr<asio::io_context> io_;
    std::unique_ptr<nixlUcxThread> sharedThread_;
    std::vector<std::unique_ptr<nixlUcxThread>> dedicatedThreads_;
    size_t numSharedWorkers_;
    std::mutex notifMutex_;
    notif_list_t notifThread_;
    size_t splitBatchSize_;
};

#endif
