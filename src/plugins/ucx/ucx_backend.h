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
#include <cstring>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <memory>
#include <tuple>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <optional>

#include "nixl.h"

#include "backend/backend_engine.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"

#include "mem_list.h"
#include "rkey.h"
#include "ucx_enums.h"
#include "ucx_utils.h"

struct nixlUcxBackendRecvH : public nixlBackendReqH {
    const nixl_meta_dlist_t &local;

    nixl_status_t status = NIXL_ERR_NOT_POSTED;

    nixlUcxBackendRecvH(const nixl_meta_dlist_t &local)
        : local(local) {}
};

struct nixlUcxRecvKey {
    std::string remoteAgent;
    std::string sendRecvTag;

    [[nodiscard]] auto
    tie() const noexcept {
        return std::tie(remoteAgent, sendRecvTag);
    }
};

[[nodiscard]] bool inline
operator<(const nixlUcxRecvKey &l, const nixlUcxRecvKey &r) noexcept {
    return l.tie() < r.tie();
}

[[nodiscard]] bool inline
operator==(const nixlUcxRecvKey &l, const nixlUcxRecvKey &r) noexcept {
    return l.tie() == r.tie();
}

struct nixlUcxRecvValue {
    nixlUcxBackendRecvH *handle = nullptr;
    std::optional<std::string> eager;

    nixlUcxRecvValue() noexcept = default;
};

class nixlUcxRecvMap {
public:
    nixlUcxRecvMap() = default;

    [[nodiscard]] nixl_status_t
    postRecv(const std::string &remote,
             const std::string &tag,
             nixlUcxBackendRecvH *handle) {
        NIXL_ASSERT(handle);
        NIXL_ASSERT(handle->local.descCount() == 1); // TODO: Generalize

        const std::lock_guard lg(mutex_);
        const auto [iter, inserted] = map_.try_emplace({remote, tag});

        if (inserted) {
            iter->second.handle = handle;
            handle->status = NIXL_IN_PROG;
            return handle->status;
        }

        if (iter->second.handle) {
            // TODO: Handle repost
            handle->status = NIXL_ERR_REPOST_ACTIVE;
            return handle->status;
        }

        if (!iter->second.eager) {
            // TODO: Handle rndv
            std::terminate();
        }

        if (iter->second.eager->size() != handle->local[0].len) {
            handle->status = NIXL_ERR_MISMATCH;
            return handle->status;
        }

        // TODO: Handle device memory etc.
        // TODO: Move copy outside of lock
        memcpy(handle->local[0], *iter->second.eager);
        handle->status = NIXL_SUCCESS;
        map_.erase(iter);
        return handle->status;
    }

    void
    recvEager(const std::string &remote,
              const std::string &tag,
              const std::string_view payload) {
        const std::lock_guard lg(mutex_);
        const auto [iter, inserted] = map_.try_emplace({remote, tag});

        if (inserted) {
            iter->second.eager.emplace(payload);
            return;
        }

        NIXL_ASSERT(iter->second.handle != nullptr);
        NIXL_ASSERT(iter->second.handle->local.descCount() == 1); // TODO: Generalise

        if (payload.size() != iter->second.handle->local[0].len) {
            iter->second.handle->status = NIXL_ERR_MISMATCH;
            // TODO: Log error?
            // TODO: Make progress how?
            return;
        }

        // TODO: Handle device memory etc.
        // TODO: Move copy outside of lock
        memcpy(iter->second.handle->local[0], payload);
        iter->second.handle->status = NIXL_SUCCESS;
        map_.erase(iter);
        // TODO: Make progress how?
    }

private:
    std::mutex mutex_;
    std::map<nixlUcxRecvKey, nixlUcxRecvValue, std::less<void>> map_;

    static void
    memcpy(const nixlMetaDesc &desc, const std::string_view view) {
        std::memcpy(reinterpret_cast<void *>(desc.addr), view.data(), view.size());
    }
};

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

    bool
    supportsSendRecv() const override {
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
    prepTagXfer(nixl_xfer_op_t operation,
                const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args = nullptr
                ) const override;

    nixl_status_t
    postTagXfer(nixl_xfer_op_t operation,
                const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args = nullptr
                ) const override;

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

    // public function for UCX worker to mark connections as connected
    nixl_status_t
    checkConn(const std::string &remote_agent);

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
    const std::vector<std::unique_ptr<nixlUcxWorker>> &
    getWorkers() const {
        return uws;
    }

    const std::unique_ptr<nixlUcxWorker> &
    getWorker(size_t worker_id) const {
        return uws[worker_id];
    }

    [[nodiscard]] size_t
    getWorkerId(const nixl_opt_b_args_t *opt_args = nullptr) const noexcept;

    virtual size_t
    getSharedWorkersSize() const {
        return uws.size();
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

    nixlUcxEngine(const nixlBackendInitParams &init_params);

    notif_list_t notifList_;

private:
    // Memory management helpers
    nixl_status_t
    internalMDHelper(const nixl_blob_t &blob, const std::string &agent, nixlBackendMD *&output);

    // TOOD: virtual for locking in derived classes
    void
    recvAmEager(const std::string &remote,
                const std::string &tag,
                const void *data,
                std::size_t size);

    static ucs_status_t
    recvAmCb(void *arg,
             const void *header,
             size_t header_length,
             void *data,
             size_t length,
             const ucp_am_recv_param_t *param);

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
                  const std::unique_ptr<nixlUcxEp> &ep,
                  nixlUcxReq *req = nullptr) const;

    ucx_connection_ptr_t
    getConnection(const std::string &remote_agent) const;

    struct batchResult {
        nixl_status_t status;
        size_t size;
        nixlUcxReq req;
    };

    static batchResult
    sendXferRangeBatch(nixlUcxEp &ep,
                       nixl_xfer_op_t operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       size_t worker_id,
                       size_t start_idx,
                       size_t end_idx);

    nixl_status_t
    prepTagSend(const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args
                ) const;

    nixl_status_t
    prepTagRecv(const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args
                ) const;

    nixl_status_t
    postTagSend(const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args
                ) const;

    nixl_status_t
    postTagRecv(const nixl_meta_dlist_t &local,
                const std::string &tag,
                const std::string &remote_agent,
                nixlBackendReqH* &handle,
                const nixl_opt_b_args_t *opt_args
                ) const;

    /**
     * Get the worker ID from the optional arguments.
     * Returns std::nullopt if the 'worker_id' option extraction fails.
     */
    [[nodiscard]] std::optional<size_t>
    getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept;

    /* UCX data */
    std::unique_ptr<nixlUcxContext> uc;
    std::vector<std::unique_ptr<nixlUcxWorker>> uws;
    std::string workerAddr;
    mutable std::atomic<size_t> sharedWorkerIndex_;

    mutable nixlUcxRecvMap recvMap_;

    // Map of agent name to saved nixlUcxConnection info
    std::unordered_map<std::string, ucx_connection_ptr_t> remoteConnMap;
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
    void
    appendNotif(std::string &&remote_name, std::string &&msg) override;

private:
    std::unique_ptr<nixlUcxThread> thread_;
    std::mutex notifMutex_;
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
    void
    appendNotif(std::string &&remote_name, std::string &&msg) override;

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
    size_t splitBatchSize_;
};

#endif
