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
#include <iostream>
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <deque>
#include <poll.h>
#include <optional>
#include <unordered_map>

#include "nixl.h"

#include "backend/backend_engine.h"
#include "common/nixl_time.h"

#include "mem_list.h"
#include "rkey.h"
#include "ucx_enums.h"
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
    struct VramStagingConfig {
        bool enabled = false;
        size_t chunkSize = 16 * 1024 * 1024;
        size_t slotsPerGpu = 4;
        bool forceProgressThread = true;
        size_t cudaCopyStreams = 1;
        size_t slotRequestWindow = 0;
        bool batchFlush = false;
        bool targetH2DWorker = false;
        bool sourceD2HPrefetch = false;
        bool localStaging = false;
        bool localStagingAutoEnabled = false;
        bool localStagingFallback = true;
        std::string localStagingShmDir = "/dev/shm/nixl";
    };

    [[nodiscard]] bool
    vramStagingEnabled() const noexcept {
        return vramStagingConfig_.enabled;
    }

    [[nodiscard]] const VramStagingConfig &
    vramStagingConfig() const noexcept {
        return vramStagingConfig_;
    }

    const std::vector<std::unique_ptr<nixlUcxWorker>> &
    getWorkers() const {
        return uws;
    }

    const std::unique_ptr<nixlUcxWorker> &
    getWorker(size_t worker_id) const {
        return uws[worker_id];
    }

    void
    stopStagedH2DWorker();

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

    nixl_status_t
    internalStagedMDHelper(const nixl_blob_t &blob,
                           const std::string &agent,
                           nixlBackendMD *&output);

    uint64_t
    nextStagedTransferId() const noexcept;

    nixl_status_t
    registerPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const;

    void
    unregisterPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const;

    void
    completePendingStagedSlotGrant(uint64_t transfer_id,
                                   uint64_t chunk_id,
                                   uint64_t slot_id,
                                   uint64_t lease_id,
                                   nixl_status_t status) const;

    void
    completePendingStagedReq(uint64_t transfer_id,
                             uint64_t chunk_id,
                             uint64_t lease_id,
                             nixl_status_t status) const;

    void
    registerStagedRegion(nixlBackendMD *metadata);

    void
    unregisterStagedRegion(nixlBackendMD *metadata);

    nixl_status_t
    postStagedWrite(const nixl_meta_dlist_t &local,
                    const nixl_meta_dlist_t &remote,
                    const std::string &remote_agent,
                    nixlBackendReqH *handle,
                    const nixl_opt_b_args_t *opt_args) const;

    nixl_status_t
    checkStagedXfer(nixlBackendReqH *handle) const;

    nixl_status_t
    sendStagedSlotReq(const std::string &remote_agent,
                      uint64_t transfer_id,
                      uint64_t chunk_id,
                      uintptr_t remote_gpu_addr,
                      uint64_t remote_gpu_dev,
                      size_t size,
                      const std::unique_ptr<nixlUcxEp> &ep,
                      nixlUcxReq *req) const;

    nixl_status_t
    sendStagedSlotGrant(const std::string &remote_agent,
                        uint64_t transfer_id,
                        uint64_t chunk_id,
                        uint64_t slot_id,
                        uint64_t lease_id,
                        nixl_status_t status,
                        ucp_ep_h reply_ep = nullptr) const;

    nixl_status_t
    sendStagedControlAm(const std::string &remote_agent,
                        ucp_ep_h reply_ep,
                        nixl::ucx::am_cb_op_t msg_id,
                        std::string *buffer) const;

    nixl_status_t
    sendStagedSlotRelease(const std::string &remote_agent,
                          uint64_t transfer_id,
                          uint64_t chunk_id,
                          uint64_t slot_id,
                          uint64_t lease_id,
                          uintptr_t remote_gpu_addr,
                          uint64_t remote_gpu_dev,
                          size_t size) const;

    nixl_status_t
    sendStagedWriteReady(const std::string &remote_agent,
                         uint64_t transfer_id,
                         uint64_t chunk_id,
                         uint64_t remote_slot_id,
                         uint64_t lease_id,
                         uintptr_t remote_gpu_addr,
                         uint64_t remote_gpu_dev,
                         size_t size,
                         const std::unique_ptr<nixlUcxEp> &ep,
                         nixlUcxReq *req) const;

    nixl_status_t
    sendStagedLocalWriteReady(const std::string &remote_agent,
                              uint64_t transfer_id,
                              uint64_t chunk_id,
                              uint64_t source_region_id,
                              const std::string &source_region_cookie,
                              uint64_t source_slot_id,
                              uint64_t source_slot_generation,
                              const std::string &source_shared_path,
                              size_t source_slot_offset,
                              size_t source_mapping_size,
                              uintptr_t remote_gpu_addr,
                              uint64_t remote_gpu_dev,
                              size_t size,
                              const std::unique_ptr<nixlUcxEp> &ep,
                              nixlUcxReq *req) const;

    nixl_status_t
    sendStagedAck(const std::string &remote_agent,
                  uint64_t transfer_id,
                  uint64_t chunk_id,
                  uint64_t lease_id,
                  nixl_status_t status,
                  ucp_ep_h reply_ep = nullptr) const;

    nixl_status_t
    handleStagedSlotReq(const nixl_blob_t &message, ucp_ep_h reply_ep = nullptr) const;

    nixl_status_t
    handleStagedSlotRelease(const nixl_blob_t &message) const;

    nixl_status_t
    handleStagedWriteReady(const nixl_blob_t &message, ucp_ep_h reply_ep = nullptr) const;

    nixl_status_t
    handleStagedLocalWriteReady(const nixl_blob_t &message, ucp_ep_h reply_ep = nullptr) const;

    struct StagedH2DTask {
        nixlBackendMD *region = nullptr;
        void *hostAddr = nullptr;
        std::string remoteAgent;
        ucp_ep_h replyEp = nullptr;
        uint64_t transferId = 0;
        uint64_t chunkId = 0;
        uint64_t slotId = 0;
        uint64_t leaseId = 0;
        uintptr_t gpuAddr = 0;
        uint64_t gpuDev = 0;
        size_t size = 0;
        bool profileEnabled = false;
        uint64_t callbackUs = 0;
    };

    struct LocalSharedAttachment {
        std::string path;
        std::string remoteAgent;
        void *base = nullptr;
        size_t mappingSize = 0;
        int fd = -1;
        bool hostRegistered = false;
    };

    struct LocalSharedRegionInfo {
        std::string remoteAgent;
        uint64_t regionId = 0;
        std::string regionCookie;
        std::string sharedPath;
        size_t mappingSize = 0;
        size_t slotSize = 0;
        size_t slotCount = 0;
        size_t refCount = 0;
    };

    void
    startStagedH2DWorker();

    [[nodiscard]] nixl_status_t
    enqueueStagedH2D(StagedH2DTask &&task) const;

    void
    stagedH2DWorkerLoop() const;

    void
    registerLocalSharedRegion(const std::string &remote_agent,
                              const nixlBackendMD *metadata) const;

    void
    unregisterLocalSharedRegion(const nixlBackendMD *metadata) const;

    [[nodiscard]] bool
    validateLocalSharedReady(const std::string &remote_agent,
                             uint64_t region_id,
                             const std::string &region_cookie,
                             const std::string &shared_path,
                             uint64_t slot_id,
                             size_t slot_offset,
                             size_t mapping_size,
                             size_t size) const;

    nixl_status_t
    getLocalSharedAttachment(const std::string &remote_agent,
                             const std::string &path,
                             size_t mapping_size,
                             std::shared_ptr<LocalSharedAttachment> &attachment) const;

    void
    cleanupLocalSharedAttachments();

    void
    cleanupLocalSharedAttachmentsForAgent(const std::string &remote_agent);

    void
    cleanupLocalSharedAttachmentPath(const std::string &path) const;

    static void
    releaseLocalSharedAttachment(std::shared_ptr<LocalSharedAttachment> &attachment);

    static ucs_status_t
    stagedSlotReqAmCb(void *arg,
                      const void *header,
                      size_t header_length,
                      void *data,
                      size_t length,
                      const ucp_am_recv_param_t *param);

    static ucs_status_t
    stagedSlotGrantAmCb(void *arg,
                        const void *header,
                        size_t header_length,
                        void *data,
                        size_t length,
                        const ucp_am_recv_param_t *param);

    static ucs_status_t
    stagedSlotReleaseAmCb(void *arg,
                          const void *header,
                          size_t header_length,
                          void *data,
                          size_t length,
                          const ucp_am_recv_param_t *param);

    static ucs_status_t
    stagedWriteReadyAmCb(void *arg,
                         const void *header,
                         size_t header_length,
                         void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param);

    static ucs_status_t
    stagedLocalWriteReadyAmCb(void *arg,
                              const void *header,
                              size_t header_length,
                              void *data,
                              size_t length,
                              const ucp_am_recv_param_t *param);

    static ucs_status_t
    stagedAckAmCb(void *arg,
                  const void *header,
                  size_t header_length,
                  void *data,
                  size_t length,
                  const ucp_am_recv_param_t *param);

    static VramStagingConfig
    makeVramStagingConfig(const nixl_b_params_t *custom_params);

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
    VramStagingConfig vramStagingConfig_;
    mutable std::atomic<uint64_t> nextStagedTransferId_;
    mutable std::mutex stagedReqMutex_;
    mutable std::unordered_map<uint64_t, nixlBackendReqH *> pendingStagedReqs_;
    mutable std::mutex stagedRegionMutex_;
    std::vector<nixlBackendMD *> stagedRegions_;
    mutable std::atomic<uint64_t> stagedProfileTargetReadyCount_{0};
    mutable std::atomic<uint64_t> stagedProfileTargetBytes_{0};
    mutable std::atomic<uint64_t> stagedProfileTargetH2DUs_{0};
    mutable std::atomic<uint64_t> stagedProfileTargetCallbackUs_{0};
    mutable std::atomic<uint64_t> stagedProfileLocalReadyCount_{0};
    mutable std::atomic<uint64_t> stagedProfileLocalErrors_{0};
    mutable std::atomic<uint64_t> stagedProfileLocalBytes_{0};
    mutable std::atomic<uint64_t> stagedProfileLocalH2DUs_{0};
    mutable std::atomic<uint64_t> stagedProfileLocalCallbackUs_{0};
    mutable std::atomic<uint64_t> localSharedAttachCacheHits_{0};
    mutable std::atomic<uint64_t> localSharedAttachCacheMisses_{0};
    mutable std::atomic<uint64_t> localSharedAttachFailures_{0};
    mutable std::atomic<uint64_t> localSharedAttachUs_{0};
    mutable std::mutex stagedH2DMutex_;
    mutable std::condition_variable stagedH2DCv_;
    mutable std::deque<StagedH2DTask> stagedH2DQueue_;
    mutable bool stagedH2DStop_ = false;
    std::thread stagedH2DThread_;
    mutable std::mutex localSharedAttachMutex_;
    mutable std::unordered_map<std::string, std::shared_ptr<LocalSharedAttachment>>
        localSharedAttachments_;
    mutable std::unordered_map<std::string, LocalSharedRegionInfo> localSharedRegions_;

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
