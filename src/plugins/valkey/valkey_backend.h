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

#ifndef NIXL_SRC_PLUGINS_VALKEY_VALKEY_BACKEND_H
#define NIXL_SRC_PLUGINS_VALKEY_VALKEY_BACKEND_H

#include "backend/backend_engine.h"
#include "valkey_client.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

inline constexpr const char *VALKEY_PLUGIN_NAME = "VALKEY";
inline constexpr const char *VALKEY_PLUGIN_VERSION = "0.1.0";

class nixlValkeyReqH;

/**
 * Storage backend for Valkey with the valkey-large-object module, through
 * GLIDE. Payloads move by server-initiated RDMA (BLOB.RDMA_SET/GET).
 */
class nixlValkeyEngine : public nixlBackendEngine {
public:
    explicit nixlValkeyEngine(const nixlBackendInitParams *init_params);
    ~nixlValkeyEngine() override;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    // Also the plugin's list (valkey_plugin.cpp).
    static nixl_mem_list_t
    supportedMems() {
        return {DRAM_SEG, OBJ_SEG};
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return supportedMems();
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;
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

private:
    using client = nixl::valkey::client;

    // One descriptor (a whole object), run by a worker.
    struct job {
        nixl_xfer_op_t op;
        uint64_t regionId; // local region holding [addr, addr + len)
        std::string key;
        uintptr_t addr;
        size_t len;
        nixlValkeyReqH *req;
    };

    struct memRegion {
        uintptr_t addr;
        size_t len;
    };

    // regions is shared with registerMem/deregisterMem on the calling thread
    // and guarded by lock; conn is written by the worker thread only (others
    // read it under lock). Neither is locked during a transfer. deadline is the
    // running transfer's cancel time for the watchdog (steady_clock ns; 0 = idle).
    struct worker {
        std::mutex lock;
        client *conn = nullptr;
        std::unordered_map<uint64_t, glide::RdmaRegion *> regions; // region id -> registration
        std::atomic<int64_t> deadline{0};
    };

    std::string
    objectKey(const nixlBlobDesc &desc) const;

    // Registration
    nixl_status_t
    registerDram(const nixlBlobDesc &mem, nixlBackendMD *&out);
    void
    dropRegistrations(uint64_t id) const;
    std::vector<worker *>
    workers() const;

    // Workers
    void
    workerLoop(size_t idx);
    void
    watchdogLoop();
    nixl_status_t
    runJob(worker &w, const job &j) const;
    nixl_status_t
    regionFor(worker &w, uint64_t id, const memRegion &reg, glide::RdmaRegion *&region) const;
    void
    dropConn(worker &w) const;

    const nixlValkeyConfig config_;

    // One blocking client per worker thread: num_workers is the transfer
    // concurrency.
    std::vector<std::thread> workers_;
    std::thread watchdog_; // cancels transfers past transfer_timeout_ms
    mutable std::mutex queueMutex_;
    mutable std::condition_variable queueCv_;
    mutable std::deque<job> queue_;
    bool stop_ = false;
    std::vector<worker *> workerState_; // protected by queueMutex_
    std::condition_variable startCv_;

    // DRAM regions. registerMem registers a region with every worker's client
    // from the calling thread and deregisterMem frees those
    // registrations the same way, without waiting for transfers in progress:
    // callers do not deregister memory a transfer still uses.
    mutable std::mutex regMutex_;
    std::unordered_map<uint64_t, memRegion> liveRegions_;
    uint64_t nextRegion_ = 1;

    // Non-RDMA client for queryMem.
    mutable std::mutex ctrlMutex_;
    client *ctrl_ = nullptr;
};

#endif // NIXL_SRC_PLUGINS_VALKEY_VALKEY_BACKEND_H
