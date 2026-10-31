/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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
#ifndef NIXL_SRC_PLUGINS_RADOS_RADOS_BACKEND_H
#define NIXL_SRC_PLUGINS_RADOS_RADOS_BACKEND_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/backend_engine.h"
#include "rados_api.h"
#include "rados_config.h"

namespace nixl::rados {

class statusWaitTimer {
public:
    statusWaitTimer(const statusWaitTimer &) = delete;
    statusWaitTimer &
    operator=(const statusWaitTimer &) = delete;
    virtual ~statusWaitTimer() = default;

    // Block for the budget; the caller holds no registry or request lock.
    virtual void
    sleep(uint64_t budget_us) = 0;

protected:
    statusWaitTimer() = default;
};

} // namespace nixl::rados

class nixlRadosObjectMD : public nixlBackendMD {
public:
    nixlRadosObjectMD(uint64_t dev_id, std::string object_key)
        : nixlBackendMD(true),
          devId(dev_id),
          objectKey(std::move(object_key)) {}

    nixlRadosObjectMD(const nixlRadosObjectMD &) = delete;
    nixlRadosObjectMD &
    operator=(const nixlRadosObjectMD &) = delete;

    const uint64_t devId;
    const std::string objectKey;
};

class directRequestState;

class nixlRadosEngine : public nixlBackendEngine {
public:
    explicit nixlRadosEngine(const nixlBackendInitParams *init_params);

    nixlRadosEngine(const nixlBackendInitParams *init_params,
                    std::unique_ptr<nixl::rados::radosApi> api,
                    std::unique_ptr<nixl::rados::statusWaitTimer> status_wait_timer = nullptr);

    ~nixlRadosEngine() override;

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

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, OBJ_SEG};
    }

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
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

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
    void
    initialize(const nixl::rados::backendConfig &config);

    std::shared_ptr<directRequestState>
    lookupActiveLocked(nixlBackendReqH *handle) const;

    void
    drainRequestState(const std::shared_ptr<directRequestState> &state) const;

    std::unique_ptr<nixl::rados::radosApi> api_;
    uint64_t statusWaitBudgetUs_ = 0;
    std::unique_ptr<nixl::rados::statusWaitTimer> statusWaitTimer_;

    struct radosClient {
        librados::Rados rados;
        librados::IoCtx ioctx;
    };

    // Fixed in place: Rados precedes IoCtx so each context closes before its cluster shuts down.
    mutable radosClient data_;
    // Stat-only query client; declared after data so it dies first, after transfers drain.
    mutable radosClient meta_;

    mutable std::mutex objectsMutex_;
    std::unordered_map<uint64_t, std::string> objects_;

    mutable std::mutex activeMutex_;
    std::atomic<bool> shuttingDown_{false};
    mutable std::unordered_map<nixlBackendReqH *, std::shared_ptr<directRequestState>> handles_;
};

#endif
