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

#ifndef NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_BACKEND_H
#define NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_BACKEND_H

#include <ostream>
#include <string>
#include <atomic>
#include <vector>
#include <memory>
#include <sys/uio.h> // for struct iovec
#include "backend/backend_engine.h"
#include "doca_memos_progress_engine.h"

// Forward declarations for DOCA types
struct doca_kvdev;
struct doca_nvme_kernel_kvdev;

// Static cap on object-key size. DOCA does not expose a compile-time maximum,
// so this is set to the current spec value (128-bit) and verified at runtime
// via doca_kvdev_get_max_key_len() during engine init.
inline constexpr size_t DOCA_MEMOS_MAX_OBJECT_KEY_LEN = 16;

// Forward declaration for backend init params
class nixlBackendInitParams;

// Default-constructed instances have keyLen = 0, which DOCA will reject at
// submit time. Callers must populate via nixlDocaMemosEngine::resolveMemosKey()
// (or convertToMemosKey()) before handing the key to any DOCA API.
struct docaMemosKey {
    uint8_t key[DOCA_MEMOS_MAX_OBJECT_KEY_LEN] = {};
    uint16_t keyLen = 0;
};

std::ostream &
operator<<(std::ostream &os, const docaMemosKey &k);

// Per-task scratch passed to DOCA via task_user_data and handed back to the
// completion / error callbacks. Lifetime invariants:
//   - reqH is a non-owning pointer; the engine guarantees the request handle
//     outlives every task it submitted (see nixlDocaMemosBackendReqH and the
//     destructors in nixlNoThreadProgressEngine / nixlThreadedProgressEngine).
//   - Instances live inside reqH->taskContexts_, so the address published to
//     DOCA stays valid as long as that vector is not resized after submit.
struct docaMemosTaskContext {
    class nixlDocaMemosBackendReqH *reqH;
    int taskIndex;
};

class nixlDocaMemosBackendReqH : public nixlBackendReqH {
public:
    explicit nixlDocaMemosBackendReqH(int num_tasks) : totalTasks_(num_tasks) {}

    ~nixlDocaMemosBackendReqH() = default;

    // Move construction is needed so this type satisfies MoveInsertable for
    // std::vector (callers reserve() then emplace_back()). Move assignment is
    // deleted because reassigning an in-flight request handle is never valid.
    nixlDocaMemosBackendReqH(nixlDocaMemosBackendReqH &&other) noexcept
        : totalTasks_(other.totalTasks_),
          submittedTasks_(other.submittedTasks_),
          completedTasks_(other.completedTasks_),
          allTasksCompleted_(other.allTasksCompleted_.load(std::memory_order_acquire)),
          cancelled_(other.cancelled_.load(std::memory_order_acquire)),
          overallStatus_(other.overallStatus_),
          taskResult_(other.taskResult_.load(std::memory_order_acquire)),
          isExistQuery_(other.isExistQuery_),
          ignoreNotFound_(other.ignoreNotFound_),
          nextDescriptorIndex_(other.nextDescriptorIndex_),
          isPending_(other.isPending_),
          storedOperation_(other.storedOperation_),
          storedLocal_(std::move(other.storedLocal_)),
          storedRemote_(std::move(other.storedRemote_)),
          valueIovecs_(std::move(other.valueIovecs_)),
          objectKeys_(std::move(other.objectKeys_)),
          taskContexts_(std::move(other.taskContexts_)) {}

    nixlDocaMemosBackendReqH(const nixlDocaMemosBackendReqH &) = delete;
    nixlDocaMemosBackendReqH &
    operator=(const nixlDocaMemosBackendReqH &) = delete;
    nixlDocaMemosBackendReqH &
    operator=(nixlDocaMemosBackendReqH &&) = delete;

    bool
    allTasksCompleted() const {
        return allTasksCompleted_.load(std::memory_order_acquire);
    }

    nixl_status_t
    getOverallStatus() const {
        return overallStatus_;
    }

private:
    // Engines and their static helpers are the sole owners of the bookkeeping
    // below. Keeping these private prevents accidental external mutation while
    // letting the engines manipulate request state directly.
    friend class nixlDocaMemosEngine;
    friend class nixlDocaMemosProgressEngine;
    friend class nixlNoThreadProgressEngine;
    friend class nixlThreadedProgressEngine;

    int totalTasks_ = 0;
    int submittedTasks_ = 0;
    int completedTasks_ = 0;
    std::atomic<bool> allTasksCompleted_{false};
    std::atomic<bool> cancelled_{false};
    nixl_status_t overallStatus_ = NIXL_IN_PROG;
    std::atomic<int> taskResult_{0};
    bool isExistQuery_ = false;
    bool ignoreNotFound_ = false;

    int nextDescriptorIndex_ = 0;
    bool isPending_ = false;

    nixl_xfer_op_t storedOperation_ = NIXL_WRITE;
    std::unique_ptr<nixl_meta_dlist_t> storedLocal_;
    std::unique_ptr<nixl_meta_dlist_t> storedRemote_;

    std::vector<struct iovec> valueIovecs_;
    std::vector<docaMemosKey> objectKeys_;
    std::vector<docaMemosTaskContext> taskContexts_;
};

class nixlDocaMemosEngine : public nixlBackendEngine {
public:
    nixlDocaMemosEngine(const nixlBackendInitParams *init_params);
    ~nixlDocaMemosEngine() override;

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
        return {OBJ_SEG, DRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
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

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    // Stateless helpers used by both the engine and the progress engine to turn
    // a registration's metaInfo blob into a docaMemosKey. Exposed as static
    // methods (rather than free functions) to keep them out of the plugin's
    // exported global namespace.
    static bool
    convertToMemosKey(const std::string &meta_info, docaMemosKey &key);
    static bool
    resolveMemosKey(uint64_t dev_id, const std::string &meta_info, docaMemosKey &key);

private:
    struct doca_kvdev *kvdev_ = nullptr;
    struct doca_nvme_kernel_kvdev *nvmeKvdev_ = nullptr;
    std::unique_ptr<nixlDocaMemosProgressEngine> progressEngine_;

    static constexpr const char *kDefaultNguid = "00000000000000000000000000000000";
    static constexpr uint32_t kDefaultNumTasks = 8192;

    std::string deviceName_;
    uint32_t numTasks_ = kDefaultNumTasks;
    std::string nguid_ = kDefaultNguid;
    bool queryMemAssumeSuccess_ = true;
    bool ignoreReadNotFound_ = false;

    nixl_status_t
    parseInitParams(const nixl_b_params_t *params);

    nixl_status_t
    initDocaDevice();

    nixl_status_t
    createProgressEngine(const nixlBackendInitParams *init_params);

    void
    cleanupDocaResources();
};

#endif // NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_BACKEND_H
