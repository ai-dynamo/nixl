/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 DataDirect Networks, Inc.
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

#ifndef NIXL_SRC_PLUGINS_INFINIA_INFINIA_BACKEND_H
#define NIXL_SRC_PLUGINS_INFINIA_INFINIA_BACKEND_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "backend/backend_engine.h"

#include <red/red_status.h>
#include <red/red_async.hpp>

#include "infinia_client.h"

// INFINIA default configuration values
inline constexpr const char* INFINIA_DEFAULT_CLUSTER = "cluster1";
inline constexpr const char* INFINIA_DEFAULT_TENANT = "red";
inline constexpr const char* INFINIA_DEFAULT_SUBTENANT = "red";
inline constexpr const char* INFINIA_DEFAULT_DATASET = "nixl";
inline constexpr int INFINIA_DEFAULT_STHREADS = 8;
inline constexpr int INFINIA_DEFAULT_BUFFERS = 512;
inline constexpr int INFINIA_DEFAULT_RING_ENTRIES = 512;
inline constexpr const char* INFINIA_DEFAULT_COREMASK = "0x2";

// Batch executor configuration defaults
inline constexpr int INFINIA_DEFAULT_MAX_BATCH_SIZE = 64;
inline constexpr int INFINIA_DEFAULT_MAX_CONCURRENT_BATCHES = 4;
inline constexpr int INFINIA_DEFAULT_WORKER_THREADS = 0;  // 0 = auto-detect
inline constexpr bool INFINIA_DEFAULT_AUTO_TUNE = true;

// RED Client Environment variable strings
inline constexpr const char* RED_CLUSTER_ENV = "RED_CLUSTER";
inline constexpr const char* RED_TENANT_ENV = "RED_TENANT";
inline constexpr const char* RED_DATASET_ENV = "RED_DATASET";

// Forward declarations
class nixlInfiniaBackendReqH;

/**
 * @brief Infinia backend engine implementation
 *
 * This class implements the nixlBackendEngine interface for Infinia storage systems.
 * It provides support for high-performance data transfers using Infinia's native APIs.
 */
class infinia_engine : public nixlBackendEngine {
private:
    // Private members for Infinia-specific configuration
    std::string infinia_cluster_;
    std::string infinia_tenant_;
    std::string infinia_subtenant_;
    std::string infinia_dataset_;
    uint32_t    infinia_sthreads_;
    uint32_t    infinia_num_buffers_;
    uint32_t    infinia_num_ring_entries_;
    std::string infinia_coremasks_;
    bool infinia_coremasks_set_;
    bool initialized_;

    std::shared_ptr<InfiniaClient> client_;

    mutable red_api_user_t  red_user;

    // Map to store devId -> key mapping for registered memory
    std::unordered_map<uint64_t, std::string> devid_to_key_map_;

    // Batch task configuration (passed to request handles)
    red_async::rae_batch_config_t batch_config_;

    // Helper methods
    [[nodiscard]] bool validateTransferParams(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent) const;

public:
    /**
     * @brief Constructor for Infinia engine
     * @param init_params Initialization parameters from the backend framework
     */
    explicit infinia_engine(const nixlBackendInitParams* init_params);

    /**
     * @brief Destructor
     */
    virtual ~infinia_engine();

    // Backend capability methods
    [[nodiscard]] bool supportsRemote() const noexcept override {
        return false;  // Infinia supports remote operations
    }

    [[nodiscard]] bool supportsLocal() const noexcept override {
        return true;  // Infinia supports local operations
    }

    [[nodiscard]] bool supportsNotif() const noexcept override {
        return false;  // Notifications not implemented yet
    }

    [[nodiscard]] bool supportsProgTh() const noexcept {
        return false;  // Progress thread not implemented yet
    }

    [[nodiscard]] nixl_mem_list_t getSupportedMems() const noexcept override {
        return {VRAM_SEG, DRAM_SEG, OBJ_SEG};
    }

    // Memory management methods
    [[nodiscard]] nixl_status_t registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD* &out) override;

    [[nodiscard]] nixl_status_t deregisterMem(nixlBackendMD* meta) override;

    [[nodiscard]] nixl_status_t queryMem(const nixl_reg_dlist_t &descs,
                           std::vector<nixl_query_resp_t> &resp) const override;

    // Connection management methods
    [[nodiscard]] nixl_status_t connect(const std::string &remote_agent) override;
    [[nodiscard]] nixl_status_t disconnect(const std::string &remote_agent) override;

    // Metadata management
    [[nodiscard]] nixl_status_t unloadMD(nixlBackendMD* input) override;

    // Transfer operations
    [[nodiscard]] nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH* &handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    [[nodiscard]] nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

    // Remote operations (required since supportsRemote() returns true)
    [[nodiscard]] nixl_status_t getPublicData(const nixlBackendMD* meta,
                                std::string &str) const override;

    [[nodiscard]] nixl_status_t getConnInfo(std::string &str) const override;

    [[nodiscard]] nixl_status_t loadRemoteConnInfo(const std::string &remote_agent,
                                     const std::string &remote_conn_info) override;

    [[nodiscard]] nixl_status_t loadRemoteMD(const nixlBlobDesc &input,
                               const nixl_mem_t &nixl_mem,
                               const std::string &remote_agent,
                               nixlBackendMD* &output) override;

    // Local operations (required since supportsLocal() returns true)
    [[nodiscard]] nixl_status_t loadLocalMD(nixlBackendMD* input,
                              nixlBackendMD* &output) override;
};

// Infinia metadata class to store key information
class nixlInfiniaMetadata : public nixlBackendMD {
public:
    nixlInfiniaMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, const std::string& obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key),
          buffer(nullptr),
          length(0),
          iomem_handle{} {}

    ~nixlInfiniaMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;

    // Pre-registered memory info
    void* buffer;                    // Registered buffer address
    size_t length;                   // Registered buffer length
    red_iomem_hndl_t iomem_handle;   // RED memory handle
};

/**
 * @brief Request handle for Infinia backend operations
 *
 * This class manages the state of individual transfer operations
 * for the Infinia backend using the red_async::BatchTask API.
 */
class nixlInfiniaBackendReqH : public nixlBackendReqH {
private:
    const nixl_xfer_op_t &operation_;
    const nixl_meta_dlist_t &local_;
    const nixl_meta_dlist_t &remote_;
    const std::string &remote_agent_;
    const nixl_opt_b_args_t *opt_args_;

    // Infinia-specific transfer state
    bool transfer_prepared_;
    bool transfer_posted_;
    bool transfer_completed_;
    red_status_t transfer_status_;
    size_t operation_count_;  // Track number of operations for reserve()

    std::shared_ptr<InfiniaClient> client_;

    // Batch task configuration
    red_async::rae_batch_config_t batch_config_;

    // BatchTask for coroutine-based async execution
    red_async::BatchTask batch_task_;

    // Storage for operation data (must persist until batch completes)
    // Pointers to these elements are passed to batch_task_ and accessed asynchronously
    std::vector<std::string> keys_;           // Store keys so pointers remain valid
    std::vector<red_sg_elem_t> sg_elements_;  // Scatter-gather elements
    std::vector<red_sg_list_t> sg_lists_;     // Scatter-gather lists

public:
    nixlInfiniaBackendReqH(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           const nixl_opt_b_args_t* opt_args,
                           std::shared_ptr<InfiniaClient> client,
                           const red_async::rae_batch_config_t& batch_config);

    virtual ~nixlInfiniaBackendReqH();

    // Transfer operation methods
    [[nodiscard]] nixl_status_t prepareTransfer();
    [[nodiscard]] nixl_status_t postTransfer();
    [[nodiscard]] nixl_status_t checkTransfer();

    // State query methods
    [[nodiscard]] bool isPrepared() const noexcept { return transfer_prepared_; }
    [[nodiscard]] bool isPosted() const noexcept { return transfer_posted_; }
    [[nodiscard]] bool isCompleted() const noexcept { return transfer_completed_; }
    [[nodiscard]] red_status_t getStatus() const noexcept { return transfer_status_; }
    [[nodiscard]] size_t getOperationCount() const noexcept { return operation_count_; }

    // Operation building methods
    void reserveOperations(size_t count);
    void addOperation(red_async::red_async_op_type_t op_type,
                     const std::string& key,
                     void* value_addr,
                     size_t value_size,
                     red_iomem_hndl_t iomem_handle);
};

#endif // NIXL_SRC_PLUGINS_INFINIA_INFINIA_BACKEND_H
