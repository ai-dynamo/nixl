/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_MANAGER_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_MANAGER_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <mutex>
#include <atomic>
#include "libfabric_rail.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward declarations
class nixlLibfabricTopology;
class nixlLibfabricRailManager;

/** @brief Rail selection policy parent interface type. */
class nixlLibfabricRailSelectionPolicy {
public:
    virtual ~nixlLibfabricRailSelectionPolicy() {}

    nixlLibfabricRailSelectionPolicy(const nixlLibfabricRailSelectionPolicy &) = delete;
    nixlLibfabricRailSelectionPolicy &
    operator=(const nixlLibfabricRailSelectionPolicy &) = delete;

    /**
     * @brief Loads the policy from custom engine configuration.
     * @param rail_manager The rail manager.
     * @return True if succeeded.
     */
    virtual bool
    load(nixlLibfabricRailManager &rail_manager) = 0;

    /**
     * @brief Selects a set of rails for memory registration.
     * @param buffer The memory buffer for which data rails are to be selected.
     * @param[out] selected_rails The resulting selected rail (index list).
     * @return True if succeeded.
     */
    virtual bool
    selectRails(void *buffer, std::vector<size_t> &selected_rails) = 0;

protected:
    nixlLibfabricRailSelectionPolicy() {}
};

/** Central manager for multi-rail RDMA operations with topology awareness */
class nixlLibfabricRailManager {
public:
    /** Initialize rail manager with topology discovery and create rails based on available
     * network devices
     * @param striping_threshold Size threshold for enabling multi-rail striping
     * @throws std::runtime_error if initialization fails
     */
    nixlLibfabricRailManager(size_t striping_threshold);
    /** Destroy rail manager and cleanup all resources */
    ~nixlLibfabricRailManager();

    /**
     * @brief Initialize rail manager with provided configuration.
     * @param custom_params Custom configuration parameters from engine.
     * @return NIXL_SUCCESS on success, error code on failure
     *
     * @note Currently the following parameters can be passed in the parameter map to control the
     * behavior of the rail manager:
     *
     * - max_bw_per_dram_seg (matching environment variable NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG):
     * Controls the bandwidth limit on DRAM_SEG memory type buffers per NUMA node. Specified in
     * decimal Gbps, as multiples of 10^9 (e.g. 100, 200, etc.). If passed as key-value pair to the
     * custom parameter map, the value should be passed as a string that can be parsed as integer
     * (e.g. {"max_bw_per_dram_seg", "100"}). If not specified, then computed as the maximum
     * possible bandwidth that would not saturate the topmost PCIe bridge/switch devices of the NUMA
     * node of the origin buffer. This value (whether computed or provided by user) is converted to
     * rail count limit and used in NUMA-aware rail selection policy for DRAM_SEG, in order to limit
     * the number of rails used for this memory type. Rail selection is also limited to NUMA node of
     * the origin buffer. If user override exceeds the total topmost PCIe switch capacity of the
     * NUMA node, then rail selection spills over to additional rails on the PCI switches of the
     * NUMA node, and subsequently, if still not reaching user-specified limit, spills over to
     * adjacent NUMA nodes if required. If user override exceeds total machine network capacity,
     * then all rails will be used for DRAM_SEG memory type.
     */
    nixl_status_t
    init(const nixl_b_params_t &custom_params);

    // Rail management
    /** Create rails for high-bandwidth transfers (one per EFA device)
     * @param efa_devices List of EFA device names to create rails on
     * @param provider_name Provider name ("efa" or "efa-direct")
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    createRails(const std::vector<std::string> &efa_devices, const std::string &provider_name);

    // Access rails
    /** Get reference to rail by ID */
    nixlLibfabricRail &
    getRail(size_t rail_id) {
        return *rails_[rail_id];
    }

    /** Get const reference to rail by ID */
    const nixlLibfabricRail &
    getRail(size_t rail_id) const {
        return *rails_[rail_id];
    }

    /** Get total number of rails */
    size_t
    getNumRails() const {
        return rails_.size();
    }

    // Memory registration management
    /** Register memory with topology-aware rail selection based on memory type and location
     * @param buffer Memory buffer to register
     * @param length Buffer size in bytes
     * @param mem_type Memory type (DRAM_SEG or VRAM_SEG)
     * @param device_id Device ID (used for VRAM_SEG, ignored for DRAM_SEG)
     * @param device_pci_bus_id PCI bus ID for VRAM device (queried in backend layer), empty for
     * DRAM
     * @param mr_list_out Memory registration handles, indexed by rail ID
     * @param key_list_out Remote access keys, indexed by rail ID
     * @param selected_rails_out List of rail IDs where memory was registered
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    registerMemory(void *buffer,
                   size_t length,
                   nixl_mem_t mem_type,
                   int device_id,
                   const std::string &device_pci_bus_id,
                   std::vector<struct fid_mr *> &mr_list_out,
                   std::vector<uint64_t> &key_list_out,
                   std::vector<size_t> &selected_rails_out);
    /** Deregister memory from specified rails
     * @param selected_rails List of rail IDs to deregister from
     * @param mr_list Memory registration handles to deregister
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deregisterMemory(const std::vector<size_t> &selected_rails,
                     const std::vector<struct fid_mr *> &mr_list);

    // Connection Management APIs
    /** Insert addresses into address vectors for all rails
     * @param endpoints Remote endpoint addresses to insert
     * @param fi_addrs_out Libfabric address handles for inserted endpoints,
     *                     indexed by local rail id.
     * @param ep_names_out Local endpoint names for reference
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    insertAllAddresses(const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints,
                       std::unordered_map<size_t, std::vector<fi_addr_t>> &fi_addrs_out,
                       std::vector<char *> &ep_names_out);
    /** Clean up connection resources for rails
     * @param fi_addrs_to_remove Libfabric addresses to remove
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    cleanupConnection(const std::vector<fi_addr_t> &fi_addrs_to_remove);

    /** Single-pass transfer preparation and submission with automatic striping/round-robin
     * @param op_type Operation type (WRITE or READ)
     * @param local_addr Local memory address
     * @param transfer_size Total transfer size
     * @param remote_target_addr Remote memory target address (where to read/write)
     * @param remote_registered_base Remote registered buffer base address (for offset calculation)
     * @param selected_rails Rails to use for the transfer
     * @param local_mrs Local memory registrations
     * @param remote_keys Remote access keys
     * @param remote_selected_endpoints Selected remote endpoints, where remote keys are registered
     * @param dest_addrs Destination addresses for each rail
     * @param agent_idx Remote agent index for immediate data
     * @param xfer_id Transfer ID for tracking
     * @param completion_callback Callback for completion notification
     * @param submitted_count_out Number of requests successfully submitted
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    prepareAndSubmitTransfer(nixlLibfabricReq::OpType op_type,
                             void *local_addr,
                             size_t transfer_size,
                             uint64_t remote_target_addr,
                             uint64_t remote_registered_base,
                             const std::vector<size_t> &selected_rails,
                             const std::vector<struct fid_mr *> &local_mrs,
                             const std::vector<uint64_t> &remote_keys,
                             const std::vector<size_t> &remote_selected_endpoints,
                             const std::unordered_map<size_t, std::vector<fi_addr_t>> &dest_addrs,
                             uint16_t agent_idx,
                             uint16_t xfer_id,
                             std::function<void()> completion_callback,
                             size_t &submitted_count_out);
    /** Determine if striping should be used for given transfer size
     * @param transfer_size Size of the transfer in bytes
     * @return true if striping should be used, false for round-robin
     */
    bool
    shouldUseStriping(size_t transfer_size) const;

    // Control Message APIs
    /** Control message types for rail communication */
    enum class ControlMessageType : int {
        NOTIFICATION, ///< User notification message
    };
    /** Send control message via control rail
     * @param msg_type Type of control message
     * @param req Control request with data buffer
     * @param dest_addr Destination address
     * @param agent_idx Agent index for message routing
     * @param completion_callback Optional completion callback
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    postControlMessage(ControlMessageType msg_type,
                       nixlLibfabricReq *req,
                       fi_addr_t dest_addr,
                       uint16_t agent_idx = 0,
                       std::function<void()> completion_callback = nullptr);
    // Progress APIs
    /** Process completions on active rails only (optimized for CPU overhead)
     * @return NIXL_SUCCESS if completions processed, NIXL_IN_PROG if none, error on failure
     */
    nixl_status_t
    progressActiveRails();
    /** Validate that all rails are properly initialized
     * @return NIXL_SUCCESS if all rails initialized, error code otherwise
     */
    nixl_status_t
    validateAllRailsInitialized();

    // Active Rail Management APIs
    /** Increment active reference count on a rail */
    void
    incRailActive(size_t rail_id);

    /** Decrement active reference count on a rail; rail becomes inactive when count reaches zero */
    void
    decRailActive(size_t rail_id);

    /** Clear all active rail markings */
    void
    clearActiveRails();

    /** Get count of currently active rails */
    size_t
    getActiveRailCount() const;

    // Memory Descriptor APIs
    /** Get memory descriptor for specified rail and MR */
    struct fid_mr *
    getMemoryDescriptor(size_t rail_id, struct fid_mr *mr);

    // SerDes-based Memory Key Serialization
    /** Serialize memory keys and buffer address for remote access
     * @param keys Remote access keys for all rails
     * @param buffer Memory buffer address
     * @param str Serialized data string
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    serializeMemoryKeys(const std::vector<uint64_t> &keys, void *buffer, std::string &str) const;
    /** Deserialize memory keys and remote address
     * @param serialized_data Serialized memory information
     * @param num_keys Number of keys
     * @param keys_out Remote access keys for all rails
     * @param remote_addr_out Remote buffer address
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deserializeMemoryKeys(const std::string &serialized_data,
                          const size_t num_keys,
                          std::vector<uint64_t> &keys_out,
                          uint64_t &remote_addr_out) const;
    // SerDes-based Connection Info Serialization
    /** Serialize connection information for all rails
     * @param user_prefix Prefix for serialization keys (e.g., "src" or "dest")
     * @param str Serialized connection information
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    serializeConnectionInfo(const std::string &user_prefix, std::string &str) const;
    /** Deserialize connection information for all rails
     * @param user_prefix Prefix used during serialization
     * @param serialized_data Serialized connection information
     * @param data_endpoints_out Rail endpoint addresses
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deserializeConnectionInfo(
        const std::string &user_prefix,
        const std::string &serialized_data,
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_endpoints_out) const;

    const nixlLibfabricTopology *
    getTopology() const {
        return topology.get();
    }

    /**
     * @brief Retrieves the rail selection policy in use for DRAM_SEG memory type.
     * @return The rail selection policy.
     */
    const std::unique_ptr<nixlLibfabricRailSelectionPolicy> &
    getDramRailSelectionPolicy() const {
        return dram_rail_selection_policy_;
    }

    /** Get the system's runtime type.
     * @return fi_hmem_iface runtime type (CUDA, NEURON, or SYSTEM)
     */
    fi_hmem_iface
    getRuntime() const;

private:
    size_t striping_threshold_;

    // System runtime type (determined once at initialization)
    fi_hmem_iface runtime_;

    // Rail allocation
    std::vector<std::unique_ptr<nixlLibfabricRail>> rails_;

    size_t num_rails_;

    std::unique_ptr<nixlLibfabricTopology> topology;

    // EFA device to rail mapping
    std::unordered_map<std::string, size_t> efa_device_to_rail_map;

    // active rails with reference counting (always positive)
    std::unordered_map<size_t, size_t> active_rails_;
    mutable std::mutex active_rails_mutex_;

    // rail selection policy for DRAM memory type
    std::unique_ptr<nixlLibfabricRailSelectionPolicy> dram_rail_selection_policy_;

    // get rail count limit for DRAM memory type, either computed or from user
    bool
    getDramRailLimit(const nixl_b_params_t &custom_params, size_t &max_bw, size_t &max_rails);

    // Internal rail selection method
    std::vector<size_t>
    selectRailsForMemory(void *mem_addr,
                         nixl_mem_t mem_type,
                         int device_id,
                         const std::string &pci_bus_id = "") const;

    // Helper functions for connection SerDes
    void
    serializeRailEndpoints(nixlSerDes &ser_des, const std::string &key_prefix) const;
    nixl_status_t
    deserializeRailEndpoints(
        nixlSerDes &ser_des,
        const std::string &key_prefix,
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints_out) const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_MANAGER_H
