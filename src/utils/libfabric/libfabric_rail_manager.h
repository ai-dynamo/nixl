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

    // Rail management
    /** Create data rails for high-bandwidth transfers (one per EFA device)
     * @param efa_devices List of EFA device names to create rails on
     * @param provider_name Provider name ("efa" or "efa-direct")
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    createDataRails(const std::vector<std::string> &efa_devices, const std::string &provider_name);

    /** Create control rails for connection management and notifications
     * @param efa_devices List of EFA device names
     * @param provider_name Provider name ("efa" or "efa-direct")
     * @param num_control_rails Number of control rails to create
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    createControlRails(const std::vector<std::string> &efa_devices,
                       const std::string &provider_name,
                       size_t num_control_rails);

    // Access rails
    /** Get reference to data rail by ID */
    nixlLibfabricRail &
    getDataRail(size_t rail_id) {
        return *data_rails_[rail_id];
    }

    /** Get const reference to data rail by ID */
    const nixlLibfabricRail &
    getDataRail(size_t rail_id) const {
        return *data_rails_[rail_id];
    }

    /** Get reference to control rail by ID */
    nixlLibfabricRail &
    getControlRail(size_t rail_id) {
        return *control_rails_[rail_id];
    }

    /** Get const reference to control rail by ID */
    const nixlLibfabricRail &
    getControlRail(size_t rail_id) const {
        return *control_rails_[rail_id];
    }

    /** Get total number of data rails */
    size_t
    getNumDataRails() const {
        return data_rails_.size();
    }

    /** Get total number of control rails */
    size_t
    getNumControlRails() const {
        return control_rails_.size();
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
    /** Rail type enumeration for connection operations */
    enum class RailType { DATA, CONTROL };
    /** Insert addresses into address vectors for all rails of specified type
     * @param rail_type Type of rails to operate on (DATA or CONTROL)
     * @param endpoints Remote endpoint addresses to insert
     * @param fi_addrs_out Libfabric address handles for inserted endpoints,
     *                     indexed by local rail id.
     * @param ep_names_out Local endpoint names for reference
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    insertAllAddresses(RailType rail_type,
                       const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints,
                       std::unordered_map<size_t, std::vector<fi_addr_t>> &fi_addrs_out,
                       std::vector<char *> &ep_names_out);
    /** Clean up connection resources for specified rail type
     * @param rail_type Type of rails to clean up (DATA or CONTROL)
     * @param fi_addrs_to_remove Libfabric addresses to remove
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    cleanupConnection(RailType rail_type, const std::vector<fi_addr_t> &fi_addrs_to_remove);

    /** Single-pass transfer preparation and submission with automatic striping/round-robin
     * @param op_type Operation type (WRITE or READ)
     * @param local_addr Local memory address
     * @param transfer_size Total transfer size
     * @param remote_base_addr Remote memory base address
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
                             uint64_t remote_base_addr,
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
    /** Process completions on active data rails only (optimized for CPU overhead)
     * @return NIXL_SUCCESS if completions processed, NIXL_IN_PROG if none, error on failure
     */
    nixl_status_t
    progressActiveDataRails();
    /** Process completions on all control rails for connection management and notifications
     * @return NIXL_SUCCESS if completions processed, NIXL_IN_PROG if none, error on failure
     */
    nixl_status_t
    progressAllControlRails();
    /** Validate that all rails are properly initialized
     * @return NIXL_SUCCESS if all rails initialized, error code otherwise
     */
    nixl_status_t
    validateAllRailsInitialized();

    // Active Rail Management APIs
    /** Mark rail as active for progress tracking optimization */
    void
    markRailActive(size_t rail_id);

    /** Mark rail as inactive for progress tracking optimization */
    void
    markRailInactive(size_t rail_id);

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
     * @param data_endpoints_out Data rail endpoint addresses
     * @param control_endpoints_out Control rail endpoint addresses
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deserializeConnectionInfo(
        const std::string &user_prefix,
        const std::string &serialized_data,
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_endpoints_out,
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &control_endpoints_out) const;

    const nixlLibfabricTopology *
    getTopology() const {
        return topology.get();
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
    std::vector<std::unique_ptr<nixlLibfabricRail>> data_rails_;
    std::vector<std::unique_ptr<nixlLibfabricRail>> control_rails_;

    size_t num_data_rails_;
    size_t num_control_rails_;

    std::unique_ptr<nixlLibfabricTopology> topology;

    // EFA device to rail mapping
    std::unordered_map<std::string, size_t> efa_device_to_rail_map;

    // Active Rail Tracking System
    std::unordered_set<size_t> active_rails_;
    mutable std::mutex active_rails_mutex_;

    // Internal rail selection method
    std::vector<size_t>
    selectRailsForMemory(void *mem_addr,
                         nixl_mem_t mem_type,
                         int device_id,
                         const std::string &pci_bus_id = "") const;

    // Helper functions for connection SerDes
    void
    serializeRailEndpoints(nixlSerDes &ser_des,
                           const std::string &key_prefix,
                           RailType rail_type) const;
    nixl_status_t
    deserializeRailEndpoints(
        nixlSerDes &ser_des,
        const std::string &key_prefix,
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints_out) const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_RAIL_MANAGER_H
