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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_BACKEND_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_BACKEND_H

#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/nixl_time.h"
#include "serdes/serdes.h"

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// Forward declarations
class nixlLibfabricEngine;

#ifdef HAVE_CUDA
/** CUDA context management for libfabric backend */
class nixlLibfabricCudaCtx {
private:
    CUcontext pthrCudaCtx_;
    int myDevId_;

public:
    nixlLibfabricCudaCtx() {
        pthrCudaCtx_ = NULL;
        myDevId_ = -1;
    }

    /** Reset CUDA context pointer to initial state */
    void
    cudaResetCtxPtr();

    /** Update CUDA context pointer for given memory address and device */
    int
    cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated);

    /** Set the current CUDA context */
    int
    cudaSetCtx();
};
#endif

/** Private metadata for locally registered memory */
class nixlLibfabricPrivateMetadata : public nixlBackendMD {
private:
    void *buffer_; // Local memory buffer address
    size_t length_; // Buffer length in bytes
    int gpu_device_id_; // GPU device ID for VRAM, -1 for DRAM
    std::vector<struct fid_mr *> rail_mr_list_; // Memory registrations, one per rail
    std::vector<uint64_t> rail_key_list_; // Remote access keys, one per rail
    std::vector<char *> src_ep_names_; // Source endpoint names, one per rail
    std::vector<size_t> selected_rails_; // Rails selected based on memory topology

public:
    nixlLibfabricPrivateMetadata() : nixlBackendMD(true), gpu_device_id_(-1) {}
    friend class nixlLibfabricEngine;
};

/** Public metadata for remote memory access */
class nixlLibfabricPublicMetadata : public nixlBackendMD {
private:
    uint64_t remote_buf_addr_; // Remote buffer base address
    std::shared_ptr<nixlLibfabricConnection> conn_; // Connection to remote agent
    std::vector<uint64_t> rail_remote_key_list_; // Remote access keys, one per rail
    std::vector<char *> src_ep_names_; // Source endpoint names, one per rail
    std::vector<size_t>
        remote_selected_endpoints_; // Remote rails selected, derived from rail_remote_key_list_.

public:
    nixlLibfabricPublicMetadata() : nixlBackendMD(false) {}

    void
    derive_remote_selected_endpoints();

    friend class nixlLibfabricEngine;
};

/** Multi-rail connection metadata for remote agents */
class nixlLibfabricConnection : public nixlBackendConnMD {
private:
    size_t agent_index_; // Unique agent identifier in agent_names vector
    std::string remoteAgent_; // Remote agent name
    std::unordered_map<size_t, std::vector<fi_addr_t>>
        rail_remote_addr_list_; // Data rail libfabric addresses. key=data rail id.
    std::unordered_map<size_t, std::vector<fi_addr_t>>
        control_rail_remote_addr_list_; // Control rail libfabric addresses. key=control rail id.
    std::vector<char *> src_ep_names_; // Data rail endpoint names
    std::vector<char *> control_ep_names_; // Control rail endpoint names
    ConnectionState overall_state_; // Current connection state
    std::mutex conn_state_mutex_; // Protects connection state
    std::condition_variable cv_; // For blocking connection establishment
    size_t num_connected_rails_; // Number of successfully connected rails
    std::string initiator_addr_; // Local endpoint address
    std::string remote_addr_; // Remote endpoint address
public:
    friend class nixlLibfabricEngine;
    friend class nixlLibfabricRail;
};

/** Request handle for multi-rail transfer operations */
class nixlLibfabricBackendH : public nixlBackendReqH {
private:
    std::atomic<size_t> completed_requests_; // Atomic count of completed requests
    std::atomic<size_t> submitted_requests_; // Total number of submitted requests

public:
    uint16_t post_xfer_id;
    const nixl_xfer_op_t operation_;
    const std::string remote_agent_;
    bool has_notif;
    uint32_t total_notif_msg_len; // Total length of notification message across all fragments

    std::vector<BinaryNotification> binary_notifs; // Vector of BinaryNotification for fragmentation

    nixlLibfabricBackendH(nixl_xfer_op_t op, const std::string &remote_agent);
    ~nixlLibfabricBackendH();

    /** Check if all requests in this transfer have completed */
    bool
    is_completed() const;

    /** Initialize completion tracking for multi-request transfer */
    void
    init_request_tracking(size_t num_requests);

    /** Atomically increment completed request count */
    void
    increment_completed_requests();

    /** Get current count of requests completed as part of this transfer */
    size_t
    get_completed_requests_count() const;

    /** Get total number of requests submitted as part of this transfer */
    size_t
    get_submitted_requests_count() const;

    /** Adjust total submitted request count to actual value after submissions complete */
    void
    adjust_total_submitted_requests(size_t actual_count);
};

class nixlLibfabricEngine : public nixlBackendEngine {
    friend class nixlLibfabricRail; // Allow nixlLibfabricRail to access private members

private:
    // Threading infrastructure - declared first to match initialization order
    std::atomic<bool> cm_thread_stop_;

    // Store user's original progress thread preference
    bool progress_thread_enabled_;

    // Progress thread delay in microseconds
    std::chrono::microseconds progress_thread_delay_;

    // Rail Manager - Stack allocated for better performance (mutable for const methods)
    mutable nixlLibfabricRailManager rail_manager;

    // Configurable striping threshold
    size_t striping_threshold_;

    mutable size_t total_transfer_size_;

    // Map of agent name to connection info
    // <remoteAgent, <connection>>
    mutable std::unordered_map<std::string, std::shared_ptr<nixlLibfabricConnection>> connections_;
    mutable std::vector<std::string> agent_names_; // List of agent names for easy access

    // Threading infrastructure - remaining members
    // Connection Management (CM) thread
    std::mutex cm_mutex_;
    std::thread cm_thread_;
    std::condition_variable cm_cv_;

    // Progress thread for data rail CQs only
    std::thread progress_thread_;
    std::atomic<bool> progress_thread_stop_;

    // Mutex for connection state tracking
    mutable std::mutex connection_state_mutex_;

    void
    cleanup();

    // Central notification storage
    std::mutex notif_mutex_;
    notif_list_t notifMainList_;

    // Receiver Side XFER_ID Tracking
    std::mutex receiver_tracking_mutex_;
    std::unordered_set<uint32_t> received_remote_writes_; // All received XFER_IDs (global)

    // Notification Queuing
    struct PendingNotification {
        std::string remote_agent;
        std::vector<std::string> message_fragments; // Store each fragment separately
        uint16_t notif_xfer_id;
        uint32_t expected_completions; // Expected transfer requests for this notif_xfer_id
        uint32_t received_completions; // Actual remote transfer completions received for this
                                       // notif_xfer_id
        uint16_t expected_msg_fragments; // Total fragments expected (from notif_seq_len)
        uint16_t received_msg_fragments; // Fragments received so far
        uint32_t total_message_length; // Total length of complete message (all fragments)
        uint16_t agent_name_length; // Length of agent_name in combined payload

        PendingNotification(uint16_t xfer_id)
            : notif_xfer_id(xfer_id),
              expected_completions(0),
              received_completions(0),
              expected_msg_fragments(0),
              received_msg_fragments(0),
              total_message_length(0),
              agent_name_length(0) {}
    };

    // O(1) lookup with postXferID key
    std::unordered_map<uint16_t, PendingNotification> pending_notifications_;

    // Connection management helpers
    nixl_status_t
    establishConnection(const std::string &remote_agent) const;

    // Common connection creation helper
    nixl_status_t
    createAgentConnection(const std::string &agent_name,
                          const std::vector<std::array<char, 56>> &data_rail_endpoints,
                          const std::vector<std::array<char, 56>> &control_rail_endpoints);

    // Private notification implementation with unified binary notification system
    nixl_status_t
    notifSendPriv(const std::string &remote_agent,
                  std::vector<BinaryNotification> &binary_notifications,
                  uint32_t total_message_length,
                  uint16_t notif_xfer_id,
                  uint32_t expected_completions) const;

    // Private function to fragment notification messages to binary notifications
    void
    fragmentNotificationMessage(const std::string &message,
                                const std::string &agent_name,
                                uint32_t &total_message_length,
                                std::vector<BinaryNotification> &fragments_out) const;
#ifdef HAVE_CUDA
    // CUDA context management
    std::unique_ptr<nixlLibfabricCudaCtx> cudaCtx_;
    bool cuda_addr_wa_; // CUDA address workaround flag
#endif

    // ConnectionManagement thread and completion processing
    nixl_status_t
    cmThread();
    void
    postShutdownCompletion();
    // Progress thread for data rail CQs only
    nixl_status_t
    progressThread();


    // Engine message processing methods
    void
    processNotification(const std::string &serialized_notif);
    void
    processConnectionAck(uint16_t agent_idx,
                         nixlLibfabricConnection *conn_info,
                         ConnectionState state);
    nixl_status_t
    processConnectionRequest(uint16_t agent_idx,
                             const std::string &serialized_data,
                             nixlLibfabricRail *rail);
    nixl_status_t
    loadMetadataHelper(const std::vector<uint64_t> &rail_keys,
                       void *buffer,
                       std::shared_ptr<nixlLibfabricConnection> conn,
                       nixlBackendMD *&output);

#ifdef HAVE_CUDA
    // CUDA context management methods
    void
    vramInitCtx();
    int
    vramUpdateCtx(void *address, uint64_t devId, bool &restart_reqd);
    int
    vramApplyCtx();
    void
    vramFiniCtx();
#endif

public:
    /** Initialize multi-rail libfabric backend engine */
    nixlLibfabricEngine(const nixlBackendInitParams *init_params);
    /** Destroy engine and cleanup all resources */
    ~nixlLibfabricEngine();

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

    /** Get list of supported memory types */
    nixl_mem_list_t
    getSupportedMems() const override;

    /* Object management */
    /** Serialize memory metadata for remote access */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;

    /** Get local connection information for all rails */
    nixl_status_t
    getConnInfo(std::string &str) const override;

    /** Load remote agent connection information */
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent,
                       const std::string &remote_conn_info) override;

    /** Establish connection to remote agent */
    nixl_status_t
    connect(const std::string &remote_agent) override;
    /**
     * @brief Gracefully disconnects from a remote agent and cleans up associated resources
     *
     * This function performs a complete disconnect sequence that ensures proper cleanup
     * of all libfabric resources and notifies the remote peer of the disconnection.
     *
     * The disconnect process follows these steps:
     * 1. Validates that an active connection exists for the specified remote agent
     * 2. Sends disconnect notification message to the remote peer via control rail, with best
     * effort semantics
     * 3. Releases libfabric resources like address vector entries via rail manager
     * 4. Updates internal connection state to DISCONNECTED
     * 5. Removes connection entry from the active connection map
     *
     * @param[in] remote_agent The identifier of the remote agent to disconnect from
     *
     * @return nixl_status_t Status code indicating the result of the disconnect operation
     * @retval NIXL_SUCCESS Connection successfully disconnected and cleaned up
     * @retval NIXL_ERR_NOT_FOUND No active connection exists for the specified remote agent
     * @retval NIXL_ERR_TIMEOUT Remote peer did not acknowledge disconnect within timeout period
     * @retval NIXL_ERR_BACKEND Libfabric resource cleanup failed
     *
     * @note This function is thread-safe and can be called concurrently from multiple threads
     * @note Active transfers will be cancelled, which may result in incomplete data transmission
     * @note The function implements best-effort graceful disconnect; if acknowledgment times out,
     *       local cleanup still proceeds to prevent resource leaks
     *
     * @warning Calling disconnect on a non-existent connection returns NIXL_ERR_NOT_FOUND
     * @warning This operation is irreversible; a new connection must be established to resume
     * communication
     *
     * @see connect() for establishing connections
     * @see getConnectionState() for querying connection status
     */
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    /**
     * @brief Register memory for RDMA operations with GPU Direct support
     *
     * Registers memory buffer with libfabric on topology-appropriate rails.
     * Supports both DRAM and VRAM with automatic rail selection based on
     * memory location and system topology.
     *
     * @param[in] mem Memory descriptor with address, length, and device info
     * @param[in] nixl_mem Memory type (DRAM_SEG or VRAM_SEG)
     * @param[out] out Private metadata containing registration information
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    /**
     * @brief Deregister memory from libfabric
     *
     * Cleans up memory registrations on all rails where the memory was registered.
     *
     * @param[in] meta Private metadata from registerMem()
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /**
     * @brief Create public metadata from local private metadata
     *
     * Converts private memory registration into public metadata that can be
     * used for local transfers (loopback operations).
     *
     * @param[in] input Private metadata from registerMem()
     * @param[out] output Public metadata for local operations
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

    /**
     * @brief Create public metadata from remote serialized data
     *
     * Deserializes remote memory information and creates public metadata
     * for accessing remote memory via RDMA operations.
     *
     * @param[in] input Blob descriptor containing serialized remote metadata
     * @param[in] nixl_mem Memory type of the remote memory
     * @param[in] remote_agent Name of the remote agent owning the memory
     * @param[out] output Public metadata for remote memory access
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;

    /**
     * @brief Release metadata resources
     *
     * Cleans up metadata object and associated resources.
     *
     * @param[in] input Metadata to release
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    // Data transfer
    /**
     * @brief Prepare transfer operation handle
     *
     * Creates and initializes a request handle for upcoming data transfer.
     * Validates connection and prepares internal structures.
     *
     * @param[in] operation Transfer operation type (NIXL_WRITE or NIXL_READ)
     * @param[in] local Local memory descriptors
     * @param[in] remote Remote memory descriptors
     * @param[in] remote_agent Target remote agent name
     * @param[out] handle Request handle for the transfer
     * @param[in] opt_args Optional arguments (notifications, etc.)
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * @brief Estimate transfer cost and timing
     *
     * Provides cost estimation for the specified transfer operation.
     * Currently returns success without detailed estimation.
     *
     * @param[in] operation Transfer operation type
     * @param[in] local Local memory descriptors
     * @param[in] remote Remote memory descriptors
     * @param[in] remote_agent Target remote agent name
     * @param[in] handle Request handle
     * @param[out] duration Estimated transfer duration
     * @param[out] err_margin Error margin for the estimate
     * @param[out] method Cost estimation method used
     * @param[in] opt_args Optional arguments
     * @return NIXL_SUCCESS on success, error code on failure
     */
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

    /**
     * @brief Execute data transfer with multi-rail striping
     *
     * Performs high-performance data transfer using multiple rails with automatic
     * striping for large transfers or round-robin for small transfers. Supports
     * immediate notifications and completion tracking.
     *
     * @param[in] operation Transfer operation type (NIXL_WRITE or NIXL_READ)
     * @param[in] local Local memory descriptors
     * @param[in] remote Remote memory descriptors
     * @param[in] remote_agent Target remote agent name
     * @param[in,out] handle Request handle for tracking completion
     * @param[in] opt_args Optional arguments including notifications
     * @return NIXL_SUCCESS if complete, NIXL_IN_PROG if ongoing, error code on failure
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * @brief Check transfer completion status
     *
     * Polls for transfer completion and processes any pending completions
     * if progress thread is disabled.
     *
     * @param[in] handle Request handle to check
     * @return NIXL_SUCCESS if complete, NIXL_IN_PROG if ongoing, error code on failure
     */
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    /**
     * @brief Release request handle resources
     *
     * Cleans up request handle after transfer completion.
     *
     * @param[in] handle Request handle to release
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    // Notification system
    /**
     * @brief Retrieve available notifications
     *
     * Returns all currently available notifications and processes any pending
     * completions if progress thread is disabled.
     *
     * @param[out] notif_list List to store retrieved notifications
     * @return NIXL_SUCCESS if notifications available, NIXL_IN_PROG if none, error code on failure
     */
    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

    /**
     * @brief Send notification to remote agent
     *
     * Sends a notification message to the specified remote agent using
     * the binary notification protocol.
     *
     * @param[in] remote_agent Target remote agent name
     * @param[in] msg Notification message to send
     * @return NIXL_SUCCESS on success, error code on failure
     */
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

    // Receiver Side XFER_ID Tracking Helper Methods
    /**
     * @brief Add received XFER_ID with counter-based matching
     *
     * Thread-safe method to track received data transfers.
     *
     * @param[in] xfer_id 16-bit transfer ID that was received
     */
    void
    addReceivedXferId(uint16_t xfer_id);

    // Notification Queuing Helper Methods
    /**
     * @brief Process pending notifications that are now ready
     *
     * Checks pending notifications to see if their associated transfers
     * have completed and moves them to the main notification list.
     */
    void
    checkPendingNotifications();
};

#endif
