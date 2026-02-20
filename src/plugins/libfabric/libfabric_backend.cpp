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

#include "libfabric_backend.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#include <dlfcn.h>
#include <limits>
#include <cstring>
#include <unistd.h>

#include <iomanip>
#include <numeric>

#include "absl/strings/numbers.h"

/****************************************
 * Neuron Address Query
 *****************************************/
namespace {

void *
dlopen_libnrt() {
    static void *const handle = dlopen("libnrt.so.1", RTLD_NOW);
    return handle;
}

template<class Fn>
Fn *
_load_nrt_symbol(const char *fn_name, Fn *) {
    void *libnrt_handle = dlopen_libnrt();
    if (libnrt_handle) {
        return reinterpret_cast<Fn *>(dlsym(libnrt_handle, fn_name));
    }
    return nullptr;
}

#define LOAD_NRT_SYMBOL(sym) _load_nrt_symbol(#sym, &sym)

int
nrt_get_attached_efa_bdf(const void *va, char *efa_bdf, size_t *len) {
    static const auto fn = LOAD_NRT_SYMBOL(nrt_get_attached_efa_bdf);
    if (fn == nullptr) {
        NIXL_ERROR << "Could not resolve libnrt symbol: " << __func__;
        return -1;
    }
    return fn(va, efa_bdf, len);
}

int
nrtQueryAddr(const void *va, std::string *efa_bdf) {
    char buf[] = "0000:00:00.0";
    size_t buflen = sizeof(buf);

    if (nrt_get_attached_efa_bdf(va, buf, &buflen) == 0) {
        efa_bdf->assign(buf, buflen);
        return 0;
    }

    return -1;
}

} // namespace

#ifdef HAVE_CUDA
// CUDA error checking macros
#define CHECK_CUDA_ERROR(result, message)                                                         \
    do {                                                                                          \
        if (result != cudaSuccess) {                                                              \
            NIXL_ERROR << "CUDA Error: " << message << " (" << cudaGetErrorString(result) << ")"; \
            return NIXL_ERR_BACKEND;                                                              \
        }                                                                                         \
    } while (0)

#define CHECK_CUDA_DRIVER_ERROR(result, message)                                        \
    do {                                                                                \
        if (result != CUDA_SUCCESS) {                                                   \
            const char *error_str;                                                      \
            cuGetErrorString(result, &error_str);                                       \
            NIXL_ERROR << "CUDA Driver Error: " << message << " (" << error_str << ")"; \
            return NIXL_ERR_BACKEND;                                                    \
        }                                                                               \
    } while (0)
#endif

/****************************************
 * CUDA Context Management
 *****************************************/

#ifdef HAVE_CUDA
static int
cudaQueryAddr(void *address, bool &is_dev, CUdevice &dev, CUcontext &ctx, std::string &pci_bus_id) {
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
    CUpointer_attribute attr_type[4];
    void *attr_data[4];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);
    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    // Get PCI bus ID if device memory
    if (result == CUDA_SUCCESS && is_dev) {
        char pci_buf[32];
        CUresult pci_result = cuDeviceGetPCIBusId(pci_buf, sizeof(pci_buf), dev);
        if (pci_result == CUDA_SUCCESS) {
            pci_bus_id = std::string(pci_buf);
        } else {
            pci_bus_id = "";
        }
    } else {
        pci_bus_id = "";
    }

    return (CUDA_SUCCESS != result);
}

void
nixlLibfabricCudaCtx::cudaResetCtxPtr() {
    pthrCudaCtx_ = NULL;
    myDevId_ = -1;
}

int
nixlLibfabricCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated) {
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    std::string pci_bus_id; // Not used here, but required by cudaQueryAddr
    int ret;

    was_updated = false;

    if (expected_dev == -1) return -1;
    if (myDevId_ != -1 && expected_dev != myDevId_) return -1;

    ret = cudaQueryAddr(address, is_dev, dev, ctx, pci_bus_id);
    if (ret) return ret;
    if (!is_dev) return 0;
    if (dev != expected_dev) return -1;

    if (pthrCudaCtx_) {
        if (pthrCudaCtx_ != ctx) return -1;
        return 0;
    }

    pthrCudaCtx_ = ctx;
    was_updated = true;
    myDevId_ = expected_dev;

    return 0;
}

int
nixlLibfabricCudaCtx::cudaSetCtx() {
    CUresult result;
    if (NULL == pthrCudaCtx_) return 0;

    result = cuCtxSetCurrent(pthrCudaCtx_);
    return (CUDA_SUCCESS == result);
}

void
nixlLibfabricEngine::vramInitCtx() {
    cudaCtx_ = std::make_unique<nixlLibfabricCudaCtx>();
}

int
nixlLibfabricEngine::vramUpdateCtx(void *address, uint64_t devId, bool &restart_reqd) {
    int ret;
    bool was_updated;

    restart_reqd = false;

    if (!cuda_addr_wa_) {
        return 0; // Nothing to do
    }

    ret = cudaCtx_->cudaUpdateCtxPtr(address, devId, was_updated);
    if (ret) {
        return ret;
    }

    restart_reqd = was_updated;
    return 0;
}

int
nixlLibfabricEngine::vramApplyCtx() {
    if (!cuda_addr_wa_) {
        return 0; // Nothing to do
    }
    return cudaCtx_->cudaSetCtx();
}

void
nixlLibfabricEngine::vramFiniCtx() {
    cudaCtx_.reset();
}
#endif

/****************************************
 * Request Management
 *****************************************/

nixlLibfabricBackendH::nixlLibfabricBackendH(nixl_xfer_op_t op, const std::string &remote_agent)
    : completed_requests_(0),
      submitted_requests_(0),
      operation_(op),
      remote_agent_(remote_agent),
      total_notif_msg_len(0) {
    // Initialize BinaryNotification vector
    binary_notifs.clear();

    NIXL_DEBUG << " handle constructor called, address: " << this
               << " total_requests_used=" << submitted_requests_.load()
               << " BinaryNotification vector initialized";
}

nixlLibfabricBackendH::~nixlLibfabricBackendH() {
    NIXL_DEBUG << "handle destructor called, address: " << this;
}

// Multi-request completion tracking methods
void
nixlLibfabricBackendH::init_request_tracking(size_t num_requests) {
    submitted_requests_.store(num_requests);
    completed_requests_.store(0);
    NIXL_DEBUG << "Initialized request tracking for " << num_requests << " requests";
}

void
nixlLibfabricBackendH::increment_completed_requests() {
    completed_requests_.fetch_add(1);
    NIXL_DEBUG << "Request completed, total completed: " << completed_requests_.load() << "/"
               << submitted_requests_.load();
}

size_t
nixlLibfabricBackendH::get_completed_requests_count() const {
    return completed_requests_.load();
}

size_t
nixlLibfabricBackendH::get_submitted_requests_count() const {
    return submitted_requests_.load();
}

void
nixlLibfabricBackendH::adjust_total_submitted_requests(size_t actual_count) {
    submitted_requests_.store(actual_count);
    NIXL_DEBUG << "Adjusted total requests to actual count: " << actual_count;
}

bool
nixlLibfabricBackendH::is_completed() const {
    // Transfer is completed when all requests have local completions
    return completed_requests_.load() == submitted_requests_.load();
}

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlLibfabricEngine::nixlLibfabricEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      progress_thread_enabled_(init_params->enableProgTh),
      progress_thread_delay_(std::chrono::microseconds(init_params->pthrDelay)),
      rail_manager(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
      runtime_(FI_HMEM_SYSTEM) {

    NIXL_DEBUG << "Initializing Libfabric Backend";

    // Query system runtime type from rail manager (determined once at topology discovery)
    runtime_ = rail_manager.getRuntime();

    NIXL_INFO << "System runtime: "
              << (runtime_ == FI_HMEM_CUDA       ? "CUDA" :
                      runtime_ == FI_HMEM_NEURON ? "NEURON" :
                                                   "SYSTEM");

#ifdef HAVE_CUDA
    if (runtime_ == FI_HMEM_CUDA) {
        // Initialize CUDA context management
        vramInitCtx();
        // CUDA address workaround
        if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
            NIXL_DEBUG << "Disabling CUDA address workaround";
            cuda_addr_wa_ = false;
        } else {
            cuda_addr_wa_ = true;
            NIXL_DEBUG << "CUDA address workaround enabled";
        }
    }
#endif

    // Parse striping threshold parameter
    std::string threshold_str;
    striping_threshold_ = NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD;

    if (getInitParam("striping_threshold", threshold_str) == NIXL_SUCCESS) {
        try {
            striping_threshold_ = std::stoull(threshold_str);
            NIXL_DEBUG << "Using custom striping threshold: " << striping_threshold_ << " bytes";
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Invalid striping_threshold value '" << threshold_str
                      << "', using default: " << striping_threshold_ << " bytes";
        }
    } else {
        NIXL_DEBUG << "Using default striping threshold: " << striping_threshold_ << " bytes";
    }

    // Initialize Rail Manager which will discover the topology and create all rails.
    try {
        NIXL_DEBUG << "Rail Manager created with " << rail_manager.getNumDataRails()
                   << " data rails and " << rail_manager.getNumControlRails() << " control rails";

        // Set up callbacks on each rail using Engine's static callback functions
        size_t control_rail_id = 0;
        NIXL_DEBUG << "Set notification processor for control rail 0";
        rail_manager.getControlRail(control_rail_id)
            .setNotificationCallback([this](const std::string &serialized_notif) {
                processNotification(serialized_notif);
            });

        // Set up XFER_ID tracking callbacks for all data rails
        NIXL_DEBUG << "Setting up XFER_ID tracking callbacks for " << rail_manager.getNumDataRails()
                   << " data rails";
        for (size_t data_rail_id = 0; data_rail_id < rail_manager.getNumDataRails();
             ++data_rail_id) {
            rail_manager.getDataRail(data_rail_id).setXferIdCallback([this](uint64_t imm_data) {
                // Extract XFER_ID from immediate data
                uint16_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(imm_data);
                addReceivedXferId(xfer_id);
            });
            NIXL_DEBUG << "Set XFER_ID callback for data rail " << data_rail_id;
        }

        // Create self-connection
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints(
            rail_manager.getNumDataRails());
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> control_endpoints(
            rail_manager.getNumControlRails());
        // Prepare data rail endpoints
        for (size_t rail_id = 0; rail_id < rail_manager.getNumDataRails(); ++rail_id) {
            std::memcpy(data_endpoints[rail_id].data(),
                        rail_manager.getDataRail(rail_id).ep_name,
                        sizeof(rail_manager.getDataRail(rail_id).ep_name));
        }
        // Prepare control rail endpoints
        for (size_t rail_id = 0; rail_id < rail_manager.getNumControlRails(); ++rail_id) {
            std::memcpy(control_endpoints[rail_id].data(),
                        rail_manager.getControlRail(rail_id).ep_name,
                        sizeof(rail_manager.getControlRail(rail_id).ep_name));
        }
        // Create self-connection using common method
        nixl_status_t conn_status =
            createAgentConnection(localAgent, data_endpoints, control_endpoints);
        if (conn_status != NIXL_SUCCESS) {
            throw std::runtime_error(
                "createAgentConnection failed for self-connection with status: " +
                std::to_string(conn_status));
        }

        NIXL_DEBUG << "Created self-connection for agent: " << localAgent << " on "
                   << rail_manager.getNumDataRails() << " data rails and "
                   << rail_manager.getNumControlRails() << " control rails";

        // Start Progress thread for data rail completion processing
        if (progress_thread_enabled_) {
            NIXL_DEBUG << "Starting Progress thread for data rails with delay: "
                       << progress_thread_delay_.count() << " microseconds";
            progress_thread_stop_ = false;
            progress_thread_ = std::thread(&nixlLibfabricEngine::progressThread, this);

            if (!progress_thread_.joinable()) {
                NIXL_ERROR << "Failed to start Progress thread";
                throw std::runtime_error("Failed to start Progress thread");
            }
            NIXL_DEBUG << "Progress thread started successfully";
        } else {
            NIXL_DEBUG << "Progress thread disabled, using manual progress in checkXfer/getNotifs";
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to initialize libfabric backend: " << e.what();
        cleanup();
        throw;
    }
}

nixlLibfabricEngine::~nixlLibfabricEngine() {
    NIXL_DEBUG
        << "Destructor starting, stopping all threads FIRST to prevent timing report interruption";

    if (progress_thread_enabled_) {
        progress_thread_stop_.store(true);
    }

    nixl_status_t progress_status = rail_manager.progressAllControlRails();
    if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to progress control rails in ~nixlLibfabricEngine().";
    }

    if (progress_thread_enabled_ && progress_thread_.joinable()) {
        NIXL_DEBUG << "Waiting for Progress thread to exit";
        progress_thread_.join();
        NIXL_DEBUG << "Progress thread joined successfully";
    } else if (!progress_thread_enabled_) {
        NIXL_DEBUG << "Progress thread was not running";
    }
    NIXL_DEBUG << "All threads stopped, now cleaning up resources";
    cleanup();
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t
nixlLibfabricEngine::getConnInfo(std::string &str) const {
    // Verify all rail endpoints are initialized
    for (size_t rail_id = 0; rail_id < rail_manager.getNumDataRails(); ++rail_id) {
        if (!rail_manager.getDataRail(rail_id).endpoint) {
            NIXL_ERROR << "Rail " << rail_id << " endpoint not initialized";
            return NIXL_ERR_BACKEND;
        }
    }

    NIXL_DEBUG << "Retrieving local endpoint addresses for all " << rail_manager.getNumDataRails()
               << " rails";

    // Use Rail Manager's connection SerDes method with "dest" prefix for remote consumption
    nixl_status_t status = rail_manager.serializeConnectionInfo("dest", str);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager serializeConnectionInfo failed";
        return status;
    }

    NIXL_DEBUG << "Rail Manager serialized connection info for " << rail_manager.getNumDataRails()
               << " rails, " << rail_manager.getNumControlRails() << " control rails, "
               << "total size=" << str.length();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                        const std::string &remote_conn_info) {
    std::lock_guard<std::mutex> lock(connection_state_mutex_);

    NIXL_DEBUG << "Loading remote info for agent: " << remote_agent
               << ", info length=" << remote_conn_info.length() << ", info (hex): "
               << LibfabricUtils::hexdump(remote_conn_info.data(), remote_conn_info.length());

    if (remote_conn_info.empty()) {
        NIXL_ERROR << "Empty remote connection info received";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_DEBUG << "Processing " << rail_manager.getNumDataRails() << " data rails and "
               << rail_manager.getNumControlRails() << " control rails for agent: " << remote_agent;

    // Use Rail Manager's connection SerDes method with "dest" prefix (remote is sending us their
    // endpoints as "dest")
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints;
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> control_endpoints;
    nixl_status_t status = rail_manager.deserializeConnectionInfo(
        "dest", remote_conn_info, data_endpoints, control_endpoints);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager deserializeConnectionInfo failed";
        return status;
    }
    // Create connection to remote agent
    nixl_status_t conn_status =
        createAgentConnection(remote_agent, data_endpoints, control_endpoints);
    if (conn_status != NIXL_SUCCESS) {
        NIXL_ERROR << "createAgentConnection failed with status: " << conn_status;
        return conn_status;
    }

    NIXL_DEBUG << "Successfully stored multirail connection for " << remote_agent << " on "
               << rail_manager.getNumDataRails() << " rails";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::connect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(connection_state_mutex_);

    NIXL_DEBUG << "Connecting to agent: " << remote_agent
               << ", connections_ size=" << connections_.size();

    // Check if connection is already established
    auto it = connections_.find(remote_agent);
    if (it != connections_.end() && it->second->overall_state_ == ConnectionState::CONNECTED) {
        NIXL_DEBUG << "Connection already established for " << remote_agent
                   << ", fi_addr=" << it->second->rail_remote_addr_list_[0][0];
        return NIXL_SUCCESS;
    }

    // Connection exists but not established - trigger establishConnection()
    NIXL_DEBUG << "Connection exists but not established, triggering establishConnection for "
               << remote_agent;

    // Release the lock before calling establishConnection since it acquires the same mutex
    lock.~lock_guard();

    nixl_status_t status = establishConnection(remote_agent);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to establish connection with " << remote_agent;
        return status;
    }

    it = connections_.find(remote_agent);
    if (it == connections_.end()) {
        NIXL_DEBUG << "Connect failed. No metadata connection info for " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    NIXL_DEBUG << "Successfully established connection for " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(connection_state_mutex_);
    auto it = connections_.find(remote_agent);
    if (it == connections_.end()) {
        NIXL_ERROR << "Disconnect failed. No metadata connection info for " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }
    // Connection exists - check if already disconnected
    if (it->second->overall_state_ == ConnectionState::DISCONNECTED) {
        NIXL_DEBUG << "Connection already disconnected for " << remote_agent
                   << ", fi_addr=" << it->second->rail_remote_addr_list_[0][0];
        return NIXL_SUCCESS;
    }
    // TODO: Implement disconnect logic to cleanup the AV Address Entries from both local and remote
    // AV.

    // Update connection state to DISCONNECTED before removing
    it->second->overall_state_ = ConnectionState::DISCONNECTED;

    // Remove connection from map
    connections_.erase(remote_agent);
    NIXL_DEBUG << "Connection erased from the connection map for agent: " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::createAgentConnection(
    const std::string &agent_name,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_rail_endpoints,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &control_rail_endpoints) {

    NIXL_DEBUG << "Creating connection for agent: " << agent_name;

    if (data_rail_endpoints.size() != rail_manager.getNumDataRails()) {
        NIXL_INFO << "Local " << rail_manager.getNumDataRails() << " data rail endpoints, remote "
                  << data_rail_endpoints.size();
    }

    if (control_rail_endpoints.size() != rail_manager.getNumControlRails()) {
        NIXL_ERROR << "Expected " << rail_manager.getNumControlRails()
                   << " control rail endpoints, got " << control_rail_endpoints.size();
        return NIXL_ERR_INVALID_PARAM;
    }

    // Create connection object
    auto conn = std::make_shared<nixlLibfabricConnection>();
    if (!conn) {
        NIXL_ERROR << "Failed to allocate connection object";
        return NIXL_ERR_BACKEND;
    }

    conn->remoteAgent_ = agent_name;
    conn->rail_remote_addr_list_.reserve(rail_manager.getNumDataRails());
    conn->control_rail_remote_addr_list_.reserve(rail_manager.getNumControlRails());

    // Process all data rails in one operation
    nixl_status_t data_status =
        rail_manager.insertAllAddresses(nixlLibfabricRailManager::RailType::DATA,
                                        data_rail_endpoints,
                                        conn->rail_remote_addr_list_,
                                        conn->src_ep_names_);
    if (data_status != NIXL_SUCCESS) {
        NIXL_ERROR << "insertAllAddresses failed for data rails with status: " << data_status;
        return NIXL_ERR_BACKEND;
    }

    // Process all control rails in one operation
    nixl_status_t control_status =
        rail_manager.insertAllAddresses(nixlLibfabricRailManager::RailType::CONTROL,
                                        control_rail_endpoints,
                                        conn->control_rail_remote_addr_list_,
                                        conn->control_ep_names_);
    if (control_status != NIXL_SUCCESS) {
        NIXL_ERROR << "insertAllAddresses failed for control rails with status: " << control_status;
        return NIXL_ERR_BACKEND;
    }

    // Manage agent names and index
    agent_names_.push_back(agent_name);
    int index = 0;
    std::for_each(agent_names_.begin(), agent_names_.end(), [&index](const std::string &name) {
        NIXL_DEBUG << "Index " << index << ": " << name;
        index++;
    });
    conn->agent_index_ = agent_names_.size() - 1;

    // Store connection
    connections_[agent_name] = conn;

    NIXL_DEBUG << "Successfully created connection for agent: " << agent_name << " on "
               << rail_manager.getNumDataRails() << " data rails and "
               << rail_manager.getNumControlRails() << " control rails";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::establishConnection(const std::string &remote_agent) const {
    // Use existing connection_state_mutex_ to serialize connection establishment
    std::lock_guard<std::mutex> lock(connection_state_mutex_);

    // Check if another thread already established the connection
    auto it = connections_.find(remote_agent);
    if (it != connections_.end() && it->second->overall_state_ == ConnectionState::CONNECTED) {
        NIXL_DEBUG << "Connection already established by another thread for " << remote_agent;
        return NIXL_SUCCESS;
    }

    if (it == connections_.end()) {
        NIXL_ERROR << "No connection found for agent: " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    // Verify we have addresses for all data rails
    if (it->second->rail_remote_addr_list_.size() != rail_manager.getNumDataRails()) {
        NIXL_ERROR << "Remote connection has " << it->second->rail_remote_addr_list_.size()
                   << " data rails, expected " << rail_manager.getNumDataRails();
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Establishing connections_ on control rails and data rails for agent: "
               << remote_agent;

    // Use single "Communicator" for CM
    auto *conn_info = it->second.get();
    if (!conn_info) {
        NIXL_ERROR << "Connection info for agent " << remote_agent << " is null";
        return NIXL_ERR_BACKEND;
    }

    NIXL_DEBUG << "Using connection info with " << conn_info->src_ep_names_.size()
               << " data rails and " << conn_info->control_ep_names_.size() << " control rails";
    for (size_t i = 0; i < conn_info->src_ep_names_.size(); ++i) {
        NIXL_DEBUG << "Data rail " << i << ": "
                   << LibfabricUtils::hexdump(conn_info->src_ep_names_[i], LF_EP_NAME_MAX_LEN);
    }
    for (size_t i = 0; i < conn_info->control_ep_names_.size(); ++i) {
        NIXL_DEBUG << "Control rail " << i << ": "
                   << LibfabricUtils::hexdump(conn_info->control_ep_names_[i], LF_EP_NAME_MAX_LEN);
    }
    NIXL_DEBUG << "Agent index: " << it->second->agent_index_;

    conn_info->overall_state_ = ConnectionState::CONNECTED;
    NIXL_DEBUG << "Connection state for agent " << remote_agent << " is now "
               << conn_info->overall_state_;

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/

nixl_mem_list_t
nixlLibfabricEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
#ifdef HAVE_CUDA
    if (runtime_ == FI_HMEM_CUDA) {
        NIXL_DEBUG << "CUDA runtime detected, adding VRAM support";
        mems.push_back(VRAM_SEG);
    } else {
        NIXL_DEBUG << "Non-CUDA runtime, skipping VRAM support";
    }
#endif
    return mems;
}

nixl_status_t
nixlLibfabricEngine::registerMem(const nixlBlobDesc &mem,
                                 const nixl_mem_t &nixl_mem,
                                 nixlBackendMD *&out) {
    const auto supported = getSupportedMems();
    if (std::find(supported.begin(), supported.end(), nixl_mem) == supported.end()) {
        NIXL_ERROR << "Memory type " << nixl_mem << " is not supported by libfabric backend.";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    auto priv = std::make_unique<nixlLibfabricPrivateMetadata>();

    priv->buffer_ = (void *)mem.addr;
    priv->length_ = mem.len;
    priv->device_id_ = mem.devId; // Store device ID

    std::string pci_bus_id = "";

    // Use system runtime type to determine device-specific operations
    if (nixl_mem == VRAM_SEG) {
#ifdef HAVE_CUDA
        if (runtime_ == FI_HMEM_CUDA) {
            // CUDA-specific address query
            // For multi-GPU support, skip CUDA address workaround
            if (cuda_addr_wa_) {
                bool need_restart;
                if (vramUpdateCtx((void *)mem.addr, mem.devId, need_restart)) {
                    NIXL_WARN << "CUDA address workaround failed for device " << mem.devId
                              << ", disabling workaround for multi-GPU support";
                    cuda_addr_wa_ = false; // Disable workaround for subsequent registrations
                } else if (need_restart) {
                    // Restart progress thread if needed
                    NIXL_DEBUG << "CUDA context updated, restarting progress thread";
                    vramApplyCtx();
                }
            } else {
                // Set CUDA device context directly for multi-GPU support
                cudaError_t cuda_ret = cudaSetDevice(mem.devId);
                if (cuda_ret != cudaSuccess) {
                    NIXL_ERROR << "Failed to set CUDA device " << mem.devId << ": "
                               << cudaGetErrorString(cuda_ret);
                    return NIXL_ERR_NOT_SUPPORTED;
                }
                NIXL_DEBUG << "Set CUDA device context to GPU " << mem.devId;
            }

            // Query PCI bus ID from memory address (AFTER setting context)
            bool is_dev;
            CUdevice dev;
            CUcontext ctx;

            int ret = cudaQueryAddr((void *)mem.addr, is_dev, dev, ctx, pci_bus_id);
            if (ret || !is_dev) {
                NIXL_ERROR << "Failed to query device from memory " << (void *)mem.addr;
                return NIXL_ERR_BACKEND;
            }

            NIXL_DEBUG << "Queried PCI bus ID: " << pci_bus_id << " for GPU " << mem.devId;
        }
#endif
        if (runtime_ == FI_HMEM_NEURON) {
            // Neuron-specific address query
            int ret = nrtQueryAddr((void *)mem.addr, &pci_bus_id);
            if (ret) {
                NIXL_ERROR << "Could not query EFA device from memory " << (void *)mem.addr;
                // Fall back to all rails.
            }
            NIXL_DEBUG << "Queried PCI bus ID: " << pci_bus_id << " for Neuron device "
                       << mem.devId;
        }
    }

    // Initialize vectors to accommodate all possible rails (for indexing consistency)
    priv->rail_mr_list_.resize(rail_manager.getNumDataRails(), nullptr);
    priv->rail_key_list_.clear();
    priv->rail_key_list_.resize(rail_manager.getNumDataRails(), FI_KEY_NOTAVAIL);

#ifdef HAVE_CUDA
    // Set CUDA context before libfabric operations for VRAM
    if (nixl_mem == VRAM_SEG && runtime_ == FI_HMEM_CUDA) {
        vramApplyCtx();
    }
#endif

    // Use Rail Manager for centralized memory registration with GPU Direct RDMA support
    NIXL_TRACE << "Registering memory: addr=" << (void *)mem.addr << " len=" << mem.len
               << " mem_type=" << nixl_mem << " devId=" << mem.devId
               << (nixl_mem == VRAM_SEG ? " pci_bus_id=" + pci_bus_id : "");

    nixl_status_t status = rail_manager.registerMemory((void *)mem.addr,
                                                       mem.len,
                                                       nixl_mem,
                                                       mem.devId,
                                                       pci_bus_id,
                                                       priv->rail_mr_list_,
                                                       priv->rail_key_list_,
                                                       priv->selected_rails_);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager registerMemory failed";
        return status;
    }

    NIXL_DEBUG << "Rail Manager successfully registered "
               << (nixl_mem == VRAM_SEG ? "VRAM" : "DRAM") << " memory on "
               << priv->selected_rails_.size() << " rails"
               << (nixl_mem == VRAM_SEG ? " with GPU Direct RDMA support" : "");

    NIXL_DEBUG << "Successfully registered memory on " << priv->selected_rails_.size()
               << " rails for " << (nixl_mem == VRAM_SEG ? "accelerator" : "CPU") << " device "
               << mem.devId;
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::deregisterMem(nixlBackendMD *meta) {
    auto *priv = static_cast<nixlLibfabricPrivateMetadata *>(meta);
    // Use Rail Manager for centralized memory deregistration
    nixl_status_t status =
        rail_manager.deregisterMemory(priv->selected_rails_, priv->rail_mr_list_);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager deregisterMemory failed";
        // Continue with cleanup even if deregistration failed
    }

    delete priv;
    return status;
}

nixl_status_t
nixlLibfabricEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    const nixlLibfabricPrivateMetadata *priv =
        static_cast<const nixlLibfabricPrivateMetadata *>(meta);

    return rail_manager.serializeMemoryKeys(priv->rail_key_list_, priv->buffer_, str);
}

nixl_status_t
nixlLibfabricEngine::loadMetadataHelper(const std::vector<uint64_t> &rail_keys,
                                        void *buffer,
                                        std::shared_ptr<nixlLibfabricConnection> conn,
                                        nixlBackendMD *&output) {
    auto pub_md = std::make_unique<nixlLibfabricPublicMetadata>();

    pub_md->rail_remote_key_list_ = std::move(rail_keys);
    pub_md->derive_remote_selected_endpoints();
    pub_md->remote_buf_addr_ = reinterpret_cast<uint64_t>(buffer);
    pub_md->conn_ = conn;

    NIXL_DEBUG << "Metadata loaded with" << " Remote addr: " << (void *)pub_md->remote_buf_addr_
               << " Remote keys for " << pub_md->rail_remote_key_list_.size() << " rails"
               << " Remote fi_addr: " << pub_md->conn_->rail_remote_addr_list_[0][0];
    output = pub_md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    nixlLibfabricPrivateMetadata *input_md = static_cast<nixlLibfabricPrivateMetadata *>(input);
    return loadMetadataHelper(
        input_md->rail_key_list_, input_md->buffer_, connections_[localAgent], output);
}

nixl_status_t
nixlLibfabricEngine::loadRemoteMD(const nixlBlobDesc &input,
                                  const nixl_mem_t &nixl_mem,
                                  const std::string &remote_agent,
                                  nixlBackendMD *&output) {
    NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

    auto conn_it = connections_.find(remote_agent);
    if (conn_it == connections_.end()) {
        NIXL_ERROR << "Could not find connection for agent: " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    // Delegate to Rail Manager for SerDes operations (returns raw data)
    std::vector<uint64_t> remote_keys;
    uint64_t remote_addr;
    nixl_status_t status =
        rail_manager.deserializeMemoryKeys(input.metaInfo,
                                           conn_it->second->rail_remote_addr_list_.at(0).size(),
                                           remote_keys,
                                           remote_addr);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager deserializeMemoryKeys failed";
        return status;
    }

    return loadMetadataHelper(
        remote_keys, reinterpret_cast<void *>(remote_addr), conn_it->second, output);
}

nixl_status_t
nixlLibfabricEngine::unloadMD(nixlBackendMD *input) {
    delete input;
    return NIXL_SUCCESS;
}

/****************************************
 * Public Metadata Methods
 *****************************************/

void
nixlLibfabricPublicMetadata::derive_remote_selected_endpoints() {
    remote_selected_endpoints_.clear();

    for (size_t i = 0; i < rail_remote_key_list_.size(); ++i) {
        if (rail_remote_key_list_[i] != FI_KEY_NOTAVAIL) {
            remote_selected_endpoints_.push_back(i);
        } else {
            NIXL_DEBUG << "Skipping remote endpoint " << i << " with FI_KEY_NOTAVAIL";
        }
    }
}

/****************************************
 * Data movement
 *****************************************/

nixl_status_t
nixlLibfabricEngine::prepXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {
    NIXL_DEBUG << "Preparing transfer for remote_agent: " << remote_agent;

    auto conn_it = connections_.find(remote_agent);
    if (conn_it == connections_.end() || !conn_it->second) {
        NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    auto backend_handle = new nixlLibfabricBackendH(operation, remote_agent);
    if (!backend_handle) {
        NIXL_ERROR << "Failed to allocate nixlLibfabricBackendH";
        return NIXL_ERR_BACKEND;
    }

    // Set agent name and message in BinaryNotification during prepXfer
    if (opt_args && opt_args->hasNotif) {
        backend_handle->has_notif = true;

        // Use common fragmentation helper function
        fragmentNotificationMessage(opt_args->notifMsg,
                                    localAgent,
                                    backend_handle->total_notif_msg_len,
                                    backend_handle->binary_notifs);

        NIXL_DEBUG << "prepXfer: Fragmented notification into "
                   << backend_handle->binary_notifs.size()
                   << " fragments, total_length=" << backend_handle->total_notif_msg_len;
    }

    handle = backend_handle; // Assign to base class pointer

    NIXL_DEBUG << "Transfer preparation complete, handle address: " << handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::estimateXferCost(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH *const &handle,
                                      std::chrono::microseconds &duration,
                                      std::chrono::microseconds &err_margin,
                                      nixl_cost_t &method,
                                      const nixl_opt_args_t *opt_args) const {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::postXfer(const nixl_xfer_op_t &operation,
                              const nixl_meta_dlist_t &local,
                              const nixl_meta_dlist_t &remote,
                              const std::string &remote_agent,
                              nixlBackendReqH *&handle,
                              const nixl_opt_b_args_t *opt_args) const {

    // Validate connection
    auto conn_it = connections_.find(remote_agent);
    if (conn_it == connections_.end() || !conn_it->second) {
        NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    if (conn_it->second->overall_state_ == ConnectionState::DISCONNECTED) {
        NIXL_DEBUG << "No existing connection for " << remote_agent
                   << ", establishing new connection";
        nixl_status_t status = this->establishConnection(remote_agent);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to establish connection with " << remote_agent;
            return status;
        }
        NIXL_DEBUG << "Established new connection with remote_agent: " << remote_agent;
    }

    NIXL_DEBUG << "Posting transfer for remote_agent: " << remote_agent
               << ", handle address: " << handle;

    auto backend_handle = static_cast<nixlLibfabricBackendH *>(handle);
    if (!backend_handle) {
        NIXL_ERROR << "Failed to cast handle to nixlLibfabricBackendH";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Allocate xfer_id once in prepXfer
    backend_handle->post_xfer_id = LibfabricUtils::getNextXferId();

    nixlLibfabricReq::OpType op_type;
    int desc_count = local.descCount();

    NIXL_DEBUG << "Processing " << desc_count
               << " descriptors using optimized single-pass approach";

    op_type = (operation == NIXL_WRITE) ? nixlLibfabricReq::WRITE : nixlLibfabricReq::READ;

    // Set initial submit request count to maximum possible requests for this xfer.
    size_t max_possible_requests = desc_count * rail_manager.getNumDataRails();
    backend_handle->init_request_tracking(max_possible_requests);

    size_t total_submitted = 0;

    // Core transfer submission to process each descriptor with direct submission
    for (int desc_idx = 0; desc_idx < desc_count; ++desc_idx) {
        auto *local_md = static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP);
        auto *remote_md = static_cast<nixlLibfabricPublicMetadata *>(remote[desc_idx].metadataP);
        if (!local_md || !remote_md || !remote_md->conn_) {
            NIXL_ERROR << "Invalid metadata pointers for descriptor " << desc_idx;
            return NIXL_ERR_INVALID_PARAM;
        }

        // Validate connection for this descriptor
        if (remote_md->conn_ != conn_it->second) {
            NIXL_ERROR << "Connection mismatch for descriptor " << desc_idx;
            return NIXL_ERR_MISMATCH;
        }
        // Get transfer info for THIS descriptor
        void *transfer_addr = (void *)local[desc_idx].addr;
        size_t transfer_size = local[desc_idx].len;
        int device_id = local[desc_idx].devId;

        NIXL_DEBUG << "Processing descriptor " << desc_idx << " device " << device_id
                   << " local_addr: " << transfer_addr << " size=" << transfer_size
                   << " remote_addr=" << (void *)remote[desc_idx].addr;

        NIXL_DEBUG << "DEBUG: remote_agent='" << remote_agent << "' localAgent='" << localAgent
                   << "'";

        // Prepare and submit transfer for remote agents
        // Use descriptor's specific target address
        uint64_t remote_target_addr = remote[desc_idx].addr;

        size_t submitted_count = 0;
        nixl_status_t status = rail_manager.prepareAndSubmitTransfer(
            op_type,
            transfer_addr,
            transfer_size,
            remote_target_addr,
            local_md->selected_rails_,
            local_md->rail_mr_list_,
            remote_md->rail_remote_key_list_,
            remote_md->remote_selected_endpoints_,
            conn_it->second->rail_remote_addr_list_,
            conn_it->second->agent_index_,
            backend_handle->post_xfer_id,
            [backend_handle]() {
                backend_handle->increment_completed_requests();
            }, // Completion callback
            submitted_count);

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "prepareAndSubmitTransfer failed for descriptor " << desc_idx
                       << " device " << device_id;
            return status;
        }

        // Add submitted requests to the total count
        total_submitted += submitted_count;

        NIXL_DEBUG << "Successfully processed descriptor " << desc_idx << " with "
                   << submitted_count << " requests submitted (accumulated: " << total_submitted
                   << ")";
    }

    NIXL_DEBUG << "Processing complete: submitted " << total_submitted << " requests from "
               << desc_count << " descriptors" << " for xfer_id" << backend_handle->post_xfer_id;

    // For same-agent transfers, override to 0 since we bypassed all rail operations
    if (remote_agent == localAgent) {
        backend_handle->adjust_total_submitted_requests(0);
        NIXL_DEBUG << "Same-agent transfer: adjusted total requests to 0 (all handled via memcpy)";
    } else {
        // Adjust to actual request count after all submissions complete
        backend_handle->adjust_total_submitted_requests(total_submitted);
    }

    // Send notification immediately after successful request submission
    if (backend_handle->has_notif && backend_handle->operation_ == nixl_xfer_op_t::NIXL_WRITE) {
        nixl_status_t notif_status = notifSendPriv(remote_agent,
                                                   backend_handle->binary_notifs,
                                                   backend_handle->total_notif_msg_len,
                                                   backend_handle->post_xfer_id,
                                                   backend_handle->get_submitted_requests_count());
        if (notif_status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to send notification";
            return notif_status;
        }
        NIXL_DEBUG << "Notification sent immediately with XFER_ID=" << backend_handle->post_xfer_id
                   << ", expected_completions: " << backend_handle->get_submitted_requests_count();
    }

    // Progress data rails to kick off transfers
    if (!progress_thread_enabled_) {
        nixl_status_t progress_status = rail_manager.progressActiveDataRails();
        if (progress_status == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }
    }

    // For very small transfers we can check for local completions immediately.
    if (backend_handle->is_completed()) {
        if (backend_handle->has_notif && backend_handle->operation_ == nixl_xfer_op_t::NIXL_READ) {
            nixl_status_t notif_status = notifSendPriv(remote_agent,
                                                       backend_handle->binary_notifs,
                                                       backend_handle->total_notif_msg_len,
                                                       backend_handle->post_xfer_id,
                                                       0);
            if (notif_status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to send notification";
                return notif_status;
            }
        }
        return NIXL_SUCCESS;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricEngine::checkXfer(nixlBackendReqH *handle) const {
    auto backend_handle = static_cast<nixlLibfabricBackendH *>(handle);

    if (!progress_thread_enabled_) {
        nixl_status_t progress_status = rail_manager.progressActiveDataRails();
        if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress data rails in checkXfer";
            return progress_status;
        }
    }
    // Then check for completions after processing any pending completions
    if (backend_handle->is_completed()) {
        NIXL_DEBUG << "Data transfer completed successfully";
        if (backend_handle->has_notif && backend_handle->operation_ == nixl_xfer_op_t::NIXL_READ) {
            nixl_status_t notif_status = notifSendPriv(backend_handle->remote_agent_,
                                                       backend_handle->binary_notifs,
                                                       backend_handle->total_notif_msg_len,
                                                       backend_handle->post_xfer_id,
                                                       0);
            if (notif_status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to send notification";
                return notif_status;
            }
        }
        return NIXL_SUCCESS;
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricEngine::releaseReqH(nixlBackendReqH *handle) const {
    // Add any necessary cleanup for libfabric specific request handling
    // For example, if we're using a custom request structure:
    // nixlLibfabricReqH* req = static_cast<nixlLibfabricReqH*>(handle);
    // // Perform any necessary cleanup
    // delete req;

    if (!handle) {
        return NIXL_SUCCESS;
    }

    // Let NIXL framework handle the deletion
    NIXL_DEBUG << "releaseReqH completed successfully";
    return NIXL_SUCCESS;
}

/****************************************
 * Notification Functions
 *****************************************/

void
nixlLibfabricEngine::fragmentNotificationMessage(
    const std::string &message,
    const std::string &agent_name,
    uint32_t &total_message_length,
    std::vector<BinaryNotification> &fragments_out) const {
    // agent_name + message forms a single combined payload
    std::string combined_payload = agent_name + message;
    total_message_length = static_cast<uint32_t>(combined_payload.length());

    const size_t max_control_msg_size = BinaryNotification::MAX_FRAGMENT_SIZE;

    // Calculate fragment 0 capacity (has extra headers)
    size_t frag0_overhead = sizeof(BinaryNotificationHeader) + sizeof(BinaryNotificationMetadata);
    size_t frag0_capacity = max_control_msg_size - frag0_overhead;

    // Calculate fragment 1+ capacity (only has minimal header)
    size_t frag_overhead = sizeof(BinaryNotificationHeader);
    size_t frag_capacity = max_control_msg_size - frag_overhead;

    // Calculate number of fragments needed
    size_t num_fragments = 1; // At least fragment 0
    size_t remaining = 0;
    if (total_message_length > frag0_capacity) {
        remaining = total_message_length - frag0_capacity;
        num_fragments += (remaining + frag_capacity - 1) / frag_capacity;
    }

    fragments_out.clear();
    fragments_out.resize(num_fragments);

    NIXL_DEBUG << "Fragmenting: agent_name=" << agent_name.length()
               << "B, message=" << message.length()
               << "B, combined_payload=" << total_message_length << "B, fragments=" << num_fragments
               << ", frag0_capacity=" << frag0_capacity << ", frag_capacity=" << frag_capacity;

    size_t offset = 0;

    for (size_t frag_idx = 0; frag_idx < num_fragments; ++frag_idx) {
        // Set header fields
        BinaryNotificationHeader header;
        header.notif_xfer_id = 0; // Will be set later in notifSendPriv
        header.notif_seq_id = static_cast<uint16_t>(frag_idx);
        header.notif_seq_len = static_cast<uint16_t>(num_fragments);

        if (frag_idx == 0) {
            // Fragment 0: Pack metadata + combined_payload_chunk
            size_t payload_chunk_len =
                std::min(frag0_capacity, static_cast<size_t>(total_message_length));
            header.payload_length = static_cast<uint32_t>(payload_chunk_len);

            fragments_out[0].setHeader(header);
            fragments_out[0].setMetadata(total_message_length,
                                         0, // expected_completions set later
                                         static_cast<uint16_t>(agent_name.length()));
            // Set the payload chunk directly
            fragments_out[0].setPayload(combined_payload.substr(0, payload_chunk_len));

            offset = payload_chunk_len;

            NIXL_DEBUG << "Fragment 0: combined_payload_chunk=" << payload_chunk_len << "B";
        } else {
            // Fragment 1+: Pack only combined_payload continuation
            size_t payload_chunk_len =
                std::min(frag_capacity, static_cast<size_t>(total_message_length) - offset);
            header.payload_length = static_cast<uint32_t>(payload_chunk_len);

            fragments_out[frag_idx].setHeader(header);
            // Set the payload chunk directly
            fragments_out[frag_idx].setPayload(combined_payload.substr(offset, payload_chunk_len));

            offset += payload_chunk_len;

            NIXL_DEBUG << "Fragment " << frag_idx
                       << ": combined_payload_chunk=" << payload_chunk_len << "B";
        }
    }

    NIXL_DEBUG << "Fragmentation complete: " << num_fragments
               << " fragments, total_payload=" << total_message_length << "B";
}

// notifSendPriv that accepts vector of BinaryNotifications for fragmentation support
nixl_status_t
nixlLibfabricEngine::notifSendPriv(const std::string &remote_agent,
                                   std::vector<BinaryNotification> &binary_notifications,
                                   uint32_t total_message_length,
                                   uint16_t notif_xfer_id,
                                   uint32_t expected_completions) const {
    auto it = connections_.find(remote_agent);
    if (it == connections_.end()) {
        NIXL_ERROR << "No connection found for agent: " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    const auto &connection = it->second;
    const size_t control_rail_id = 0; // Only use control rail 0 for notifications

    NIXL_DEBUG << "Sending " << binary_notifications.size() << " notification fragments"
               << " total_message_length=" << total_message_length;

    // Send each notification fragment
    for (size_t seq_id = 0; seq_id < binary_notifications.size(); ++seq_id) {
        auto &binary_notification = binary_notifications[seq_id];

        // Update header fields for this notification
        BinaryNotificationHeader header = binary_notification.getHeader();
        header.notif_xfer_id = notif_xfer_id;
        binary_notification.setHeader(header);

        // Update first fragment header with expected_completions (only for fragment 0)
        // Note: agent_name_length was already set during fragmentation
        if (seq_id == 0) {
            const BinaryNotificationMetadata &metadata = binary_notification.getMetadata();
            binary_notification.setMetadata(
                total_message_length, expected_completions, metadata.agent_name_length);
        }

        // Allocate control request for this notification fragment
        size_t max_size = BinaryNotification::MAX_FRAGMENT_SIZE;
        nixlLibfabricReq *control_request = rail_manager.getControlRail(control_rail_id)
                                                .allocateControlRequest(max_size, notif_xfer_id);

        if (!control_request) {
            NIXL_ERROR << "Failed to allocate control request for notification fragment " << seq_id;
            return NIXL_ERR_BACKEND;
        }

        // Serialize BinaryNotification to control request buffer
        size_t serialized_size = binary_notification.serialize(control_request->buffer);
        control_request->buffer_size = serialized_size;

        NIXL_DEBUG << "Sending binary notification fragment " << seq_id << "/"
                   << binary_notifications.size() << " size=" << serialized_size << "B"
                   << " payload_chunk_size=" << header.payload_length << "B"
                   << " notif_xfer_id=" << header.notif_xfer_id;

        nixl_status_t status = rail_manager.postControlMessage(
            nixlLibfabricRailManager::ControlMessageType::NOTIFICATION,
            control_request,
            connection->control_rail_remote_addr_list_[control_rail_id][0],
            connection->agent_index_);

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "postControlMessage failed on control rail " << control_rail_id
                       << " for fragment " << seq_id;
            return NIXL_ERR_BACKEND;
        }

        // Progress the control rail to ensure the message is sent.
        status = rail_manager.progressAllControlRails();
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress control rails in notifSendPriv.";
            return status;
        }
    }

    NIXL_DEBUG << "Successfully sent all " << binary_notifications.size()
               << " notification fragments" << " total_length=" << total_message_length;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    // Use common fragmentation helper function
    uint32_t total_msg_len = 0;
    std::vector<BinaryNotification> notifications;
    fragmentNotificationMessage(msg, localAgent, total_msg_len, notifications);

    NIXL_DEBUG << "genNotif: Fragmented notification into " << notifications.size()
               << " fragments, total_length=" << total_msg_len;

    return notifSendPriv(remote_agent, notifications, total_msg_len, 0, 0);
}

nixl_status_t
nixlLibfabricEngine::getNotifs(notif_list_t &notif_list) {
    if (!progress_thread_enabled_) {
        nixl_status_t progress_status = rail_manager.progressActiveDataRails();
        if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress data rails in getNotifs";
            return progress_status;
        }
    }

    nixl_status_t progress_status = rail_manager.progressAllControlRails();
    if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to progress control rails in getNotifs.";
        return progress_status;
    }

    // Then check for available notifications after processing completions
    // Thread-safe access to internal notification list
    {
        std::lock_guard<std::mutex> lock(notif_mutex_);

        // Move all notifications from internal list to user's list
        notif_list.insert(notif_list.end(), notifMainList_.begin(), notifMainList_.end());

        if (!notifMainList_.empty()) {
            NIXL_DEBUG << "Retrieved " << notifMainList_.size() << " notifications";
            // Clear the internal list after copying
            notifMainList_.clear();
            return NIXL_SUCCESS;
        }

        // Clear the internal list after copying (even if empty)
        notifMainList_.clear();
    }

    return NIXL_IN_PROG;
}

/****************************************
 * Progress Thread Function (Data Rails Only)
 *****************************************/

// Progress thread that continuously processes completions only on data rails
nixl_status_t
nixlLibfabricEngine::progressThread() {
    NIXL_DEBUG << "PT: Thread started successfully for data rails only";
    // Main progress loop - continuously process completions only on data rails
    while (!progress_thread_stop_.load()) {
        // Process completions only on data rails (non-blocking)
        bool any_completions = false;
        nixl_status_t status = rail_manager.progressActiveDataRails();
        if (status == NIXL_SUCCESS) {
            any_completions = true;
            NIXL_DEBUG << "PT: Processed completions on data rails";
        } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
            NIXL_ERROR << "PT: Failed to process completions on data rails";
            // Don't return error, continue for robustness
        }
        if (!any_completions) {
            std::this_thread::sleep_for(progress_thread_delay_);
        }
    }
    NIXL_DEBUG << "PT: Thread exiting cleanly";
    return NIXL_SUCCESS;
}

/****************************************
 * Static Callback Functions
 *****************************************/

void
nixlLibfabricEngine::processNotification(const std::string &serialized_notif) {
    NIXL_DEBUG << "Received notification size=" << serialized_notif.size();

    // Deserialize binary notification
    BinaryNotification binary_notif;
    BinaryNotification::deserialize(serialized_notif.data(), serialized_notif.size(), binary_notif);

    // Extract fields
    const BinaryNotificationHeader &header = binary_notif.getHeader();
    uint16_t notif_xfer_id = header.notif_xfer_id;
    uint16_t notif_seq_id = header.notif_seq_id;
    uint16_t notif_seq_len = header.notif_seq_len;

    // Get payload chunk (combined agent_name + message chunk for all fragments)
    const std::string &payload_chunk = binary_notif.getPayload();

    // Get metadata from first fragment (only valid for fragment 0)
    uint32_t expected_completions = 0;
    uint32_t total_payload_length = 0;
    uint16_t agent_name_length = 0;
    if (notif_seq_id == 0) {
        const BinaryNotificationMetadata &metadata = binary_notif.getMetadata();
        expected_completions = metadata.expected_completions;
        total_payload_length = metadata.total_payload_length;
        agent_name_length = metadata.agent_name_length;
    }

    NIXL_TRACE << "Received notification fragment" << " notif_xfer_id=" << notif_xfer_id
               << " notif_seq_id=" << notif_seq_id << "/" << notif_seq_len
               << " payload_chunk_size=" << payload_chunk.size()
               << " expected_completions=" << expected_completions;

    {
        std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);

        // Use try_emplace to construct in-place - eliminates extra copy
        auto [it, inserted] = pending_notifications_.try_emplace(notif_xfer_id, notif_xfer_id);

        if (inserted) {
            NIXL_DEBUG << "Created pending notification" << " notif_xfer_id=" << notif_xfer_id
                       << " expected_completions=" << expected_completions
                       << " expected_msg_fragments=" << notif_seq_len;
        }

        // Initialize fragment vector on first fragment (check if vector is empty)
        if (it->second.message_fragments.empty()) {
            it->second.message_fragments.resize(notif_seq_len);
            it->second.expected_msg_fragments = notif_seq_len;
        }

        // Validate fragment index
        if (notif_seq_id >= notif_seq_len) {
            NIXL_ERROR << "Invalid fragment sequence: notif_seq_id=" << notif_seq_id
                       << " >= notif_seq_len=" << notif_seq_len;
            return;
        }

        // Check for duplicate fragment
        if (!it->second.message_fragments[notif_seq_id].empty()) {
            NIXL_WARN << "Duplicate fragment received: notif_seq_id=" << notif_seq_id;
            return;
        }

        // Store payload chunk (combined agent_name + message chunk)
        it->second.message_fragments[notif_seq_id] = payload_chunk;
        it->second.received_msg_fragments++;

        // Update metadata from fragment 0 (agent_name will be extracted after reassembly)
        if (notif_seq_id == 0) {
            it->second.expected_completions = expected_completions;
            it->second.total_message_length = total_payload_length;
            it->second.agent_name_length = agent_name_length;
        }

        NIXL_DEBUG << "Stored fragment" << " notif_xfer_id=" << notif_xfer_id << " fragment "
                   << notif_seq_id << "/" << notif_seq_len
                   << " received_msg_fragments=" << it->second.received_msg_fragments
                   << " expected_completions=" << it->second.expected_completions
                   << " received_completions=" << it->second.received_completions;
    }

    // Check if any notifications can now be completed (after releasing the lock)
    checkPendingNotifications();
}

/****************************************
 * Receiver Side XFER_ID Tracking Helper Methods
 *****************************************/

void
nixlLibfabricEngine::addReceivedXferId(uint16_t xfer_id) {
    {
        std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);

        // Use try_emplace to construct in-place - eliminates extra copy
        // First parameter: map key for lookup
        // Second parameter: constructor argument for PendingNotification
        auto [it, inserted] = pending_notifications_.try_emplace(xfer_id, xfer_id);

        if (inserted) {
            // Set placeholder values for write-arrived-first case
            it->second.remote_agent = "";
            it->second.expected_completions = INT_MAX;
            it->second.received_completions = 0;
            it->second.expected_msg_fragments = 1; // Default to 1 fragment
            it->second.received_msg_fragments = 0;
            NIXL_DEBUG << "Created placeholder notification for notif_xfer_id " << xfer_id
                       << " (write arrived first)";
        }

        it->second.received_completions++;
        NIXL_DEBUG << "Incremented received count for notif_xfer_id " << xfer_id << ": "
                   << it->second.received_completions << "/" << it->second.expected_completions;
    }

    // Check if any notifications can now be completed (after releasing the lock)
    checkPendingNotifications();
}

/****************************************
 * Notification Queuing Helper Methods
 *****************************************/

void
nixlLibfabricEngine::checkPendingNotifications() {
    std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
    auto it = pending_notifications_.begin();
    while (it != pending_notifications_.end()) {
        // Check BOTH conditions: fragments complete AND writes complete
        bool fragments_complete =
            (it->second.received_msg_fragments >= it->second.expected_msg_fragments);
        bool writes_complete = (it->second.received_completions >= it->second.expected_completions);

        if (fragments_complete && writes_complete) {
            NIXL_TRACE << "Notification complete: fragments=" << it->second.received_msg_fragments
                       << "/" << it->second.expected_msg_fragments
                       << " writes=" << it->second.received_completions << "/"
                       << it->second.expected_completions;

            // Reassemble combined payload from fragments
            std::string combined_payload;
            combined_payload.reserve(it->second.total_message_length);
            for (const auto &fragment : it->second.message_fragments) {
                combined_payload.append(fragment);
            }

            // Extract agent_name and message from combined payload
            uint16_t agent_name_len = it->second.agent_name_length;
            std::string remote_agent;
            std::string message;

            if (agent_name_len > 0 && combined_payload.size() >= agent_name_len) {
                remote_agent = combined_payload.substr(0, agent_name_len);
                if (combined_payload.size() > agent_name_len) {
                    message = combined_payload.substr(agent_name_len);
                }
            } else {
                NIXL_ERROR << "Invalid combined payload: agent_name_len=" << agent_name_len
                           << " combined_payload_size=" << combined_payload.size();
            }

            // Move notification to main list (need to acquire notif_mutex_)
            {
                std::lock_guard<std::mutex> notif_lock(notif_mutex_);
                notifMainList_.push_back({remote_agent, message});
            }

            NIXL_TRACE << "Processed queued notification from " << remote_agent
                       << " message_len=" << message.length();

            // Remove from pending list
            it = pending_notifications_.erase(it);
        } else {
            ++it;
        }
    }
}

void
nixlLibfabricEngine::cleanup() {
    NIXL_DEBUG << "Cleaning up all resources";
#ifdef HAVE_CUDA
    // Cleanup CUDA context
    vramFiniCtx();
#endif

    NIXL_DEBUG << "Cleanup all resources complete";
}
