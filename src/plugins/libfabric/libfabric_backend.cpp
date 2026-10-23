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
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <cstdint>
#include <dlfcn.h>
#include <cstring>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <iomanip>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>

namespace {

constexpr size_t NIXL_LIBFABRIC_DEFAULT_POST_THREADS = 0;
constexpr size_t NIXL_LIBFABRIC_DEFAULT_POST_SPLIT_BATCH_SIZE = 1024;
constexpr const char *NIXL_LIBFABRIC_POST_THREADS_PARAM = "num_threads";
constexpr const char *NIXL_LIBFABRIC_POST_SPLIT_BATCH_SIZE_PARAM = "split_batch_size";
constexpr size_t NIXL_LIBFABRIC_POST_THREAD_HW_MULTIPLIER = 4;

void
storeFirstError(std::atomic<int> &status, nixl_status_t new_status) {
    int expected = static_cast<int>(NIXL_SUCCESS);
    status.compare_exchange_strong(expected, static_cast<int>(new_status));
}

} // namespace

class nixlLibfabricPostThreadPool {
public:
    explicit nixlLibfabricPostThreadPool(size_t thread_count) {
        workers_.reserve(thread_count);
        try {
            for (size_t i = 0; i < thread_count; ++i) {
                workers_.emplace_back([this]() { workerLoop(); });
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to create libfabric post worker threads: " << e.what();
            stopAndJoin();
            throw;
        }
        catch (...) {
            NIXL_ERROR << "Failed to create libfabric post worker threads";
            stopAndJoin();
            throw;
        }
    }

    ~nixlLibfabricPostThreadPool() {
        stopAndJoin();
    }

    bool
    submit(std::function<void()> task) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                NIXL_WARN << "Ignoring libfabric post task submission after thread pool stop";
                return false;
            }
            tasks_.emplace_back(std::move(task));
        }
        cv_.notify_one();
        return true;
    }

private:
    void
    stopAndJoin() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        // Workers drain tasks queued before stop_; submit() rejects new tasks after this point.
        cv_.notify_all();

        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void
    workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop_front();
            }

            task();
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Accelerator vendor knowledge lives in two places now, neither of them here:
//   - src/utils/libfabric/nfi_hmem.cpp, the libfabric HMEM shim -- pointer attribution, the
//     fi_mr_attr::device arm, and thread-context push/pop. C-compatible interface, meant to be
//     donated upstream.
//   - src/utils/libfabric/libfabric_accel.{h,cpp}, the nixl-side policy and hwloc matching.
//
// What used to be here was a binder hierarchy plus a mediator so the progress thread could reach
// it. All of it is gone, and the reason is an ordering correction rather than a refactor: the old
// code bound a CUDA context *before* probing a pointer, on the belief that
// cuPointerGetAttributes() only sees pointers whose context is current. It does not -- unified
// virtual addressing lets the driver attribute any allocation in the process. What actually needs a
// current context is *registration*, because libfabric exports a dmabuf handle from inside
// fi_mr_regattr() for providers configured that way. UCX draws the line in the same place, for the
// same stated reason (uct_cuda_copy_md_mem_query).
//
// Probing first and binding second means there is nothing to latch, so the whole "address
// workaround" mode machine -- which existed only because it latched a context observed on the first
// registered pointer, and which permanently degraded the moment a second device appeared -- has no
// reason to exist. nixlAccelScope brackets the two operations that need a context.

/****************************************
 * Request Management
 *****************************************/

nixlLibfabricBackendH::nixlLibfabricBackendH(nixl_xfer_op_t op, const std::string &remote_agent)
    : completed_requests_(0),
      submitted_requests_(0),
      error_status_(NIXL_SUCCESS),
      successful_requests_(0),
      xfer_error_sent_(false),
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
    successful_requests_.store(0);
    error_status_.store(NIXL_SUCCESS);
    NIXL_DEBUG << "Initialized request tracking for " << num_requests << " requests";
}

void
nixlLibfabricBackendH::complete_request(nixl_status_t status) {
    if (status != NIXL_SUCCESS) {
        nixl_status_t expected = NIXL_SUCCESS;
        error_status_.compare_exchange_strong(
            expected, status, std::memory_order_relaxed, std::memory_order_relaxed);
    } else {
        successful_requests_.fetch_add(1, std::memory_order_relaxed);
    }
    // Release ensures the error store above is visible to any thread that
    // observes the incremented count via an acquire load in is_completed().
    completed_requests_.fetch_add(1, std::memory_order_release);
    NIXL_DEBUG << "Request completed (status=" << status
               << "), total completed: " << completed_requests_.load(std::memory_order_relaxed)
               << "/" << submitted_requests_.load(std::memory_order_relaxed);
}

size_t
nixlLibfabricBackendH::get_completed_requests_count() const {
    return completed_requests_.load();
}

size_t
nixlLibfabricBackendH::get_successful_requests_count() const {
    return successful_requests_.load(std::memory_order_acquire);
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

nixl_status_t
nixlLibfabricBackendH::get_error_status() const {
    return error_status_.load(std::memory_order_acquire);
}

bool
nixlLibfabricBackendH::claim_xfer_error_send() {
    return !xfer_error_sent_.exchange(true);
}

bool
nixlLibfabricBackendH::is_completed() const {
    // Acquire pairs with release in complete_request() — guarantees we see
    // any error_status_ written before the final increment.
    return completed_requests_.load(std::memory_order_acquire) ==
        submitted_requests_.load(std::memory_order_relaxed);
}

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlLibfabricEngine::nixlLibfabricEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      progress_thread_enabled_(init_params->enableProgTh),
      progress_thread_delay_(std::chrono::microseconds(init_params->pthrDelay)),
      rail_manager_(NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD),
      post_thread_count_(NIXL_LIBFABRIC_DEFAULT_POST_THREADS),
      post_split_batch_size_(NIXL_LIBFABRIC_DEFAULT_POST_SPLIT_BATCH_SIZE) {

    NIXL_INFO << "Initializing Libfabric Backend";

    // this is required for loading rail selection policy by configuration
    if (rail_manager_.init(getCustomParams()) != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to initialize the rail manager");
    }


    // The CUDA address workaround this used to switch off is gone (see the note above
    // nixlLibfabricBackendH), so the variable no longer does anything. Say so rather than let a
    // documented mitigation silently stop working.
    if (nixl::config::checkExistence("NIXL_DISABLE_CUDA_ADDR_WA")) {
        NIXL_WARN << "NIXL_DISABLE_CUDA_ADDR_WA is set but no longer has any effect: the libfabric "
                     "backend binds a CUDA context only around registration and posting, so there "
                     "is no address workaround to disable";
    }

    // Parse striping threshold parameter
    std::string threshold_str;
    striping_threshold_ = NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD;

    if (getInitParam("striping_threshold", threshold_str) == NIXL_SUCCESS) {
        try {
            striping_threshold_ = std::stoull(threshold_str);
            NIXL_INFO << "Striping threshold: " << striping_threshold_ << " bytes (custom)";
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Invalid striping_threshold value '" << threshold_str
                      << "', using default: " << striping_threshold_ << " bytes";
        }
    } else {
        NIXL_INFO << "Striping threshold: " << striping_threshold_ << " bytes (default)";
    }

    initPostThreadPool();
    // Initialize Rail Manager which will discover the topology and create all rails.
    try {
        NIXL_INFO << "Rail Manager created with " << rail_manager_.getNumRails() << " rails";

        // Set up notification + handshake callbacks on rail 0
        const size_t notification_rail_id = 0;
        NIXL_DEBUG << "Set notification + handshake processors for rail 0";
        rail_manager_.getRail(notification_rail_id)
            .setNotificationCallback(
                [this](const std::string &serialized_notif, uint32_t sender_peer_idx) {
                    processNotification(serialized_notif, sender_peer_idx);
                });
        rail_manager_.getRail(0).setHandshakeCallback(
            [this](const std::string &payload) { handleHandshake(payload); });
        rail_manager_.getRail(notification_rail_id)
            .setXferErrorCallback([this](uint16_t notif_xfer_id,
                                         uint32_t sender_peer_idx,
                                         uint32_t final_completions) {
                handleXferError(notif_xfer_id, sender_peer_idx, final_completions);
            });

        // Set up XFER_ID tracking callbacks for all rails
        NIXL_DEBUG << "Setting up XFER_ID tracking callbacks for " << rail_manager_.getNumRails()
                   << " rails";
        for (size_t rail_id = 0; rail_id < rail_manager_.getNumRails(); ++rail_id) {
            rail_manager_.getRail(rail_id).setXferIdCallback(
                [this](uint64_t imm_data, uint32_t sender_peer_idx) {
                    uint16_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(imm_data);
                    addReceivedXferId(xfer_id, sender_peer_idx);
                });
            NIXL_DEBUG << "Set XFER_ID callback for rail " << rail_id;
        }

        // Create self-connection
        std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints(
            rail_manager_.getNumRails());
        // Prepare rail endpoints
        for (size_t rail_id = 0; rail_id < rail_manager_.getNumRails(); ++rail_id) {
            std::memcpy(data_endpoints[rail_id].data(),
                        rail_manager_.getRail(rail_id).ep_name,
                        sizeof(rail_manager_.getRail(rail_id).ep_name));
        }
        // Create self-connection using common method
        nixl_status_t conn_status = createAgentConnection(localAgent, data_endpoints);
        if (conn_status != NIXL_SUCCESS) {
            throw std::runtime_error(
                "createAgentConnection failed for self-connection with status: " +
                std::to_string(conn_status));
        }

        NIXL_INFO << "Created self-connection for agent: " << localAgent << " on "
                  << rail_manager_.getNumRails() << " rails";

        // Start Progress thread for rail completion processing
        if (progress_thread_enabled_) {
            // in case of PT=1 we need to allocate post ring buffer per rail
            size_t post_queue_size = NIXL_LIBFABRC_DEFAULT_POST_QUEUE_SIZE;
            LibfabricUtils::getCustomIntParam(
                getCustomParams(), "post_queue_size", post_queue_size);

            for (size_t i = 0; i < rail_manager_.getNumRails(); ++i) {
                rail_manager_.getRail(i).setProgressThreadEnabled(true);
                if (!rail_manager_.getRail(i).initPostQueue(post_queue_size)) {
                    NIXL_ERROR << "Failed to initialize post-queue for rail " << i;
                    throw std::runtime_error("Failed to initialize the rail manager: unable to "
                                             "initialize rail post queue");
                }
            }

            NIXL_INFO << "Starting Progress thread for rails with delay: "
                      << progress_thread_delay_.count() << " microseconds";
            progress_thread_stop_ = false;
            progress_thread_ = std::thread(&nixlLibfabricEngine::progressThread, this);

            if (!progress_thread_.joinable()) {
                NIXL_ERROR << "Failed to start Progress thread";
                throw std::runtime_error("Failed to start Progress thread");
            }
            NIXL_INFO << "Progress thread started successfully";
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

void
nixlLibfabricEngine::initPostThreadPool() {
    LibfabricUtils::getCustomIntParam(
        getCustomParams(), NIXL_LIBFABRIC_POST_THREADS_PARAM, post_thread_count_);
    const size_t max_post_thread_count =
        static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency())) *
        NIXL_LIBFABRIC_POST_THREAD_HW_MULTIPLIER;
    if (post_thread_count_ > max_post_thread_count) {
        NIXL_WARN << "Capping libfabric post thread count from " << post_thread_count_ << " to "
                  << max_post_thread_count;
        post_thread_count_ = max_post_thread_count;
    }
    LibfabricUtils::getCustomIntParam(
        getCustomParams(), NIXL_LIBFABRIC_POST_SPLIT_BATCH_SIZE_PARAM, post_split_batch_size_);
    if (post_thread_count_ > 0) {
        post_thread_pool_ = std::make_unique<nixlLibfabricPostThreadPool>(post_thread_count_);
        NIXL_DEBUG << "Libfabric descriptor post thread pool enabled with " << post_thread_count_
                   << " threads, split_batch_size=" << post_split_batch_size_;
    } else {
        NIXL_DEBUG << "Libfabric descriptor post thread pool disabled";
    }
}

nixlLibfabricEngine::~nixlLibfabricEngine() {
    NIXL_DEBUG
        << "Destructor starting, stopping all threads FIRST to prevent timing report interruption";

    post_thread_pool_.reset();
    if (progress_thread_enabled_) {
        progress_thread_stop_.store(true);
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
    for (size_t rail_id = 0; rail_id < rail_manager_.getNumRails(); ++rail_id) {
        if (!rail_manager_.getRail(rail_id).endpoint) {
            NIXL_ERROR << "Rail " << rail_id << " endpoint not initialized";
            return NIXL_ERR_BACKEND;
        }
    }

    NIXL_DEBUG << "Retrieving local endpoint addresses for all " << rail_manager_.getNumRails()
               << " rails";

    // Use Rail Manager's connection SerDes method with "dest" prefix for remote consumption
    nixl_status_t status = rail_manager_.serializeConnectionInfo("dest", str);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager serializeConnectionInfo failed";
        return status;
    }

    NIXL_DEBUG << "Rail Manager serialized connection info for " << rail_manager_.getNumRails()
               << " rails, total size=" << str.length();

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                        const std::string &remote_conn_info) {

    NIXL_DEBUG << "Loading remote info for agent: " << remote_agent
               << ", info length=" << remote_conn_info.length() << ", info (hex): "
               << LibfabricUtils::hexdump(remote_conn_info.data(), remote_conn_info.length());

    if (remote_conn_info.empty()) {
        NIXL_ERROR << "Empty remote connection info received";
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_DEBUG << "Processing " << rail_manager_.getNumRails()
               << " rails for agent: " << remote_agent;

    // Use Rail Manager's connection SerDes method with "dest" prefix
    // (remote is sending us their endpoints as "dest")
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints;
    nixl_status_t status =
        rail_manager_.deserializeConnectionInfo("dest", remote_conn_info, data_endpoints);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager deserializeConnectionInfo failed with status: " << status;
        return status;
    }

    std::shared_ptr<nixlLibfabricConnection> conn_for_handshake;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);

        bool already_exists = (connections_.find(remote_agent) != connections_.end());
        if (already_exists) {
            NIXL_INFO << "Connection for " << remote_agent
                      << " already exists, skipping duplicate loadRemoteConnInfo";
            return NIXL_SUCCESS;
        }

        nixl_status_t conn_status = createAgentConnection(remote_agent, data_endpoints);
        if (conn_status != NIXL_SUCCESS) {
            NIXL_ERROR << "createAgentConnection failed with status: " << conn_status;
            return conn_status;
        }

        if (remote_agent != localAgent) {
            conn_for_handshake = connections_[remote_agent];
        }
    }

    if (conn_for_handshake) {
        nixl_status_t hs = sendHandshakeTo(*conn_for_handshake);
        if (hs != NIXL_SUCCESS) {
            NIXL_ERROR << "Handshake send to '" << remote_agent << "' failed with status " << hs
                       << "; their first transfers to us will land in the pre-handshake bucket";
            return hs;
        }
    }

    NIXL_INFO << "Successfully stored multirail connection for " << remote_agent << " on "
              << rail_manager_.getNumRails() << " rails";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::connect(const std::string &remote_agent) {
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);

        NIXL_DEBUG << "Connecting to agent: " << remote_agent
                   << ", connections_ size=" << connections_.size();

        auto it = connections_.find(remote_agent);
        if (it != connections_.end() &&
            it->second->overall_state_.load(std::memory_order_acquire) ==
                ConnectionState::CONNECTED) {
            NIXL_INFO << "Connection already established for " << remote_agent
                      << ", fi_addr=" << it->second->rail_remote_addr_list_[0][0];
            return NIXL_SUCCESS;
        }
    }

    nixl_status_t status = establishConnection(remote_agent);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to establish connection with " << remote_agent;
        return status;
    }

    NIXL_INFO << "Successfully established connection for " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(connection_state_mutex_);
    auto it = connections_.find(remote_agent);
    if (it == connections_.end()) {
        NIXL_WARN << "Disconnect failed. No metadata connection info for " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    nixl_status_t status = it->second->disconnect();
    // The address vector keeps the peer's endpoints, so a reconnect gets the same fi_addrs back
    // under a fresh agent index. Unmapping them here lets that map cleanly; left mapped, the
    // reconnect would look like two agents claiming one address.
    rail_manager_.unmapAgentAddresses(it->second->rail_remote_addr_list_,
                                      static_cast<uint32_t>(it->second->agent_index_));
    connections_.erase(it);
    return status;
}

nixl_status_t
nixlLibfabricEngine::createAgentConnection(
    const std::string &agent_name,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_rail_endpoints) {

    NIXL_DEBUG << "Creating connection for agent: " << agent_name;

    if (data_rail_endpoints.empty()) {
        NIXL_ERROR << "Remote agent " << agent_name << " published zero rail endpoints";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (data_rail_endpoints.size() != rail_manager_.getNumRails()) {
        NIXL_WARN << "Rail count mismatch (local: " << rail_manager_.getNumRails()
                  << ", remote: " << data_rail_endpoints.size() << ")";
    }

    auto existing = connections_.find(agent_name);
    if (existing != connections_.end()) {
        NIXL_INFO << "Connection already exists for agent: " << agent_name
                  << ", reusing existing connection";
        return NIXL_SUCCESS;
    }

    auto conn = std::make_shared<nixlLibfabricConnection>(agent_name, agent_names_.size());
    if (!conn) {
        NIXL_ERROR << "Failed to allocate connection object";
        return NIXL_ERR_BACKEND;
    }

    // Claim the index first, so a partial insertAllAddresses() failure can't hand it to the
    // next peer. The dead name a failure leaves is harmless: agent_names_ is used for its size.
    agent_names_.push_back(agent_name);

    conn->rail_remote_addr_list_.reserve(rail_manager_.getNumRails());

    // Process all rails in one operation
    // The AV insert must happen before sendHandshakeTo() further down the call chain:
    // the peer treats our handshake as proof that we can attribute its traffic.
    nixl_status_t data_status =
        rail_manager_.insertAllAddresses(data_rail_endpoints,
                                         static_cast<uint32_t>(conn->agent_index_),
                                         conn->rail_remote_addr_list_,
                                         conn->src_ep_names_);
    if (data_status != NIXL_SUCCESS) {
        NIXL_ERROR << "insertAllAddresses failed for rails with status: " << data_status;
        return data_status;
    }

    /*
     * Work out which of this peer's rails each local rail should actually post to. Every peer
     * endpoint was just inserted into every local rail's address vector above, so any pairing is
     * addressable; the question is only which one is reachable, and that is what this answers.
     */
    conn->peer_rail_for_local_rail_ = rail_manager_.computeRailPeering(data_rail_endpoints);

    /*
     * Control traffic has to land on the peer's rail 0, since that is where the notification and
     * handshake callbacks are installed. Pick the local rail paired with peer rail 0 so the send is
     * reachable; rail 0 stays the answer whenever pairing is the identity, which is the common case.
     */
    for (size_t rail_id = 0; rail_id < conn->peer_rail_for_local_rail_.size(); ++rail_id) {
        if (conn->peer_rail_for_local_rail_[rail_id] == 0) {
            conn->control_rail_ = rail_id;
            break;
        }
    }
    if (conn->control_rail_ != 0) {
        NIXL_INFO << "Control traffic to " << agent_name << " will use local rail "
                  << conn->control_rail_ << ", which is the rail paired with its rail 0";
    }

    for (size_t i = 0; i < agent_names_.size(); ++i) {
        NIXL_DEBUG << "Index " << i << ": " << agent_names_[i];
    }

    connections_[agent_name] = conn;

    // Drain any handshake the peer already sent us before we'd registered
    // them locally.
    if (agent_name != localAgent) {
        bool buffered = false;
        std::optional<uint16_t> rejected_ver;
        {
            std::lock_guard<std::mutex> plk(pending_handshake_mutex_);
            buffered = (pending_inbound_handshakes_.erase(agent_name) > 0);
            auto rejected = pending_rejected_handshakes_.find(agent_name);
            if (rejected != pending_rejected_handshakes_.end()) {
                rejected_ver = rejected->second;
                pending_rejected_handshakes_.erase(rejected);
            }
        }
        if (rejected_ver) {
            std::lock_guard<std::mutex> hlk(conn->handshake_mutex_);
            conn->rejected_peer_proto_ver_ = *rejected_ver;
            conn->handshake_rejected_.store(true, std::memory_order_release);
        }
        if (buffered) {
            {
                std::lock_guard<std::mutex> hlk(conn->handshake_mutex_);
                conn->handshake_received_.store(true, std::memory_order_release);
            }
            conn->handshake_cv_.notify_all();
            NIXL_INFO << "Applied buffered handshake from '" << agent_name << "'";
        }
    }

    NIXL_INFO << "Successfully created connection for agent: " << agent_name << " on "
              << rail_manager_.getNumRails() << " rails";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::establishConnection(const std::string &remote_agent) const {
    std::shared_ptr<nixlLibfabricConnection> conn;
    {
        // Use existing connection_state_mutex_ to serialize connection establishment
        std::lock_guard<std::mutex> lock(connection_state_mutex_);

        auto it = connections_.find(remote_agent);
        if (it == connections_.end()) {
            NIXL_ERROR << "No connection found for agent: " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }

        if (it->second->overall_state_.load(std::memory_order_acquire) ==
            ConnectionState::CONNECTED) {
            NIXL_DEBUG << "Connection already established for " << remote_agent;
            return NIXL_SUCCESS;
        }
        conn = it->second;
    }

    // Wait for the peer's inbound handshake so we know our agent_idx at their side.
    if (conn->remoteAgent_ != localAgent &&
        !conn->handshake_received_.load(std::memory_order_acquire)) {

        using namespace std::chrono_literals;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(NIXL_LIBFABRIC_HANDSHAKE_TIMEOUT_S);

        auto handshake_done = [&] {
            return conn->handshake_received_.load(std::memory_order_acquire) ||
                conn->handshake_rejected_.load(std::memory_order_acquire);
        };
        if (progress_thread_enabled_) {
            std::unique_lock<std::mutex> lk(conn->handshake_mutex_);
            conn->handshake_cv_.wait_until(lk, deadline, handshake_done);
        } else {
            while (std::chrono::steady_clock::now() < deadline) {
                if (handshake_done()) {
                    break;
                }
                nixl_status_t progress_status = rail_manager_.progressActiveRails();
                if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
                    NIXL_ERROR << "Failed to progress rails when waiting for handshake.";
                    return progress_status;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        if (conn->handshake_rejected_.load(std::memory_order_acquire)) {
            uint16_t peer_ver;
            {
                std::lock_guard<std::mutex> hlk(conn->handshake_mutex_);
                peer_ver = conn->rejected_peer_proto_ver_;
            }
            NIXL_ERROR << "Peer '" << remote_agent << "' speaks libfabric-plugin protocol version "
                       << peer_ver << ", we speak " << NIXL_LIBFABRIC_PROTO_VERSION
                       << "; connection cannot be established. Upgrade both ends of the cluster "
                          "to the same NIXL version.";
            return NIXL_ERR_MISMATCH;
        }
        if (!conn->handshake_received_.load(std::memory_order_acquire)) {
            NIXL_ERROR << "Handshake from peer '" << remote_agent << "' not received after "
                       << NIXL_LIBFABRIC_HANDSHAKE_TIMEOUT_S
                       << "s; connection cannot be established. The peer may be unreachable, "
                          "or speak a different libfabric-plugin protocol version (we speak "
                       << NIXL_LIBFABRIC_PROTO_VERSION << "); check the peer's log.";
            return NIXL_ERR_REMOTE_DISCONNECT;
        }
    }

    // Mark connected only after handshake wait completes
    return conn->establish();
}

/****************************************
 * Memory management
 *****************************************/

nixl_mem_list_t
nixlLibfabricEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);

    // Any usable runtime is enough: VRAM_SEG is vendor-blind, and the shim decides which runtime
    // owns a given buffer at registration time. Asks every vendor SDK, not just the first usable
    // one the rail manager names in its log.
    if (nfi_hmem_available()) {
        NIXL_DEBUG << "Accelerator runtime available, adding VRAM support";
        mems.push_back(VRAM_SEG);
    } else {
        NIXL_DEBUG << "No accelerator runtime, skipping VRAM support";
    }
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

    struct nfi_hmem_info &hmem_info = priv->hmem_info_;

    if (nixl_mem == VRAM_SEG) {
        /*
         * Probe the pointer once, and once only, with no accelerator context bound. Attribution
         * needs none -- unified virtual addressing lets the driver identify any allocation in the
         * process -- and the result is threaded down through rail_manager_.registerMemory() rather
         * than re-derived there, so no two layers can disagree about one address. Binding happens
         * further down, around the registration calls that actually require it.
         */
        if (nfi_hmem_query_ptr((void *)mem.addr, &hmem_info) != 0) {
            NIXL_ERROR << "Failed to query device from memory " << (void *)mem.addr;
            return NIXL_ERR_BACKEND;
        }

        if (hmem_info.iface == FI_HMEM_SYSTEM) {
            const enum fi_hmem_iface fallback_iface = accelUnclaimedVramIface();
            if (fallback_iface == FI_HMEM_SYSTEM) {
                // The failure this path has always reported when cudaQueryAddr() said "not device
                // memory": on a host where CUDA is usable, or with no runtime that could have
                // missed the pointer, a VRAM_SEG none of them claims is a caller error.
                NIXL_ERROR << "Failed to query device from memory " << (void *)mem.addr;
                return NIXL_ERR_BACKEND;
            }
            /*
             * Otherwise recoverable, as before. Registering it as host memory would be wrong for a
             * provider that needs the iface, so guess the runtime that could have missed it and
             * take nixl's device index as the hmem ordinal. Only reached where CUDA is not usable;
             * see accelUnclaimedVramIface(). The bus ID stays empty, which makes getNicsForPci()
             * return every device.
             *
             * Resolved here, into priv->hmem_info_, rather than at registration: the context scope
             * below and every later post bind from this same record, so they must see the guess
             * too, not the host-memory iface the probe reported.
             *
             * A guess on both counts: nixl's public API gives VRAM_SEG no vendor, and mem.devId is
             * a nixl device index, which only coincides with an hmem ordinal when the runtime
             * enumerates devices the same way nixl does. It happens to be a valid encoding for the
             * fi_mr_attr::device arms that take one -- CUDA takes a plain ordinal, and
             * fi_hmem_ze_device(0, n) == n -- while Neuron gets the -1 sentinel nfi_neuron_query()
             * would have reported. A vendor whose encoding is not the identity would need this to
             * go through the shim.
             */
            hmem_info.iface = fallback_iface;
            hmem_info.device = (fallback_iface == FI_HMEM_NEURON) ?
                static_cast<uint64_t>(-1) :
                static_cast<uint64_t>(mem.devId);
            NIXL_WARN << "No accelerator runtime claimed VRAM buffer " << (void *)mem.addr
                      << ", registering on all rails as iface "
                      << nfi_hmem_iface_name(fallback_iface) << " device " << mem.devId;
        }

        NIXL_DEBUG << "Queried PCI bus ID: " << accelBusIdString(hmem_info) << " (iface "
                   << nfi_hmem_iface_name(hmem_info.iface) << ", hmem device " << hmem_info.device
                   << ") for device " << mem.devId;
    }

    // Initialize vectors to accommodate all possible rails (for indexing consistency)
    priv->rail_mr_list_.resize(rail_manager_.getNumRails(), nullptr);
    priv->rail_key_list_.clear();
    priv->rail_key_list_.resize(rail_manager_.getNumRails(), FI_KEY_NOTAVAIL);


    // Use Rail Manager for centralized memory registration with GPU Direct RDMA support
    NIXL_TRACE << "Registering memory: addr=" << (void *)mem.addr << " len=" << mem.len
               << " mem_type=" << nixl_mem << " devId=" << mem.devId
               << (nixl_mem == VRAM_SEG ? " pci_bus_id=" + accelBusIdString(hmem_info) : "");

    /*
     * Bind here, not earlier. fi_mr_regattr() is where libfabric exports a dmabuf handle for
     * providers configured that way, and that export needs a context on the buffer's device. The
     * scope is a no-op for a runtime with no thread-current state and for host memory, so it is
     * constructed unconditionally rather than tested for; it pops on the way out, which is what
     * keeps an application thread's own context binding intact.
     */
    nixl_status_t status;
    {
        const nixlAccelScope accel_scope(hmem_info);
        if (!accel_scope.ok()) {
            // A runtime with nothing to bind reports success, so this is a real refusal: the
            // dmabuf export in fi_mr_regattr() would run with no context on the buffer's device.
            NIXL_ERROR << "Failed to bind accelerator device " << hmem_info.device << " (iface "
                       << nfi_hmem_iface_name(hmem_info.iface) << ") before registering "
                       << (void *)mem.addr;
            return NIXL_ERR_NOT_SUPPORTED;
        }

        status = rail_manager_.registerMemory((void *)mem.addr,
                                              mem.len,
                                              nixl_mem,
                                              mem.devId,
                                              hmem_info,
                                              priv->rail_mr_list_,
                                              priv->rail_key_list_,
                                              priv->selected_rails_);
    }
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
        rail_manager_.deregisterMemory(priv->selected_rails_, priv->rail_mr_list_);
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

    return rail_manager_.serializeMemoryKeys(priv->rail_key_list_, priv->buffer_, str);
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
    std::shared_ptr<nixlLibfabricConnection> conn;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        conn = connections_[localAgent];
    }
    nixlLibfabricPrivateMetadata *input_md = static_cast<nixlLibfabricPrivateMetadata *>(input);
    return loadMetadataHelper(input_md->rail_key_list_, input_md->buffer_, conn, output);
}

nixl_status_t
nixlLibfabricEngine::loadRemoteMD(const nixlBlobDesc &input,
                                  const nixl_mem_t &nixl_mem,
                                  const std::string &remote_agent,
                                  nixlBackendMD *&output) {
    NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

    std::shared_ptr<nixlLibfabricConnection> conn;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto conn_it = connections_.find(remote_agent);
        if (conn_it == connections_.end()) {
            NIXL_ERROR << "Could not find connection for agent: " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        conn = conn_it->second;
    }

    // Delegate to Rail Manager for SerDes operations (returns raw data)
    std::vector<uint64_t> remote_keys;
    uint64_t remote_addr;
    nixl_status_t status = rail_manager_.deserializeMemoryKeys(
        input.metaInfo, conn->rail_remote_addr_list_.at(0).size(), remote_keys, remote_addr);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Rail Manager deserializeMemoryKeys failed with status: " << status;
        return status;
    }

    return loadMetadataHelper(remote_keys, reinterpret_cast<void *>(remote_addr), conn, output);
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

    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto conn_it = connections_.find(remote_agent);
        if (conn_it == connections_.end() || !conn_it->second) {
            NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
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

int
nixlLibfabricEngine::batchingRail(const nixl_meta_dlist_t &local,
                                  int desc_idx,
                                  size_t xfer_base_offset) const {
    auto *md = static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP);
    if (!md || md->selected_rails_.empty()) {
        return -1;
    }
    // Striped descriptors are split across their rails and posted without FI_MORE, so they
    // are not part of the single-rail batching tracked here.
    if (rail_manager_.usesStriping(local[desc_idx].len, md->selected_rails_.size())) {
        return -1;
    }
    return (int)md->selected_rails_[nixlLibfabricRailManager::railSelectionIndex(
        xfer_base_offset, desc_idx, /*batch_write=*/true, md->selected_rails_.size())];
}

bool
nixlLibfabricEngine::useFiMore(int desc_idx,
                               int rail_id,
                               const std::vector<int> &last_desc_idx_per_rail,
                               std::vector<int> &posts_since_flush) const {
    if (rail_id < 0) {
        return false;
    }
    if (desc_idx == last_desc_idx_per_rail[rail_id] ||
        posts_since_flush[rail_id] == NIXL_LIBFABRIC_FI_MORE_BATCH_SIZE - 1) {
        posts_since_flush[rail_id] = 0;
        return false;
    }
    ++posts_since_flush[rail_id];
    return true;
}

namespace {
uint64_t
accelBindKey(const struct nfi_hmem_info &info) {
    return (static_cast<uint64_t>(info.iface) + 1) << 56 | (info.device & ((1ULL << 56) - 1));
}
} // namespace

bool
nixlLibfabricEngine::accelBindKnownOk(const struct nfi_hmem_info &info) const {
    const uint64_t key = accelBindKey(info);
    for (const auto &slot : accel_bind_ok_) {
        const uint64_t v = slot.load(std::memory_order_acquire);
        if (v == key) {
            return true;
        }
        if (v == 0) {
            return false; // slots fill in order, so nothing past an empty one
        }
    }
    return false;
}

void
nixlLibfabricEngine::accelBindRecordOk(const struct nfi_hmem_info &info) const {
    const uint64_t key = accelBindKey(info);
    for (auto &slot : accel_bind_ok_) {
        uint64_t expected = 0;
        if (slot.compare_exchange_strong(expected, key, std::memory_order_acq_rel) ||
            expected == key) {
            return;
        }
    }
    // Table full: this device will simply be probed again next time.
}

nixl_status_t
nixlLibfabricEngine::postXferDescriptors(nixlLibfabricReq::OpType op_type,
                                         const nixl_meta_dlist_t &local,
                                         const nixl_meta_dlist_t &remote,
                                         const std::shared_ptr<nixlLibfabricConnection> &conn,
                                         nixlLibfabricBackendH *backend_handle,
                                         int start_idx,
                                         int end_idx,
                                         size_t xfer_base_offset,
                                         bool allow_fi_more,
                                         size_t &submitted_count) const {
    submitted_count = 0;

    /*
     * A provider may take a host-copy path for a small transfer, which needs a current accelerator
     * context. When the progress thread is enabled the post is deferred via the ring buffer and the
     * binding has to happen on that thread instead, which is what the request's iface tells
     * drainPostQueue() to do; here we are the posting thread and do it ourselves, per descriptor.
     */
    const bool bind_accel_ctx = local.getType() == VRAM_SEG && !progress_thread_enabled_;

    // A FI_MORE post rings no doorbell; a rail's queued batch is only submitted by a later
    // non-FI_MORE post on the same rail. Rails are resolved per-buffer, so descriptors of one
    // round-robin group can land on different rails: every rail this chunk touches must have its
    // last post flushed, or its batch would never be submitted.
    // allow_fi_more is false on the thread-pool path: chunks on other threads can interleave
    // posts on the same rail, so a per-chunk walk cannot know a rail's true last post there.
    const bool batch_writes = allow_fi_more && op_type == nixlLibfabricReq::WRITE;

    std::vector<int> last_desc_idx_per_rail(rail_manager_.getNumRails(), -1);
    std::vector<int> posts_since_flush(rail_manager_.getNumRails(), 0);
    if (batch_writes) {
        for (int i = start_idx; i < end_idx; ++i) {
            const int rail_id = batchingRail(local, i, xfer_base_offset);
            if (rail_id >= 0) {
                last_desc_idx_per_rail[rail_id] = i;
            }
        }
    }

    /*
     * One scope across the descriptors, rebound only when a descriptor's buffer lives on another
     * device: consecutive descriptors usually share one, and a push/pop per descriptor put driver
     * calls on every post. Scoped to this call so the caller's own binding survives -- this may be
     * an application thread. A no-op for a runtime with no thread-current state, which is why it is
     * not guarded by a needsBinding() test.
     */
    std::optional<nixlAccelScope> accel_scope;

    for (int desc_idx = start_idx; desc_idx < end_idx; ++desc_idx) {
        const int rail_id = batch_writes ? batchingRail(local, desc_idx, xfer_base_offset) : -1;
        const bool apply_fi_more =
            useFiMore(desc_idx, rail_id, last_desc_idx_per_rail, posts_since_flush);

        auto *local_md = static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP);
        auto *remote_md = static_cast<nixlLibfabricPublicMetadata *>(remote[desc_idx].metadataP);

        void *transfer_addr = (void *)local[desc_idx].addr;
        size_t transfer_size = local[desc_idx].len;
        int device_id = local[desc_idx].devId;

        if (bind_accel_ctx && !(accel_scope && accel_scope->holds(local_md->hmem_info_))) {
            accel_scope.reset(); // pop the previous device before pushing the next
            accel_scope.emplace(local_md->hmem_info_);
            if (!accel_scope->ok()) {
                // postXfer() checked this device binds before posting anything, so only a
                // transient failure gets here.
                NIXL_ERROR << "Failed to bind accelerator device " << local_md->hmem_info_.device
                           << " (iface " << nfi_hmem_iface_name(local_md->hmem_info_.iface)
                           << ") while posting descriptor " << desc_idx;
                return NIXL_ERR_BACKEND;
            }
        }

        uint64_t remote_target_addr = remote[desc_idx].addr;
        uint64_t remote_registered_base = remote_md->remote_buf_addr_;

        size_t desc_submitted_count = 0;
        nixl_status_t status = rail_manager_.prepareAndSubmitTransfer(
            op_type,
            transfer_addr,
            transfer_size,
            remote_target_addr,
            remote_registered_base,
            local_md->selected_rails_,
            local_md->rail_mr_list_,
            remote_md->rail_remote_key_list_,
            remote_md->remote_selected_endpoints_,
            conn->peer_rail_for_local_rail_,
            conn->rail_remote_addr_list_,
            backend_handle->post_xfer_id,
            [backend_handle](nixl_status_t status) {
                backend_handle->complete_request(status);
            }, // Completion callback
            desc_submitted_count,
            desc_idx,
            xfer_base_offset,
            apply_fi_more,
            local_md->hmem_info_);

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "prepareAndSubmitTransfer failed for descriptor " << desc_idx
                       << " device " << device_id;
            return status;
        }

        submitted_count += desc_submitted_count;
    }

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
    std::shared_ptr<nixlLibfabricConnection> conn;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto conn_it = connections_.find(remote_agent);
        if (conn_it == connections_.end() || !conn_it->second) {
            NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        conn = conn_it->second;
    }

    if (conn->overall_state_.load(std::memory_order_acquire) == ConnectionState::DISCONNECTED) {
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

    // Update notification from opt_args on repost
    if (opt_args && opt_args->hasNotif) {
        backend_handle->has_notif = true;
        backend_handle->binary_notifs.clear();
        fragmentNotificationMessage(opt_args->notifMsg,
                                    localAgent,
                                    backend_handle->total_notif_msg_len,
                                    backend_handle->binary_notifs);
    } else if (opt_args && !opt_args->hasNotif) {
        backend_handle->has_notif = false;
        backend_handle->binary_notifs.clear();
        backend_handle->total_notif_msg_len = 0;
    }

    // Allocate xfer_id once in prepXfer
    backend_handle->post_xfer_id = LibfabricUtils::getNextXferId();

    nixlLibfabricReq::OpType op_type;
    int desc_count = local.descCount();

    NIXL_DEBUG << "Processing " << desc_count
               << " descriptors using optimized single-pass approach";

    op_type = (operation == NIXL_WRITE) ? nixlLibfabricReq::WRITE : nixlLibfabricReq::READ;

    // Set initial submit request count to maximum possible requests for this xfer.
    size_t max_possible_requests = desc_count * rail_manager_.getNumRails();
    backend_handle->init_request_tracking(max_possible_requests);

    size_t total_submitted = 0;

    // Validate metadata before posting. The parallel path may post descriptors out of order, so
    // simple input errors should be caught before any worker submits RDMA operations.
    for (int desc_idx = 0; desc_idx < desc_count; ++desc_idx) {
        auto *local_md = static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP);
        auto *remote_md = static_cast<nixlLibfabricPublicMetadata *>(remote[desc_idx].metadataP);
        if (!local_md || !remote_md || !remote_md->conn_) {
            NIXL_ERROR << "Invalid metadata pointers for descriptor " << desc_idx;
            return NIXL_ERR_INVALID_PARAM;
        }

        // Validate connection for this descriptor
        if (remote_md->conn_ != conn) {
            NIXL_ERROR << "Connection mismatch for descriptor " << desc_idx;
            return NIXL_ERR_MISMATCH;
        }
    }

    /*
     * Where the posting thread binds accelerator contexts itself (see postXferDescriptors), check
     * each device binds before anything is posted, for the same reason as the checks above: failing
     * mid-loop would leave earlier descriptors queued with FI_MORE and no flushing post, and requests
     * already submitted against a handle the caller then gives up on.
     *
     * Checked once per device per process, not per transfer -- probing every time doubled the
     * driver calls of every post. Done here on the calling thread even when the post thread pool
     * then posts: what can fail is the device's context, which is process-wide (a primary context),
     * not something one thread can bind and another cannot. What is left for the in-loop check is a
     * transient failure, the same exposure as any other mid-loop error -- and as main, which also
     * returned mid-loop when cudaSetDevice failed.
     */
    if (local.getType() == VRAM_SEG && !progress_thread_enabled_) {
        const struct nfi_hmem_info *last = nullptr;
        for (int desc_idx = 0; desc_idx < desc_count; ++desc_idx) {
            const auto &info =
                static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP)->hmem_info_;
            if (last && last->iface == info.iface && last->device == info.device) {
                continue;
            }
            last = &info;
            if (!nfi_hmem_needs_scope(info.iface) || accelBindKnownOk(info)) {
                continue;
            }
            const nixlAccelScope probe(info);
            if (!probe.ok()) {
                NIXL_ERROR << "Failed to bind accelerator device " << info.device << " (iface "
                           << nfi_hmem_iface_name(info.iface) << ") for descriptor " << desc_idx
                           << "; nothing posted";
                return NIXL_ERR_BACKEND;
            }
            accelBindRecordOk(info);
        }
    }

    // Reserve base_offset once per transfer so all descriptors see a stable rail assignment.
    const size_t xfer_base_offset = rail_manager_.reserveBaseOffset();
    const bool use_post_pool = post_thread_pool_ && post_thread_count_ > 0 && desc_count > 0 &&
        static_cast<size_t>(desc_count) >= post_split_batch_size_;

    if (!use_post_pool) {
        nixl_status_t status = postXferDescriptors(op_type,
                                                   local,
                                                   remote,
                                                   conn,
                                                   backend_handle,
                                                   0,
                                                   desc_count,
                                                   xfer_base_offset,
                                                   /*allow_fi_more=*/true,
                                                   total_submitted);
        if (status != NIXL_SUCCESS) {
            return status;
        }
    } else {
        const size_t num_chunks = std::min(post_thread_count_, static_cast<size_t>(desc_count));
        const size_t chunk_size = (desc_count + num_chunks - 1) / num_chunks;
        std::atomic<size_t> parallel_total_submitted{0};
        std::atomic<int> first_status{static_cast<int>(NIXL_SUCCESS)};
        std::mutex done_mutex;
        std::condition_variable done_cv;
        size_t remaining = num_chunks;
        size_t submitted_chunks = 0;

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const int start_idx = static_cast<int>(chunk_idx * chunk_size);
            const int end_idx = static_cast<int>(
                std::min(static_cast<size_t>(desc_count), (chunk_idx + 1) * chunk_size));

            if (!post_thread_pool_->submit([&, start_idx, end_idx]() {
                    nixl_status_t status = NIXL_SUCCESS;
                    size_t chunk_submitted = 0;

                    try {
                        status = postXferDescriptors(op_type,
                                                     local,
                                                     remote,
                                                     conn,
                                                     backend_handle,
                                                     start_idx,
                                                     end_idx,
                                                     xfer_base_offset,
                                                     /*allow_fi_more=*/false,
                                                     chunk_submitted);
                    }
                    catch (const std::exception &e) {
                        NIXL_ERROR << "Exception while posting libfabric descriptors [" << start_idx
                                   << ", " << end_idx << "): " << e.what();
                        status = NIXL_ERR_BACKEND;
                    }
                    catch (...) {
                        NIXL_ERROR << "Unknown exception while posting libfabric descriptors ["
                                   << start_idx << ", " << end_idx << ")";
                        status = NIXL_ERR_BACKEND;
                    }

                    if (status != NIXL_SUCCESS) {
                        storeFirstError(first_status, status);
                    } else {
                        parallel_total_submitted.fetch_add(chunk_submitted);
                    }

                    {
                        const std::lock_guard<std::mutex> lock(done_mutex);
                        remaining--;
                    }
                    done_cv.notify_one();
                })) {
                NIXL_ERROR << "Failed to submit libfabric descriptor post task";
                break;
            }
            submitted_chunks++;
        }

        if (submitted_chunks != num_chunks) {
            std::unique_lock<std::mutex> lock(done_mutex);
            done_cv.wait(lock, [&remaining, num_chunks, submitted_chunks]() {
                return remaining == num_chunks - submitted_chunks;
            });
            return NIXL_ERR_BACKEND;
        }

        {
            std::unique_lock<std::mutex> lock(done_mutex);
            done_cv.wait(lock, [&remaining]() { return remaining == 0; });
        }

        nixl_status_t status = static_cast<nixl_status_t>(first_status.load());
        if (status != NIXL_SUCCESS) {
            return status;
        }

        total_submitted = parallel_total_submitted.load();
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

    // Progress rails to kick off transfers
    if (!progress_thread_enabled_) {
        nixl_status_t progress_status = rail_manager_.progressActiveRails();
        if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress rails in postXfer";
            return progress_status;
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
        nixl_status_t progress_status = rail_manager_.progressActiveRails();
        if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress rails in checkXfer";
            return progress_status;
        }
    }

    // Then check for completions after processing any pending completions
    if (backend_handle->is_completed()) {
        // Check if any request completed with error
        nixl_status_t err = backend_handle->get_error_status();
        if (err != NIXL_SUCCESS) {
            NIXL_ERROR << "Transfer completed with CQ error status " << err;
            notifXferFailure(backend_handle, err);
            return err;
        }

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
    if (!handle) {
        return NIXL_SUCCESS;
    }

    delete static_cast<nixlLibfabricBackendH *>(handle);
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
    std::shared_ptr<nixlLibfabricConnection> connection;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto it = connections_.find(remote_agent);
        if (it == connections_.end()) {
            NIXL_ERROR << "No connection found for agent: " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        connection = it->second;
    }

    if (connection->overall_state_.load(std::memory_order_acquire) ==
        ConnectionState::DISCONNECTED) {
        nixl_status_t status = establishConnection(remote_agent);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "notifSendPriv: failed to establish connection with " << remote_agent;
            return status;
        }
    }

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

        // Send from the rail paired with the peer's rail 0; see nixlLibfabricConnection::control_rail_.
        size_t rail_id = connection->control_rail_;
        size_t max_size = BinaryNotification::MAX_FRAGMENT_SIZE;
        nixlLibfabricReq *control_request =
            rail_manager_.getRail(rail_id).allocateControlRequest(max_size, notif_xfer_id);

        if (!control_request) {
            NIXL_ERROR << "Failed to allocate control request for notification fragment " << seq_id;
            return NIXL_ERR_BACKEND;
        }

        size_t serialized_size = binary_notification.serialize(control_request->buffer);
        control_request->buffer_size = serialized_size;

        NIXL_DEBUG << "Sending binary notification fragment " << seq_id << "/"
                   << binary_notifications.size() << " size=" << serialized_size << "B"
                   << " payload_chunk_size=" << header.payload_length << "B"
                   << " notif_xfer_id=" << header.notif_xfer_id;

        nixl_status_t status = rail_manager_.postControlMessage(
            nixlLibfabricRailManager::ControlMessageType::NOTIFICATION,
            control_request,
            connection->rail_remote_addr_list_[rail_id][0]);

        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "postControlMessage failed on rail " << rail_id << " for fragment "
                       << seq_id;
            return NIXL_ERR_BACKEND;
        }

        // Progress rail 0 to ensure the message is sent
        if (!progress_thread_enabled_) {
            status = rail_manager_.getRail(rail_id).progressCompletionQueue();
            if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
                NIXL_ERROR << "Failed to progress rail 0 in notifSendPriv";
                return status;
            }
        }
    }

    NIXL_DEBUG << "Successfully sent all " << binary_notifications.size()
               << " notification fragments" << " total_length=" << total_message_length;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::notifXferErrorPriv(const std::string &remote_agent,
                                        uint16_t notif_xfer_id,
                                        uint32_t final_completions) const {
    std::shared_ptr<nixlLibfabricConnection> connection;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto it = connections_.find(remote_agent);
        if (it == connections_.end()) {
            NIXL_ERROR << "No connection found for agent: " << remote_agent
                       << ", cannot report transfer error for XFER_ID=" << notif_xfer_id;
            return NIXL_ERR_NOT_FOUND;
        }
        connection = it->second;
    }

    const size_t rail_id = 0; // Control messages always travel on rail 0
    nixlLibfabricReq *control_request = rail_manager_.getRail(rail_id).allocateControlRequest(
        sizeof(XferErrorPayload), notif_xfer_id);
    if (!control_request) {
        NIXL_ERROR << "Failed to allocate control request for transfer error, XFER_ID="
                   << notif_xfer_id;
        return NIXL_ERR_BACKEND;
    }

    XferErrorPayload payload{};
    payload.final_completions = final_completions;
    memcpy(control_request->buffer, &payload, sizeof(payload));
    control_request->buffer_size = sizeof(payload);

    // A successful postControlMessage() only means fi_senddata() accepted the message, so watch
    // the send completion as well: if it lands in the CQ as an error the target is left waiting
    // and nothing else reports it. This runs later on the progress thread, by which time the
    // caller's backend handle may already have been released, so capture values only and never
    // touch the handle.
    auto send_completion = [agent = remote_agent,
                            xfer_id = notif_xfer_id](nixl_status_t send_status) {
        if (send_status == NIXL_SUCCESS) {
            return;
        }
        NIXL_ERROR << "Transfer-error message to " << agent << " for XFER_ID=" << xfer_id
                   << " failed in its send CQ with status " << send_status << "; that agent will "
                   << "keep waiting for writes that never arrive and will not see this transfer's "
                   << "notification";
    };

    nixl_status_t status =
        rail_manager_.postControlMessage(nixlLibfabricRailManager::ControlMessageType::XFER_ERROR,
                                         control_request,
                                         connection->rail_remote_addr_list_[rail_id][0],
                                         std::move(send_completion));
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to send transfer-error message for XFER_ID=" << notif_xfer_id;
        return NIXL_ERR_BACKEND;
    }

    // Without a progress thread, progress rail to ensure the message is sent
    if (!progress_thread_enabled_) {
        status = rail_manager_.getRail(rail_id).progressCompletionQueue();
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress rail 0 in notifXferErrorPriv";
            return status;
        }
    }

    NIXL_DEBUG << "Sent transfer-error message to " << remote_agent << " XFER_ID=" << notif_xfer_id
               << " final_completions=" << final_completions;
    return NIXL_SUCCESS;
}

void
nixlLibfabricEngine::notifXferFailure(nixlLibfabricBackendH *backend_handle,
                                      nixl_status_t error_status) const {
    if (!backend_handle->has_notif || backend_handle->operation_ != nixl_xfer_op_t::NIXL_WRITE ||
        backend_handle->remote_agent_ == localAgent) {
        return;
    }

    if (!backend_handle->claim_xfer_error_send()) {
        return;
    }

    const uint32_t final_completions =
        static_cast<uint32_t>(backend_handle->get_successful_requests_count());

    NIXL_ERROR << "Transfer XFER_ID=" << backend_handle->post_xfer_id << " to "
               << backend_handle->remote_agent_ << " failed with status " << error_status
               << " after " << final_completions << " of "
               << backend_handle->get_submitted_requests_count()
               << " writes completed; notifying target so it does not wait for the rest";

    nixl_status_t status = notifXferErrorPriv(
        backend_handle->remote_agent_, backend_handle->post_xfer_id, final_completions);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Could not notify " << backend_handle->remote_agent_
                   << " of the transfer error for XFER_ID=" << backend_handle->post_xfer_id
                   << " (status " << status
                   << "); that agent will keep waiting for writes that never arrive";
    }
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
        nixl_status_t progress_status = rail_manager_.progressActiveRails();
        if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to progress rails in getNotifs";
            return progress_status;
        }
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

namespace {
// The engine whose progress thread this is, or null on any other thread. Per engine rather than a
// flag, so one engine's progress thread is never mistaken for another's.
thread_local const nixlLibfabricEngine *tls_progress_thread_engine = nullptr;
} // namespace

bool
nixlLibfabricEngine::onProgressThread() const {
    return tls_progress_thread_engine == this;
}

// Progress thread that continuously processes completions only on rails
nixl_status_t
nixlLibfabricEngine::progressThread() {
    tls_progress_thread_engine = this;
    NIXL_DEBUG << "PT: Thread started successfully for rails only";
    // Main progress loop - continuously process completions only on rails
    while (!progress_thread_stop_.load()) {
        // Process completions only on rails (non-blocking)
        bool any_completions = false;
        nixl_status_t status = rail_manager_.progressActiveRails();
        if (status == NIXL_SUCCESS) {
            any_completions = true;
            NIXL_DEBUG << "PT: Processed completions on rails";
        } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
            NIXL_ERROR << "PT: Failed to process completions on rails";
            // Don't return error, continue for robustness
        }
        // Handshakes requested from inside the CQ callbacks above, sent now that none is running
        flushDeferredHandshakes();
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
nixlLibfabricEngine::processNotification(const std::string &serialized_notif,
                                         uint32_t sender_peer_idx) {
    NIXL_DEBUG << "Received notification size=" << serialized_notif.size()
               << " sender_peer_idx=" << sender_peer_idx;

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

        const uint64_t key = makePendingKey(sender_peer_idx, notif_xfer_id);
        auto [it, inserted] = pending_notifications_.try_emplace(key, notif_xfer_id);

        if (inserted) {
            NIXL_DEBUG << "Created pending notification"
                       << " sender_peer_idx=" << sender_peer_idx
                       << " notif_xfer_id=" << notif_xfer_id
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
            NIXL_WARN << "Duplicate fragment received: sender_peer_idx=" << sender_peer_idx
                      << " notif_xfer_id=" << notif_xfer_id << " notif_seq_id=" << notif_seq_id;
            return;
        }

        // Store payload chunk (combined agent_name + message chunk)
        it->second.message_fragments[notif_seq_id] = payload_chunk;
        it->second.received_msg_fragments++;

        // Update metadata from fragment 0 (agent_name will be extracted after reassembly)
        if (notif_seq_id == 0) {
            // A transfer-error message may have arrived first and lowered expected_completions to
            // the number of writes that actually went out. That count is authoritative, so never
            // raise it back to the count the initiator optimistically sent at postXfer time.
            if (!it->second.xfer_failed) {
                it->second.expected_completions = expected_completions;
            }
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

void
nixlLibfabricEngine::handleXferError(uint16_t notif_xfer_id,
                                     uint32_t sender_peer_idx,
                                     uint32_t final_completions) {
    {
        std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);

        const uint64_t key = makePendingKey(sender_peer_idx, notif_xfer_id);
        auto [it, inserted] = pending_notifications_.try_emplace(key, notif_xfer_id);

        if (inserted) {
            it->second.remote_agent = "";
            it->second.received_completions = 0;
            it->second.expected_msg_fragments = 1; // Default to 1 fragment
            it->second.received_msg_fragments = 0;
        }

        it->second.xfer_failed = true;
        it->second.expected_completions = final_completions;

        NIXL_ERROR << "Initiator reported a failed transfer: sender_peer_idx=" << sender_peer_idx
                   << " notif_xfer_id=" << notif_xfer_id
                   << " received_completions=" << it->second.received_completions
                   << " expected_completions=" << it->second.expected_completions
                   << (inserted ? " (error arrived before the notification)" : "")
                   << "; data for this transfer is incomplete";
    }

    // Check if any notifications can now be completed
    checkPendingNotifications();
}

/****************************************
 * Receiver Side XFER_ID Tracking Helper Methods
 *****************************************/

void
nixlLibfabricEngine::addReceivedXferId(uint16_t xfer_id, uint32_t sender_peer_idx) {
    {
        std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);

        const uint64_t key = makePendingKey(sender_peer_idx, xfer_id);
        auto [it, inserted] = pending_notifications_.try_emplace(key, xfer_id);

        if (inserted) {
            // Set placeholder values for write-arrived-first case
            it->second.remote_agent = "";
            it->second.expected_completions = INT_MAX;
            it->second.received_completions = 0;
            it->second.expected_msg_fragments = 1; // Default to 1 fragment
            it->second.received_msg_fragments = 0;
            NIXL_DEBUG << "Created placeholder notification for sender_peer_idx=" << sender_peer_idx
                       << " notif_xfer_id=" << xfer_id << " (write arrived first)";
        }

        it->second.received_completions++;
        NIXL_DEBUG << "Incremented received count for sender_peer_idx=" << sender_peer_idx
                   << " notif_xfer_id=" << xfer_id << ": " << it->second.received_completions << "/"
                   << it->second.expected_completions;
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

            if (it->second.xfer_failed) {
                NIXL_WARN << "Releasing the notification for notif_xfer_id="
                          << it->second.notif_xfer_id
                          << " whose transfer failed on the initiator; only "
                          << it->second.received_completions
                          << " writes arrived, so the data for this transfer is incomplete";
            }

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
    post_thread_pool_.reset();

    NIXL_DEBUG << "Cleanup all resources complete";
}
