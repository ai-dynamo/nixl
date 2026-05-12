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
/**
 * @file nixl_metadata_manager.h
 * @brief Internal coordinator for outbound/inbound metadata work.
 *
 * Owns the metadata worker thread, command queue, peer/UUID state, and the
 * metadata backends used by both list-style and single-peer metadata APIs.
 */
#ifndef NIXL_SRC_CORE_NIXL_METADATA_MANAGER_H
#define NIXL_SRC_CORE_NIXL_METADATA_MANAGER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nixl_md.h"
#include "nixl_types.h"

class nixlAgent;
class nixlAgentData;

namespace nixl::metadata {
class nixlMetadataStrategy;

/**
 * @brief Metadata operation requested by the agent.
 */
enum class request_op {
    send,
    fetch,
    invalidate,
};

/**
 * @brief Manager strategy selected by the agent-facing legacy API.
 */
enum class request_backend {
    p2p,
    etcd,
};

/**
 * @brief Backend-specific routing data for a manager request.
 */
struct request_target {
    request_backend backend{};
    std::string peer_ip;
    std::uint16_t peer_port = 0;
    std::string remote_agent;
    std::string metadata_label;
};

/**
 * @brief Single unit of work the agent posts to the manager.
 */
struct request {
    request_op op{};
    request_target target;
    nixl_blob_t blob;
};

/**
 * @class nixlMetadataManager
 * @brief Owns the metadata worker thread and manager strategy instances.
 *
 * Lifecycle:
 *   1. `nixlAgentData` constructs a manager during agent construction.
 *   2. `nixlAgent` constructor calls `start()` once it has a valid `*this`.
 *   3. `nixlAgent` destructor calls `stop()`.
 */
class nixlMetadataManager {
public:
    explicit nixlMetadataManager(nixlAgentData &agent_data);
    ~nixlMetadataManager();

    nixlMetadataManager(nixlMetadataManager &&) = delete;
    nixlMetadataManager &
    operator=(nixlMetadataManager &&) = delete;
    nixlMetadataManager(const nixlMetadataManager &) = delete;
    nixlMetadataManager &
    operator=(const nixlMetadataManager &) = delete;

    /** @brief Start the worker thread; must be called exactly once. */
    void
    start(nixlAgent &agent);

    /** @brief Drain queue, stop the worker thread, propagate exceptions. */
    void
    stop();

    /** @brief Post a unit of work for the worker thread. */
    void
    enqueue(request req);

    /** @brief Record the UUID observed for an Agent name. */
    void
    recordPeerUuid(const std::string &agent_name, const std::string &agent_uuid);

private:
    void
    drainInto(std::vector<request> &out);

    void
    runLoop(nixlAgent &agent);

    void
    runLoopNoexcept(nixlAgent &agent) noexcept;

    [[nodiscard]] bool
    queueEmpty() const;

    nixlAgentData &agent_data_;

    // Manager-internal command queue.
    mutable std::mutex queue_lock_;
    std::vector<request> queue_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> worker_failed_{false};
    std::thread worker_;
    std::exception_ptr worker_exception_;

    // Strategies owned by the manager. P2P is created when the agent has a
    // listen thread (used as both publisher and acceptor); ETCD is created
    // when `NIXL_ETCD_ENDPOINTS` is set.
    std::vector<std::unique_ptr<nixlMetadataStrategy>> strategies_;

    // Manager-owned state used by list operations and watch deliveries.
    mutable std::mutex cache_lock_;
    std::unordered_map<std::string, std::string> name_to_uuid_;
    std::unordered_map<std::string, nixl_md_peer_addr_t> peer_registry_;

    // In-flight operation table. List operations use this to track deadlines
    // and per-peer progress.
    struct in_flight_op {
        std::chrono::steady_clock::time_point deadline;
    };

    std::unordered_map<std::string, in_flight_op> in_flight_;
};

} // namespace nixl::metadata

#endif // NIXL_SRC_CORE_NIXL_METADATA_MANAGER_H
