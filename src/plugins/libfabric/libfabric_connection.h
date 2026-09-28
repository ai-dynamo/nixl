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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "libfabric/libfabric_rail.h"

/** Multi-rail connection metadata for remote agents */
struct nixlLibfabricConnection : public nixlBackendConnMD {
    nixlLibfabricConnection(const std::string &remote_agent, size_t agent_index);

    /** Transition to CONNECTED state. */
    nixl_status_t
    establish();

    /** Transition to DISCONNECTED state. Returns success if already disconnected. */
    nixl_status_t
    disconnect();

    size_t agent_index_; // Unique agent identifier in agent_names vector
    std::string remoteAgent_; // Remote agent name
    std::unordered_map<size_t, std::vector<fi_addr_t>>
        rail_remote_addr_list_; // Rail libfabric addresses. key=rail id.
    std::vector<char *> src_ep_names_; // Rail endpoint names
    std::atomic<ConnectionState> overall_state_; // Current connection state

    // Handshake received state: true once the peer has told us it holds our endpoints in
    // its address vectors, which is what lets it attribute our writes and notifications
    // back to us (FI_SOURCE only reports addresses present in the explicit AV).
    // establishConnection() blocks on handshake_cv_ until this is set, so no transfer can
    // reach a peer that could not identify us.
    std::atomic<bool> handshake_received_{false};
    std::mutex handshake_mutex_;
    std::condition_variable handshake_cv_;
};

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H
