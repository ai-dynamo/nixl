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
#include "libfabric_connection.h"
#include "common/nixl_log.h"

nixlLibfabricConnection::nixlLibfabricConnection(const std::string &remote_agent,
                                                 size_t agent_index)
    : agent_index_(agent_index),
      remoteAgent_(remote_agent),
      overall_state_(ConnectionState::DISCONNECTED) {}

nixl_status_t
nixlLibfabricConnection::establish() {
    if (overall_state_.load(std::memory_order_acquire) == ConnectionState::CONNECTED) {
        NIXL_DEBUG << "Connection already established for " << remoteAgent_;
        return NIXL_SUCCESS;
    }

    NIXL_DEBUG << "Establishing rail connections for agent: " << remoteAgent_;
    NIXL_DEBUG << "Using connection info with " << src_ep_names_.size() << " rails";
    for (size_t i = 0; i < src_ep_names_.size(); ++i) {
        NIXL_DEBUG << "Rail " << i << ": "
                   << LibfabricUtils::hexdump(src_ep_names_[i], LF_EP_NAME_MAX_LEN);
    }
    NIXL_DEBUG << "Agent index: " << agent_index_;

    overall_state_.store(ConnectionState::CONNECTED, std::memory_order_release);
    NIXL_INFO << "Connection state for agent " << remoteAgent_ << " is now CONNECTED";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricConnection::disconnect() {
    if (overall_state_.load(std::memory_order_acquire) == ConnectionState::DISCONNECTED) {
        NIXL_DEBUG << "Connection already disconnected for " << remoteAgent_;
        return NIXL_SUCCESS;
    }

    // TODO: Implement disconnect logic to cleanup the AV Address Entries from both local and remote
    // AV.
    overall_state_.store(ConnectionState::DISCONNECTED, std::memory_order_release);
    NIXL_INFO << "Disconnected from agent: " << remoteAgent_;
    return NIXL_SUCCESS;
}
