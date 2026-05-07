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
 * @file nixl_md.h
 * @brief Public types for the NIXL metadata-manager surface.
 *
 * Type vocabulary used by the manager and metadata APIs:
 *   - peer-address pair for P2P registration
 *   - identity mode (UUID-backed vs. agent-name-key compatibility)
 *   - per-call optional args
 *   - callback typedefs
 */
#ifndef NIXL_SRC_API_CPP_NIXL_MD_H
#define NIXL_SRC_API_CPP_NIXL_MD_H

#include <cstdint>
#include <functional>
#include <string>

#include "nixl_types.h"

/**
 * @struct nixl_md_peer_addr_t
 * @brief Reachable address of a peer for P2P metadata exchange.
 *
 * Carries the IP and port the local agent should dial when publishing,
 * fetching, or invalidating against a registered remote agent.
 */
struct nixl_md_peer_addr_t {
    std::string ip;
    std::uint16_t port = 0;
};

/**
 * @enum nixl_md_identity_mode
 * @brief Selects the identity scheme used for ETCD key shape and cache lookup.
 *
 * - UUID_BACKED:            New shape `{ns}/agents/[{label}/]{src_uuid}/{dst|null_agent}`.
 * - LEGACY_AGENT_NAME_KEYS: Agent-name-key compatibility shape `{ns}/{name}/{label}`.
 */
enum class nixl_md_identity_mode {
    UUID_BACKED,
    LEGACY_AGENT_NAME_KEYS,
};

/**
 * @enum nixl_watch_event_t
 * @brief Events delivered by `nixlMetadataBackend::watch` callbacks.
 */
enum class nixl_watch_event_t {
    PUT,
    DELETE,
};

/**
 * @struct nixl_md_opt_args_t
 * @brief Per-call optional args for list-style metadata APIs.
 */
struct nixl_md_opt_args_t {
    /** Optional ETCD label segment appended to the namespace prefix. */
    std::string metadata_label;

    /** Identity scheme to use for key formation and blob parsing. */
    nixl_md_identity_mode identity_mode = nixl_md_identity_mode::UUID_BACKED;
};

/*** Callback typedefs for metadata liveness and invalidation events. ***/

/**
 * @brief Fires when liveness for a remote peer is lost (heartbeat miss
 *        or socket disconnect, depending on backend semantics).
 *
 * @param remote_agent_name  Logical agent name (may be empty if unknown).
 * @param remote_agent_uuid  UUID of the peer that went away.
 */
using nixl_on_liveness_lost_cb_t =
    std::function<void(const std::string &remote_agent_name, const std::string &remote_agent_uuid)>;

/**
 * @brief Fires when a remote agent's metadata is invalidated (explicit
 *        invalidation message or backend signal).
 */
using nixl_on_remote_md_invalidated_cb_t =
    std::function<void(const std::string &remote_agent_name, const std::string &remote_agent_uuid)>;

/**
 * @brief Fires when previously-unknown metadata for a remote peer becomes
 *        available (first PUT for a new (name, uuid) pair).
 */
using nixl_on_new_metadata_available_cb_t =
    std::function<void(const std::string &remote_agent_name, const std::string &remote_agent_uuid)>;

/**
 * @brief Fires when a known agent name reappears with a new UUID,
 *        i.e. the peer was restarted. Old UUID is invalidated; new
 *        UUID's metadata is loaded before this fires.
 */
using nixl_on_replacement_cb_t = std::function<void(const std::string &remote_agent_name,
                                                    const std::string &old_remote_agent_uuid,
                                                    const std::string &new_remote_agent_uuid)>;

#endif // NIXL_SRC_API_CPP_NIXL_MD_H
