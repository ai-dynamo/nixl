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
 * @file nixl_md_manager.h
 * @brief Opt-in, name-keyed wrapper over nixlAgent's metadata APIs.
 */
#ifndef NIXL_SRC_API_CPP_NIXL_MD_MANAGER_H
#define NIXL_SRC_API_CPP_NIXL_MD_MANAGER_H

#include "nixl_descriptors.h"
#include "nixl_types.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

class nixlAgent;
class nixlMetadataBackend;

/**
 * @class nixlMDManager
 * @brief Name-based wrapper over nixlAgent's metadata APIs.
 *
 * The owning nixlAgent constructs the manager when NIXL_MD_MANAGER is set
 * and hands it out via nixlAgent::getMDManager(). The transport is chosen at
 * construction: ETCD when NIXL_ETCD_ENDPOINTS is set (via a core-internal
 * nixlMetadataBackend), otherwise P2P. The manager stores a per-name registry;
 * P2P resolves each name to {ip, port} and delegates to the matching nixlAgent
 * method, while ETCD keys metadata by name in the store and ignores the
 * address fields.
 */
class nixlMDManager {
public:
    /**
     * @brief Construct the manager and select its backend from the environment.
     *
     * @param agent              Owning agent; delegated to for cache ops.
     * @param self_name          This agent's name, used to build KV keys and
     *                           to prefix trace logs. Passed in (rather than
     *                           read via agent.getName()) because the owning
     *                           agent is still being constructed at this point.
     * @param etcd_watch_timeout Watch timeout forwarded to the ETCD backend.
     *
     * In ETCD mode the backend connects eagerly (health gate); construction
     * throws if the store is unreachable.
     */
    nixlMDManager(nixlAgent &agent,
                  std::string self_name,
                  std::chrono::microseconds etcd_watch_timeout);

    ~nixlMDManager();

    /**
     * @brief Register a P2P peer's reachable address.
     *
     * Re-registering the same {ip, port} is a no-op; rebinding a name to a
     * different address is rejected (unregisterMDPeer first).
     *
     * @param agent_name Logical agent name.
     * @param ip         IPv4 dotted-decimal address (validated with
     *                   inet_pton; the listener path is currently IPv4-only).
     * @param port       TCP port the peer listens on. `0` is treated as
     *                   `default_comm_port`.
     * @return nixl_status_t NIXL_SUCCESS (new or identical registration);
     *                   NIXL_ERR_INVALID_PARAM if `agent_name` is empty or
     *                   `ip` is not a valid IPv4 address; NIXL_ERR_NOT_ALLOWED
     *                   if `agent_name` is already registered to a different
     *                   address.
     */
    [[nodiscard]] nixl_status_t
    registerMDPeer(const std::string &agent_name, const std::string &ip, std::uint16_t port);

    /**
     * @brief Drop a previously-registered peer and invalidate the local
     *        metadata we last sent to that peer.
     *
     * P2P: calls nixlAgent::invalidateLocalMD against the peer's {ip, port},
     * erasing the entry only on success (so the caller can retry). ETCD: just
     * forgets the peer locally. Unregistering an unknown peer is a no-op.
     *
     * @param agent_name Peer to unregister.
     * @return nixl_status_t NIXL_SUCCESS on success or unknown peer;
     *                   otherwise the status returned by the invalidate call.
     */
    [[nodiscard]] nixl_status_t
    unregisterMDPeer(const std::string &agent_name);

    /**
     * @brief Send the full local metadata blob to `agent_name`.
     *
     * @param agent_name Previously-registered peer.
     * @return nixl_status_t NIXL_SUCCESS, or NIXL_ERR_NOT_FOUND if the peer
     *                   is not registered.
     */
    [[nodiscard]] nixl_status_t
    sendLocalMD(const std::string &agent_name) const;

    /**
     * @brief Send a partial local metadata blob to `agent_name`.
     *
     * @param agent_name      Previously-registered peer.
     * @param descs           Descriptor list to include in the metadata.
     * @param md_extra_params Optional operational args forwarded to
     *                        nixlAgent::sendLocalPartialMD (e.g.
     *                        `backends`, `includeConnInfo`,
     *                        `customParam`). The manager overrides
     *                        `ipAddr`/`port` from the registry; any
     *                        caller-supplied values for those fields are
     *                        ignored. `metadataLabel` is required when the
     *                        ETCD backend is active (it is used as the key
     *                        label; an empty label returns
     *                        NIXL_ERR_INVALID_PARAM) and is ignored on the
     *                        P2P path.
     * @return nixl_status_t NIXL_SUCCESS, or NIXL_ERR_NOT_FOUND if the peer
     *                        is not registered.
     */
    [[nodiscard]] nixl_status_t
    sendLocalPartialMD(const std::string &agent_name,
                       const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t *md_extra_params = nullptr) const;

    /**
     * @brief Request the remote agent's metadata from `agent_name`.
     *
     * @param agent_name Previously-registered peer.
     * @return nixl_status_t NIXL_SUCCESS, or NIXL_ERR_NOT_FOUND if the peer
     *                   is not registered.
     */
    [[nodiscard]] nixl_status_t
    fetchRemoteMD(const std::string &agent_name) const;

    /**
     * @brief Tell `agent_name` that our metadata is no longer valid.
     *
     * @param agent_name Previously-registered peer.
     * @return nixl_status_t NIXL_SUCCESS, or NIXL_ERR_NOT_FOUND if the peer
     *                   is not registered.
     */
    [[nodiscard]] nixl_status_t
    invalidateLocalMD(const std::string &agent_name) const;

    /**
     * @brief Local readiness check passthrough to nixlAgent::checkRemoteMD.
     *        No registration is required; the agent name is looked up in
     *        nixlAgent's existing remote-sections cache.
     *
     * @param agent_name Remote agent to check for.
     * @param descs      Descriptors to check; pass an empty list to only
     *                   check for presence of the remote's metadata.
     * @return nixl_status_t NIXL_SUCCESS if the metadata is available,
     *                   NIXL_ERR_NOT_FOUND otherwise.
     */
    [[nodiscard]] nixl_status_t
    checkRemoteMD(const std::string &agent_name, const nixl_xfer_dlist_t &descs) const;

    /**
     * @brief Name of the metadata transport in use, selected at construction
     *        (`"ETCD"` when an ETCD backend is active, otherwise `"P2P"`).
     *
     * Returns a name rather than an enum so new backends can be added without
     * editing a closed public type (mirrors nixl_backend_t for transfer
     * backends); each backend reports its own name via nixlMetadataBackend.
     * Defined in the .cpp because it dereferences the core-internal backend,
     * which must not be included from this installed public header.
     *
     * @return std::string_view The active transport name.
     */
    [[nodiscard]] std::string_view
    getBackend() const noexcept;

    nixlMDManager(nixlMDManager &&) = delete;
    nixlMDManager(const nixlMDManager &) = delete;
    nixlMDManager &
    operator=(nixlMDManager &&) = delete;
    nixlMDManager &
    operator=(const nixlMDManager &) = delete;

private:
    // Registry entry for a tracked agent. ip/port are P2P-only addressing,
    // unused by centralized backends. `epoch` (from registerEpoch_) stamps
    // each (re)register so unregister's compare-then-erase won't drop an entry
    // re-registered during an in-flight invalidate; equality includes it.
    struct Peer {
        std::string ip;
        std::uint16_t port;
        std::uint64_t epoch = 0;

        [[nodiscard]] bool
        operator==(const Peer &other) const noexcept {
            return ip == other.ip && port == other.port && epoch == other.epoch;
        }
    };

    [[nodiscard]] bool
    lookupPeer(const std::string &agent_name, Peer &out) const;

    // KV key builder, owned by the manager (legacy {namespace}/{agent}/{label}).
    [[nodiscard]] std::string
    makeKey(const std::string &agent_name, const std::string &label) const;

    // Apply queued remote-invalidation events to the agent's cache. Called
    // opportunistically from fetchRemoteMD/checkRemoteMD; no-op on P2P.
    void
    drainInvalidated() const;

    // ETCD is active when a backend object exists; P2P leaves backend_ null.
    [[nodiscard]] bool
    usingEtcd() const noexcept {
        return backend_ != nullptr;
    }

    nixlAgent &agent_;
    const std::string selfName_;
    std::string namespacePrefix_;

    mutable std::mutex mutex_;
    // Bumped under mutex_ on every successful register; see Peer::epoch.
    std::uint64_t registerEpoch_ = 0;
    std::unordered_map<std::string, Peer> peers_;

    mutable std::mutex invalidatedMutex_;
    mutable std::vector<std::string> invalidatedAgents_;

    // Declared last so it is destroyed first: canceling the backend's watchers
    // before the invalidation queue their callbacks push into is torn down.
    std::unique_ptr<nixlMetadataBackend> backend_;
};

#endif // NIXL_SRC_API_CPP_NIXL_MD_MANAGER_H
