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

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

class nixlAgent;

/**
 * @brief Transport selected by the manager. Currently always P2P;
 *        ETCD is not yet supported.
 */
enum class nixl_md_backend_t {
    P2P,
    ETCD,
};

/**
 * @class nixlMDManager
 * @brief Name-based, P2P-only wrapper over nixlAgent's metadata APIs.
 *
 * The owning nixlAgent constructs the manager when NIXL_MD_MANAGER is set
 * and hands it out via nixlAgent::getMDManager(). The manager stores a
 * per-name {ip, port} registry; each name-keyed call resolves the peer and
 * delegates to the matching nixlAgent method with extra_params.ipAddr /
 * extra_params.port populated. No new threads, no new sockets, no new state
 * in nixlAgent.
 */
class nixlMDManager {
public:
    explicit nixlMDManager(nixlAgent &agent) noexcept : agent_(agent) {}

    /**
     * @brief Register a P2P peer's reachable address.
     *
     * Re-registering the same {ip, port} is a no-op; rebinding a name to a
     * different address is rejected (unregisterMDPeer first).
     *
     * @param agent_name Logical agent name.
     * @param ip         IPv4 dotted-decimal address.
     * @param port       TCP port the peer listens on. `0` is treated as
     *                   `default_comm_port`.
     * @return nixl_status_t NIXL_SUCCESS (new or identical registration);
     *                   NIXL_ERR_INVALID_PARAM if `agent_name` or `ip` is
     *                   empty; NIXL_ERR_NOT_ALLOWED if `agent_name` is
     *                   already registered to a different address.
     */
    [[nodiscard]] nixl_status_t
    registerMDPeer(const std::string &agent_name, const std::string &ip, std::uint16_t port);

    /**
     * @brief Drop a previously-registered peer and invalidate the local
     *        metadata we last sent to that peer.
     *
     * When the peer is known, the manager first calls
     * nixlAgent::invalidateLocalMD against the peer's {ip, port}. The
     * registry entry is erased only when the invalidate call succeeds, so
     * the caller can retry on a transient failure. Unregistering an
     * unknown peer is a no-op and returns `NIXL_SUCCESS`.
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
     *                        ignored. `metadataLabel` is not yet used
     *                        (reserved for future ETCD support).
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
     * @brief Backend transport in use. Currently always returns
     *        `nixl_md_backend_t::P2P`; ETCD is not yet supported.
     *
     * @return nixl_md_backend_t The active transport.
     */
    [[nodiscard]] nixl_md_backend_t
    getBackend() const noexcept {
        return nixl_md_backend_t::P2P;
    }

    nixlMDManager(nixlMDManager &&) = delete;
    nixlMDManager(const nixlMDManager &) = delete;
    nixlMDManager &
    operator=(nixlMDManager &&) = delete;
    nixlMDManager &
    operator=(const nixlMDManager &) = delete;

private:
    struct Peer {
        std::string ip;
        std::uint16_t port;
    };

    [[nodiscard]] bool
    lookupPeer(const std::string &agent_name, Peer &out) const;

    nixlAgent &agent_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Peer> peers_;
};

#endif // NIXL_SRC_API_CPP_NIXL_MD_MANAGER_H
