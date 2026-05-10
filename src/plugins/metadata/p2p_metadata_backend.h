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
 * @file p2p_metadata_backend.h
 * @brief Direct (TCP socket) metadata backend.
 */
#ifndef NIXL_SRC_PLUGINS_METADATA_P2P_METADATA_BACKEND_H
#define NIXL_SRC_PLUGINS_METADATA_P2P_METADATA_BACKEND_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "metadata/metadata_backend.h"
#include "nixl_types.h"
#include "stream/metadata_stream.h"

class nixlAgent;

namespace nixl::md {

using socket_peer_t = std::pair<std::string, int>;
using socket_map_t = std::map<socket_peer_t, int>;

} // namespace nixl::md

/**
 * @class nixlP2PMetadataBackend
 * @brief P2P (TCP socket) implementation of `nixlMetadataBackend`.
 *
 * Supports the backend key/value contract and the current single-peer
 * NIXLCOMM request flow. The manager drives inbound socket processing by
 * calling `processOnce` from its tick loop.
 */
class nixlP2PMetadataBackend final : public nixlMetadataBackend {
public:
    /**
     * @param listen_port  Port to listen on; passing 0 selects
     *                     `default_comm_port`.
     */
    nixlP2PMetadataBackend(int listen_port, std::string my_agent_name);
    ~nixlP2PMetadataBackend() override;

    nixlP2PMetadataBackend(nixlP2PMetadataBackend &&) = delete;
    nixlP2PMetadataBackend(const nixlP2PMetadataBackend &) = delete;
    nixlP2PMetadataBackend &
    operator=(nixlP2PMetadataBackend &&) = delete;
    nixlP2PMetadataBackend &
    operator=(const nixlP2PMetadataBackend &) = delete;

    /*** nixlMetadataBackend ***/

    /** @brief Publish @p value to the peer encoded by @p key (`"<ip>:<port>"`). */
    [[nodiscard]] nixl_status_t
    publish(const std::string &key, const nixl_blob_t &value) override;

    /** @brief P2P backend has no synchronous fetch; returns NOT_SUPPORTED. */
    [[nodiscard]] nixl_status_t
    fetch(const std::string &key, nixl_blob_t &value) override;

    /**
     * @brief Send an invalidation request to the peer encoded by @p key
     *        (`"<ip>:<port>"`).
     */
    [[nodiscard]] nixl_status_t
    remove(const std::string &key) override;

    [[nodiscard]] nixl_status_t
    watch(const std::string &prefix, nixl_md_watch_cb_t cb) override;

    [[nodiscard]] bool
    isHealthy() const noexcept override;

    [[nodiscard]] nixl_status_t
    fetchBatch(const std::vector<std::string> &keys,
               std::vector<nixl_blob_t> &out,
               std::vector<nixl_status_t> &per_key_status) override;

    /*** Single-peer request helpers (called from the manager tick) ***/

    /** @brief Send my metadata blob to (ip, port) as NIXLCOMM:LOAD. */
    [[nodiscard]] nixl_status_t
    sendToPeer(const std::string &ip, int port, const nixl_blob_t &blob);

    /** @brief Ask peer at (ip, port) for its metadata via NIXLCOMM:SEND. */
    [[nodiscard]] nixl_status_t
    requestMetadataFromPeer(const std::string &ip, int port);

    /** @brief Tell peer at (ip, port) to invalidate via NIXLCOMM:INVL. */
    [[nodiscard]] nixl_status_t
    invalidatePeerMetadata(const std::string &ip, int port, const std::string &my_agent_name);

    /**
     * @brief Per-tick driver work: accept new connections, then drain
     *        already-connected peers' inboxes and dispatch to @p agent.
     */
    void
    processOnce(nixlAgent &agent);

    /** @brief Move open sockets out of the backend (for owner-driven cleanup). */
    nixl::md::socket_map_t
    detachSockets();

    /** @brief Set up the listener; throws on bind/listen failure. */
    void
    setupListener();

private:
    [[nodiscard]] nixl_status_t
    ensureSocket(const std::string &ip, int port);
    [[nodiscard]] int
    socketFor(const std::string &ip, int port) const;
    void
    forgetSocket(const std::string &ip, int port);
    void
    forgetSocket(nixl::md::socket_map_t::iterator it);

    const int listen_port_;
    const std::string my_agent_name_;
    std::unique_ptr<nixlMDStreamListener> listener_;
    nixl::md::socket_map_t sockets_;
};

#endif // NIXL_SRC_PLUGINS_METADATA_P2P_METADATA_BACKEND_H
