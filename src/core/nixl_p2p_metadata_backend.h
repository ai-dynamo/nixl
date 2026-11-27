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
 * @file nixl_p2p_metadata_backend.h
 * @brief Point-to-point (socket) metadata backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H

#include "nixl_md_config.h"
#include "nixl_metadata_backend.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <asio.hpp>

class nixlMetadataContext;

/**
 * @class nixlP2PMetadataBackend
 * @brief Self-contained socket-based metadata backend.
 *
 * Owns its transport state: the open peer connections and (when the agent
 * enables listening) the accept socket. Outbound ops validate and serialize
 * synchronously, then submit the socket send to the I/O thread; inbound work
 * (accepting peers, reading LOAD/SEND/INVL replies) happens asynchronously
 * on the I/O thread too, so the connection map is only ever touched by one thread.
 * Depends only on nixlMetadataContext, not nixlAgent.
 */
class nixlP2PMetadataBackend : public nixlMetadataBackend {
public:
    nixlP2PMetadataBackend(nixlMetadataContext &ctx,
                           const nixlMDConfig &config,
                           asio::io_context &io_context);
    ~nixlP2PMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override {
        return "P2P";
    }

    [[nodiscard]] nixl_status_t
    sendLocal(const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    sendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    fetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) override;

    [[nodiscard]] nixl_status_t
    invalidateLocal(const nixl_opt_args_t *extra_params) override;

    // The I/O thread always runs.
    [[nodiscard]] bool
    usesThread() const override {
        return true;
    }

    void
    start() override;

    void
    stop() override;

private:
    struct peerAddress {
        std::string ip_;
        int port_;

        bool
        operator==(const peerAddress &other) const {
            return ip_ == other.ip_ && port_ == other.port_;
        }
    };

    struct peerHash {
        std::size_t
        operator()(const peerAddress &address) const {
            return std::hash<std::string>()(address.ip_) ^ std::hash<int>()(address.port_);
        }
    };

    class nixlP2PMetadataPeer;

    using peer_session_map_t =
        std::unordered_map<peerAddress, std::shared_ptr<nixlP2PMetadataPeer>, peerHash>;

    // Bind the acceptor to the given endpoint.
    // Throws std::runtime_error on failure.
    void
    bindListener(const asio::ip::tcp::endpoint &endpoint);

    // Accept new peers.
    void
    acceptPeers();

    // Get or create a session for the given peer address.
    nixlP2PMetadataPeer *
    getOrCreateSession(const peerAddress &address);

    // The metadata context.
    nixlMetadataContext &ctx_;

    // The I/O context.
    asio::io_context &ioContext_;
    // The acceptor for new incoming connections.
    asio::ip::tcp::acceptor acceptor_;
    // The timer for retrying the accept operation.
    asio::steady_timer acceptRetryTimer_;
    // The map of peer addresses to their sessions.
    peer_session_map_t peerSessions_;
};

#endif // NIXL_SRC_CORE_NIXL_P2P_METADATA_BACKEND_H
