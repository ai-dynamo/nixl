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

#include <cstring>
#include <limits>

/****************************************
 * Peer-id handshake protocol
 *****************************************/

// Wire format for a handshake message body (serialized via nixlSerDes):
//   "ver"      : uint16 = NIXL_LIBFABRIC_PROTO_VERSION of the sender
//   "name"     : string = agent_name of the SENDER (us). The receiver uses it to find its
//                 connection record for us, so a handshake needs no source-address lookup.
//   "has_conn" : uint8  = 1 if connection info is included, 0 otherwise
//   "conn"     : string = serialized connection info (only if has_conn == 1).
//                 Included when the peer hasn't sent us a handshake yet,
//                 meaning they likely don't have our endpoint addresses.
//
// Protocol version 1 sent "idx" (the agent index for imm_data) instead of "ver". It is not
// interoperable with version 2, so the version check rejects it at connection setup.
//
// Sized to fit inside one libfabric control buffer
// (NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE = 8 KiB).
nixl_status_t
nixlLibfabricEngine::sendHandshakeTo(const nixlLibfabricConnection &conn) const {
    const uint16_t proto_ver = static_cast<uint16_t>(NIXL_LIBFABRIC_PROTO_VERSION);
    const std::string &my_name = localAgent;

    // Include connection info if peer hasn't sent us a handshake yet (meaning
    // they likely don't have our connection info to create a connection).
    std::string piggybacked_conn_info;
    if (!conn.handshake_received_.load(std::memory_order_acquire)) {
        nixl_status_t ci_status = getConnInfo(piggybacked_conn_info);
        if (ci_status != NIXL_SUCCESS) {
            NIXL_ERROR << "getConnInfo failed in sendHandshakeTo for '" << conn.remoteAgent_
                       << "': status=" << ci_status;
            return ci_status;
        }
    }

    nixlSerDes sd;
    sd.addBuf(NIXL_HANDSHAKE_TAG_VER, &proto_ver, sizeof(proto_ver));
    sd.addStr(NIXL_HANDSHAKE_TAG_NAME, my_name);
    uint8_t has_conn = piggybacked_conn_info.empty() ? 0 : 1;
    sd.addBuf(NIXL_HANDSHAKE_TAG_HAS_CONN, &has_conn, sizeof(has_conn));
    if (has_conn) {
        sd.addStr(NIXL_HANDSHAKE_TAG_CONN, piggybacked_conn_info);
    }
    std::string payload = sd.exportStr();

    if (payload.size() > NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE) {
        NIXL_ERROR << "Handshake payload too large (" << payload.size() << " bytes) — agent_name='"
                   << my_name << "'";
        return NIXL_ERR_BACKEND;
    }

    constexpr size_t kRailId = 0;
    nixlLibfabricReq *req = rail_manager_.getRail(kRailId).allocateControlRequest(
        payload.size(), LibfabricUtils::getNextXferId());
    if (!req) {
        NIXL_ERROR << "Failed to allocate control request for handshake to '" << conn.remoteAgent_
                   << "'";
        return NIXL_ERR_BACKEND;
    }

    std::memcpy(req->buffer, payload.data(), payload.size());
    req->buffer_size = payload.size();

    NIXL_DEBUG << "Sending handshake to '" << conn.remoteAgent_ << "' proto_ver=" << proto_ver
               << " (my agent_name='" << my_name
               << "', piggybacked_conn_info_len=" << piggybacked_conn_info.size() << ")";
    return rail_manager_.postControlMessage(nixlLibfabricRailManager::ControlMessageType::HANDSHAKE,
                                            req,
                                            conn.rail_remote_addr_list_.at(kRailId)[0]);
}

void
nixlLibfabricEngine::handleHandshake(const std::string &raw_payload) {
    // Decode and validate the handshake wire format.
    nixlSerDes sd;
    if (sd.importStr(raw_payload) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize handshake payload";
        return;
    }

    uint16_t peer_proto_ver = 0;
    if (sd.getBuf(NIXL_HANDSHAKE_TAG_VER, &peer_proto_ver, sizeof(peer_proto_ver)) !=
        NIXL_SUCCESS) {
        // A protocol-version-1 peer sends "idx" and no "ver". It stamps an 8-bit agent
        // index into imm_data and expects us to do the same, which we no longer do, so
        // refuse the peer here rather than let it misattribute every transfer we send it.
        NIXL_ERROR << "Handshake has no '" << NIXL_HANDSHAKE_TAG_VER
                   << "' field; the peer speaks libfabric-plugin protocol version 1, which is "
                      "not interoperable with version "
                   << NIXL_LIBFABRIC_PROTO_VERSION
                   << ". Upgrade both ends of the cluster to the same NIXL version.";
        return;
    }
    if (peer_proto_ver != NIXL_LIBFABRIC_PROTO_VERSION) {
        NIXL_ERROR << "Handshake protocol version mismatch: peer speaks " << peer_proto_ver
                   << ", we speak " << NIXL_LIBFABRIC_PROTO_VERSION
                   << ". Upgrade both ends of the cluster to the same NIXL version.";
        return;
    }
    std::string peer_agent_name = sd.getStr(NIXL_HANDSHAKE_TAG_NAME);
    if (peer_agent_name.empty()) {
        NIXL_ERROR << "Handshake missing or empty 'name' field";
        return;
    }
    uint8_t has_conn = 0;
    if (sd.getBuf(NIXL_HANDSHAKE_TAG_HAS_CONN, &has_conn, sizeof(has_conn)) != NIXL_SUCCESS) {
        NIXL_ERROR << "Handshake missing 'has_conn' field";
        return;
    }
    // The handshake may optionally carry the sender's connection info, for peers that
    // haven't had their connection info loaded via loadRemoteConnInfo() yet.
    std::string piggybacked_conn_info;
    if (has_conn) {
        piggybacked_conn_info = sd.getStr(NIXL_HANDSHAKE_TAG_CONN);
        if (piggybacked_conn_info.empty()) {
            NIXL_ERROR << "Handshake has_conn=1 but 'conn' field is empty";
            return;
        }
    }

    NIXL_DEBUG << "Received handshake from peer '" << peer_agent_name
               << "' proto_ver=" << peer_proto_ver
               << " piggybacked_conn_info_len=" << piggybacked_conn_info.size();

    // Process the decoded handshake.
    std::shared_ptr<nixlLibfabricConnection> conn;
    {
        std::lock_guard<std::mutex> lock(connection_state_mutex_);
        auto it = connections_.find(peer_agent_name);
        if (it != connections_.end() && it->second) {
            conn = it->second;
        }
    }

    if (!conn) {
        {
            std::lock_guard<std::mutex> plk(pending_handshake_mutex_);
            pending_inbound_handshakes_.insert(peer_agent_name);
            NIXL_DEBUG << "Buffered handshake from not-yet-known peer '" << peer_agent_name
                       << "'; will apply on createAgentConnection";
        }

        if (piggybacked_conn_info.empty()) {
            // Code should not reach here. When piggybacked_conn_info is empty, the handshake
            // source should have received a handshake message from this side, which means
            // a connection ('conn' above) should have been created.
            NIXL_ERROR << "Connection not found for peer agent: " << peer_agent_name
                       << ", and no piggybacked_conn_info provided in handshake";
            return;
        } else {
            nixl_status_t load_status =
                this->loadRemoteConnInfo(peer_agent_name, piggybacked_conn_info);
            if (load_status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to loadRemoteConnInfo() received with handshake. "
                           << "peer agent: " << peer_agent_name
                           << ", piggybacked_conn_info size: " << piggybacked_conn_info.size();
            }
            return; // loadRemoteConnInfo will take care of draining the
                    // buffered handshake as part of connection creation.
        }
    } else {
        {
            std::lock_guard<std::mutex> hlk(conn->handshake_mutex_);
            conn->handshake_received_.store(true, std::memory_order_release);
        }
        conn->handshake_cv_.notify_all();
        NIXL_INFO << "Handshake stored: peer='" << peer_agent_name
                  << "' — it has our endpoints in its address vectors, so it can attribute "
                     "our transfers and we may start sending";
    }
}
