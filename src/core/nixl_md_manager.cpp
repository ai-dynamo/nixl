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
#include "nixl_md_manager.h"

#include "nixl.h"
#include "common/nixl_log.h"

#include <arpa/inet.h>
#include <utility>

namespace {

// Build the nixl_opt_args_t for a peer. Takes ip by value and moves it in to
// avoid a redundant copy. Kept .cpp-local rather than passing the manager's
// private Peer struct, which would pull this helper into the installed header.
nixl_opt_args_t
makePeerArgs(std::string ip, std::uint16_t port, const nixl_opt_args_t *base = nullptr) {
    nixl_opt_args_t args = base ? *base : nixl_opt_args_t{};
    args.ipAddr = std::move(ip);
    args.port = port;
    return args;
}

} // namespace

bool
nixlMDManager::lookupPeer(const std::string &agent_name, Peer &out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = peers_.find(agent_name);
    if (it == peers_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

nixl_status_t
nixlMDManager::registerMDPeer(const std::string &agent_name,
                              const std::string &ip,
                              std::uint16_t port) {
    if (agent_name.empty() || ip.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // The metadata listener path is IPv4-only (inet_pton(AF_INET, ...) in
    // nixl_listener.cpp), so reject malformed addresses up front.
    in_addr parsed;
    if (inet_pton(AF_INET, ip.c_str(), &parsed) != 1) {
        NIXL_DEBUG << "[" << self_name_ << "] registerMDPeer: invalid IPv4 address '" << ip
                   << "' for " << agent_name;
        return NIXL_ERR_INVALID_PARAM;
    }
    // 0 means "use the listener-side default", matching the existing
    // nixl_opt_args_t::port default.
    const std::uint16_t resolved_port = (port == 0) ? default_comm_port : port;
    std::lock_guard<std::mutex> lk(mutex_);
    // Same address is idempotent; rebinding a name is rejected (unregister first).
    const auto it = peers_.find(agent_name);
    if (it != peers_.end()) {
        if (it->second.ip == ip && it->second.port == resolved_port) {
            NIXL_DEBUG << "[" << self_name_ << "] registerMDPeer: " << agent_name
                       << " already bound to " << ip << ":" << resolved_port << " (idempotent)";
            return NIXL_SUCCESS;
        }
        NIXL_DEBUG << "[" << self_name_ << "] registerMDPeer: rejecting rebind of " << agent_name
                   << " from " << it->second.ip << ":" << it->second.port << " to " << ip << ":"
                   << resolved_port;
        return NIXL_ERR_NOT_ALLOWED;
    }
    peers_[agent_name] = Peer{ip, resolved_port};
    NIXL_DEBUG << "[" << self_name_ << "] registerMDPeer: " << agent_name << " -> " << ip << ":"
               << resolved_port;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::unregisterMDPeer(const std::string &agent_name) {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        NIXL_DEBUG << "[" << self_name_ << "] unregisterMDPeer: " << agent_name
                   << " not registered (no-op)";
        return NIXL_SUCCESS;
    }
    // Tell the remote to drop our metadata before removing the local entry.
    // Keep the entry if the call is rejected so the caller can retry.
    // peer.ip is reused below for the compare-then-erase, so it is copied
    // (not moved) into the args here.
    const auto args = makePeerArgs(peer.ip, peer.port);
    const nixl_status_t s = agent_.invalidateLocalMD(&args);
    if (s != NIXL_SUCCESS) {
        NIXL_DEBUG << "[" << self_name_ << "] unregisterMDPeer: invalidate for " << agent_name
                   << " failed: " << s;
        return s;
    }
    // Compare-then-erase: guard against a concurrent registerMDPeer
    // having replaced the entry while invalidate was in flight.
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = peers_.find(agent_name);
    if (it != peers_.end() && it->second.ip == peer.ip && it->second.port == peer.port) {
        peers_.erase(it);
    }
    NIXL_DEBUG << "[" << self_name_ << "] unregisterMDPeer: " << agent_name << " removed";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::sendLocalMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    // lookupPeer copies {ip, port}, so the lock is released before the call
    // below. A concurrent unregister/re-register can change the registry in
    // this gap; we intentionally use the snapshot rather than hold the lock
    // across a network call. Worst case the send targets the just-removed
    // address, which is acceptable.
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << self_name_ << "] sendLocalMD: " << agent_name;
    return agent_.sendLocalMD(&args);
}

nixl_status_t
nixlMDManager::sendLocalPartialMD(const std::string &agent_name,
                                  const nixl_reg_dlist_t &descs,
                                  const nixl_opt_args_t *md_extra_params) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port, md_extra_params);
    NIXL_DEBUG << "[" << self_name_ << "] sendLocalPartialMD: " << agent_name;
    return agent_.sendLocalPartialMD(descs, &args);
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << self_name_ << "] fetchRemoteMD: " << agent_name;
    return agent_.fetchRemoteMD(agent_name, &args);
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << self_name_ << "] invalidateLocalMD: " << agent_name;
    return agent_.invalidateLocalMD(&args);
}

nixl_status_t
nixlMDManager::checkRemoteMD(const std::string &agent_name, const nixl_xfer_dlist_t &descs) const {
    // TRACE (not DEBUG): callers typically poll this in a loop.
    NIXL_TRACE << "[" << self_name_ << "] checkRemoteMD: " << agent_name;
    return agent_.checkRemoteMD(agent_name, descs);
}
