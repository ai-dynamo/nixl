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

namespace {

nixl_opt_args_t
makePeerArgs(const std::string &ip, std::uint16_t port, const nixl_opt_args_t *base = nullptr) {
    nixl_opt_args_t args = base ? *base : nixl_opt_args_t{};
    args.ipAddr = ip;
    args.port = static_cast<int>(port);
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
    // 0 means "use the listener-side default", matching the existing
    // nixl_opt_args_t::port default.
    const std::uint16_t resolved_port =
        (port == 0) ? static_cast<std::uint16_t>(default_comm_port) : port;
    std::lock_guard<std::mutex> lk(mutex_);
    // Same address is idempotent; rebinding a name is rejected (unregister first).
    const auto it = peers_.find(agent_name);
    if (it != peers_.end()) {
        if (it->second.ip == ip && it->second.port == resolved_port) {
            return NIXL_SUCCESS;
        }
        return NIXL_ERR_NOT_ALLOWED;
    }
    peers_[agent_name] = Peer{ip, resolved_port};
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::unregisterMDPeer(const std::string &agent_name) {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_SUCCESS;
    }
    // Tell the remote to drop our metadata before removing the local entry.
    // Keep the entry if the call is rejected so the caller can retry.
    const auto args = makePeerArgs(peer.ip, peer.port);
    const nixl_status_t s = agent_.invalidateLocalMD(&args);
    if (s != NIXL_SUCCESS) {
        return s;
    }
    // Compare-then-erase: guard against a concurrent registerMDPeer
    // having replaced the entry while invalidate was in flight.
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = peers_.find(agent_name);
    if (it != peers_.end() && it->second.ip == peer.ip && it->second.port == peer.port) {
        peers_.erase(it);
    }
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
    const auto args = makePeerArgs(peer.ip, peer.port);
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
    const auto args = makePeerArgs(peer.ip, peer.port, md_extra_params);
    return agent_.sendLocalPartialMD(descs, &args);
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    const auto args = makePeerArgs(peer.ip, peer.port);
    return agent_.fetchRemoteMD(agent_name, &args);
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    const auto args = makePeerArgs(peer.ip, peer.port);
    return agent_.invalidateLocalMD(&args);
}

nixl_status_t
nixlMDManager::checkRemoteMD(const std::string &agent_name, const nixl_xfer_dlist_t &descs) const {
    return agent_.checkRemoteMD(agent_name, descs);
}
