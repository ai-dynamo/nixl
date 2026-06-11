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
#include "agent_data.h"
#include "backends/nixl_metadata_backend.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include "backends/tcpstore_metadata_backend.h"

#if HAVE_ETCD
#include "backends/etcd_metadata_backend.h"
#endif

#include <arpa/inet.h>
#include <utility>

namespace {

// Default KV namespace prefix. Mirrors NIXL_ETCD_NAMESPACE_DEFAULT, which is
// only defined when the ETCD backend is compiled in (HAVE_ETCD).
constexpr char kDefaultMdNamespace[] = "/nixl/agents/";

// Build the nixl_opt_args_t for a peer. Takes ip by value and moves it in to
// avoid a redundant copy. Kept .cpp-local rather than passing the manager's
// private Peer struct, which would pull this helper into the installed header.
[[nodiscard]] nixl_opt_args_t
makePeerArgs(std::string ip, std::uint16_t port, const nixl_opt_args_t *base = nullptr) {
    nixl_opt_args_t args = base ? *base : nixl_opt_args_t{};
    args.ipAddr = std::move(ip);
    args.port = port;
    return args;
}

} // namespace

nixlMDManager::nixlMDManager(nixlAgent &agent,
                             std::string self_name,
                             std::chrono::microseconds kv_backend_timeout)
    : agent_(agent),
      selfName_(std::move(self_name)) {
    bool kv_selected = false;
#if HAVE_ETCD
    if (nixl::config::checkExistence("NIXL_ETCD_ENDPOINTS")) {
        if (nixl::config::checkExistence("NIXL_TCPSTORE_ENDPOINTS")) {
            NIXL_DEBUG << "[" << selfName_
                       << "] NIXL_ETCD_ENDPOINTS and NIXL_TCPSTORE_ENDPOINTS both set; using ETCD";
        }
        namespacePrefix_ = nixl::config::getValueDefaulted<std::string>(
            "NIXL_ETCD_NAMESPACE", NIXL_ETCD_NAMESPACE_DEFAULT);
        backend_ =
            std::make_unique<nixlEtcdMetadataBackend>(makeKey(selfName_, ""), kv_backend_timeout);
        kv_selected = true;
    }
#endif
    if (!kv_selected && nixl::config::checkExistence("NIXL_TCPSTORE_ENDPOINTS")) {
        namespacePrefix_ = nixl::config::getValueDefaulted<std::string>("NIXL_TCPSTORE_NAMESPACE",
                                                                        kDefaultMdNamespace);
        backend_ = std::make_unique<nixlTcpStoreMetadataBackend>(makeKey(selfName_, ""),
                                                                 kv_backend_timeout);
    }
}

nixlMDManager::~nixlMDManager() = default;

std::string_view
nixlMDManager::getBackend() const noexcept {
    return backend_ ? backend_->name() : "P2P";
}

std::string
nixlMDManager::makeKey(const std::string &agent_name, const std::string &label) const {
    return namespacePrefix_ + "/" + agent_name + "/" + label;
}

void
nixlMDManager::drainInvalidated() const {
    // Only a watch-capable KV backend enqueues; on P2P (no backend) and
    // watch-less KV like TCPStore the queue is always empty, so skip taking the
    // lock on the (polled) checkRemoteMD hot path.
    if (!backend_ || !backend_->hasWatch()) {
        return;
    }
    std::vector<std::string> tmp;
    {
        const std::lock_guard<std::mutex> lk(invalidatedMutex_);
        tmp.swap(invalidatedAgents_);
    }
    for (const auto &agent : tmp) {
        const nixl_status_t ret = agent_.invalidateRemoteMD(agent);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "[" << selfName_ << "] failed to invalidate remote metadata for " << agent
                       << ": " << ret;
        } else {
            NIXL_DEBUG << "[" << selfName_ << "] invalidated remote metadata for " << agent;
        }
    }
}

bool
nixlMDManager::lookupPeer(const std::string &agent_name, Peer &out) const {
    const std::lock_guard lock(mutex_);
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
    if (agent_name.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (usingKvBackend()) {
        // Centralized backend: the address is ignored (and so not validated),
        // only the name matters. Re-registering a known name is an idempotent
        // no-op rather than a silent overwrite.
        const std::lock_guard<std::mutex> lk(mutex_);
        const std::uint64_t epoch = ++registerEpoch_;
        auto [it, inserted] = peers_.try_emplace(agent_name, Peer{ip, port, epoch});
        it->second.epoch = epoch; // refresh on the idempotent path too
        NIXL_DEBUG << "[" << selfName_ << "] registerMDPeer (kv): " << agent_name
                   << (inserted ? " registered" : " already registered (idempotent)");
        return NIXL_SUCCESS;
    }
    if (ip.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // The metadata listener path is IPv4-only (inet_pton(AF_INET, ...) in
    // nixl_listener.cpp), so reject malformed addresses up front.
    in_addr parsed;
    if (inet_pton(AF_INET, ip.c_str(), &parsed) != 1) {
        NIXL_DEBUG << "[" << selfName_ << "] registerMDPeer: invalid IPv4 address '" << ip
                   << "' for " << agent_name;
        return NIXL_ERR_INVALID_PARAM;
    }
    // 0 means "use the listener-side default", matching the existing
    // nixl_opt_args_t::port default.
    const std::uint16_t resolved_port = (port == 0) ? default_comm_port : port;
    const std::lock_guard lock(mutex_);
    const std::uint64_t epoch = ++registerEpoch_;
    // Same address is idempotent; rebinding a name is rejected (unregister first).
    const auto it = peers_.find(agent_name);
    if (it != peers_.end()) {
        if (it->second.ip == ip && it->second.port == resolved_port) {
            it->second.epoch = epoch; // refresh on the idempotent path too
            NIXL_DEBUG << "[" << selfName_ << "] registerMDPeer: " << agent_name
                       << " already bound to " << ip << ":" << resolved_port << " (idempotent)";
            return NIXL_SUCCESS;
        }
        NIXL_DEBUG << "[" << selfName_ << "] registerMDPeer: rejecting rebind of " << agent_name
                   << " from " << it->second.ip << ":" << it->second.port << " to " << ip << ":"
                   << resolved_port;
        return NIXL_ERR_NOT_ALLOWED;
    }
    peers_[agent_name] = Peer{ip, resolved_port, epoch};
    NIXL_DEBUG << "[" << selfName_ << "] registerMDPeer: " << agent_name << " -> " << ip << ":"
               << resolved_port;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::unregisterMDPeer(const std::string &agent_name) {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        NIXL_DEBUG << "[" << selfName_ << "] unregisterMDPeer: " << agent_name
                   << " not registered (no-op)";
        return NIXL_SUCCESS;
    }
    if (usingKvBackend()) {
        // Centralized backend: just forget the peer locally (use
        // invalidateLocalMD to withdraw our own metadata).
        const std::lock_guard<std::mutex> lk(mutex_);
        auto it = peers_.find(agent_name);
        if (it != peers_.end() && it->second == peer) {
            peers_.erase(it);
        }
        NIXL_DEBUG << "[" << selfName_ << "] unregisterMDPeer (kv): " << agent_name << " forgotten";
        return NIXL_SUCCESS;
    }
    // Tell the remote to drop our metadata before removing the local entry.
    // Keep the entry if the call is rejected so the caller can retry.
    // peer.ip is reused below for the compare-then-erase, so it is copied
    // (not moved) into the args here.
    const auto args = makePeerArgs(peer.ip, peer.port);
    const nixl_status_t s = agent_.invalidateLocalMD(&args);
    if (s != NIXL_SUCCESS) {
        NIXL_DEBUG << "[" << selfName_ << "] unregisterMDPeer: invalidate for " << agent_name
                   << " failed: " << s;
        return s;
    }
    // Compare-then-erase: epoch equality skips the erase if a register (even
    // to the same address) landed while the invalidate was in flight.
    const std::lock_guard lock(mutex_);
    auto it = peers_.find(agent_name);
    if (it != peers_.end() && it->second == peer) {
        peers_.erase(it);
    }
    NIXL_DEBUG << "[" << selfName_ << "] unregisterMDPeer: " << agent_name << " removed";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::sendLocalMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (usingKvBackend()) {
        nixl_blob_t blob;
        const nixl_status_t ret = agent_.getLocalMD(blob);
        if (ret < 0) {
            return ret;
        }
        NIXL_DEBUG << "[" << selfName_ << "] sendLocalMD (kv): publishing for " << agent_name;
        return backend_->publish(makeKey(selfName_, default_metadata_label), blob);
    }
    // lookupPeer copies {ip, port}, so the lock is released before the call
    // below. A concurrent unregister/re-register can change the registry in
    // this gap; we intentionally use the snapshot rather than hold the lock
    // across a network call. Worst case the send targets the just-removed
    // address, which is acceptable.
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << selfName_ << "] sendLocalMD: " << agent_name;
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
    if (usingKvBackend()) {
        if (!md_extra_params || md_extra_params->metadataLabel.empty()) {
            NIXL_ERROR << "metadata label is required for kv send of local partial metadata";
            return NIXL_ERR_INVALID_PARAM;
        }
        nixl_blob_t blob;
        const nixl_status_t ret = agent_.getLocalPartialMD(descs, blob, md_extra_params);
        if (ret < 0) {
            return ret;
        }
        NIXL_DEBUG << "[" << selfName_ << "] sendLocalPartialMD (kv): publishing label '"
                   << md_extra_params->metadataLabel << "' for " << agent_name;
        return backend_->publish(makeKey(selfName_, md_extra_params->metadataLabel), blob);
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port, md_extra_params);
    NIXL_DEBUG << "[" << selfName_ << "] sendLocalPartialMD: " << agent_name;
    return agent_.sendLocalPartialMD(descs, &args);
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (usingKvBackend()) {
        drainInvalidated();
        NIXL_DEBUG << "[" << selfName_ << "] fetchRemoteMD (kv): " << agent_name;
        nixl_blob_t blob;
        nixl_status_t ret = backend_->fetch(makeKey(agent_name, default_metadata_label), blob);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
        std::string loaded_name;
        ret = agent_.loadRemoteMD(blob, loaded_name);
        if (ret < 0) {
            return ret;
        }
        // Watch the peer's subtree; a DELETE invalidates our cached copy. The
        // callback runs on the backend's watcher thread, so it only enqueues;
        // draining happens on later fetch/check calls. Watch-less backends
        // (TCPStore) treat this as a no-op and rely on republish.
        return backend_->watch(
            makeKey(agent_name, ""),
            [this, agent_name](const std::string &, nixl_watch_event_t event, const nixl_blob_t &) {
                if (event == nixl_watch_event_t::DELETE) {
                    const std::lock_guard<std::mutex> lk(invalidatedMutex_);
                    invalidatedAgents_.push_back(agent_name);
                }
            });
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << selfName_ << "] fetchRemoteMD: " << agent_name;
    return agent_.fetchRemoteMD(agent_name, &args);
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const std::string &agent_name) const {
    Peer peer;
    if (!lookupPeer(agent_name, peer)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (usingKvBackend()) {
        // Single published copy: remove our whole subtree (legacy behavior).
        NIXL_DEBUG << "[" << selfName_ << "] invalidateLocalMD (kv): removing published metadata";
        return backend_->remove(makeKey(selfName_, ""));
    }
    const auto args = makePeerArgs(std::move(peer.ip), peer.port);
    NIXL_DEBUG << "[" << selfName_ << "] invalidateLocalMD: " << agent_name;
    return agent_.invalidateLocalMD(&args);
}

nixl_status_t
nixlMDManager::checkRemoteMD(const std::string &agent_name, const nixl_xfer_dlist_t &descs) const {
    drainInvalidated();
    // TRACE (not DEBUG): callers typically poll this in a loop.
    NIXL_TRACE << "[" << selfName_ << "] checkRemoteMD: " << agent_name;
    return agent_.checkRemoteMD(agent_name, descs);
}
