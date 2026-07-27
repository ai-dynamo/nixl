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
#include "nixl_tcpstore_metadata_backend.h"

#include "agent_data.h"
#include "nixl_tcpstore_client.h"
#include "nixl_types.h"

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <charconv>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Key layout mirrors the ETCD namespace and the architecture doc (Sec 2.4):
// {namespace}/{label}/{src_agent}/{dst_agent | null_agent}. Full/broadcast
// metadata uses dst = nixl_null_agent (shared to all). No leading slash: the
// client prepends the "/" that c10d clients use.
constexpr char namespace_prefix[] = "nixl/agents";

// Bring-up budget: bounds the initial connect and how long a fetch keeps
// waiting for a peer that has not published yet. Overridable for slow bring-up.
constexpr long default_timeout_ms = 30000;

// Bound on a single store operation once running. Kept well under the bring-up
// budget because these run on the manager's shared worker thread.
constexpr long default_op_timeout_ms = 5000;

[[nodiscard]] std::string
makeKey(const std::string &label, const std::string &src, const std::string &dst) {
    return std::string(namespace_prefix) + "/" + label + "/" + src + "/" + dst;
}

[[nodiscard]] std::chrono::milliseconds
bringUpTimeout() {
    return std::chrono::milliseconds(
        nixl::config::getValueDefaulted<long>("NIXL_TCPSTORE_TIMEOUT_MS", default_timeout_ms));
}

// Parse and connect, so client_ can be a const member built in the init list.
[[nodiscard]] std::unique_ptr<nixlTcpStoreClient>
makeClient() {
    const std::string endpoint = nixl::config::getNonEmptyString("NIXL_TCPSTORE_ENDPOINT");

    // NIXL_TCPSTORE_ENDPOINT is a single host:port (rfind keeps IPv4 hosts and
    // bracketless names simple; the port is the trailing field).
    const auto pos = endpoint.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= endpoint.size()) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINT must be host:port, got: " + endpoint);
    }
    const std::string host = endpoint.substr(0, pos);
    const std::string port_str = endpoint.substr(pos + 1);
    // from_chars (not stoul) so trailing junk like "123abc" is rejected instead
    // of silently parsing to 123 and connecting to the wrong port.
    unsigned int port = 0;
    const auto [parse_end, ec] =
        std::from_chars(port_str.data(), port_str.data() + port_str.size(), port);
    if (ec != std::errc{} || parse_end != port_str.data() + port_str.size() || port == 0 ||
        port > 65535) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINT has invalid port: " + port_str);
    }

    const auto op_timeout = std::chrono::milliseconds(nixl::config::getValueDefaulted<long>(
        "NIXL_TCPSTORE_OP_TIMEOUT_MS", default_op_timeout_ms));

    return std::make_unique<nixlTcpStoreClient>(
        host, static_cast<std::uint16_t>(port), bringUpTimeout(), op_timeout);
}

} // namespace

nixlTcpStoreMetadataBackend::nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx)
    : ctx_(ctx),
      fetchTimeout_(bringUpTimeout()),
      client_(makeClient()) {
    NIXL_DEBUG << "[" << ctx_.agentName() << "] connected TCPStore client";
}

nixlTcpStoreMetadataBackend::~nixlTcpStoreMetadataBackend() = default;

nixl_status_t
nixlTcpStoreMetadataBackend::publishKey(const std::string &key, const nixl_blob_t &blob) {
    // Hold publishedMutex_ across the store write and the tracking update so a
    // concurrent invalidateLocal cannot delete this key's snapshot mid-publish,
    // which would drop the fresh value and leave publishedKeys_ out of sync.
    const std::lock_guard lk(publishedMutex_);
    try {
        client_->set(key, blob);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore set failed for key " << key << ": "
                   << e.what();
        return NIXL_ERR_BACKEND;
    }
    publishedKeys_.insert(key);
    NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore published key " << key;
    return NIXL_SUCCESS;
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareSendLocal(const nixl_opt_args_t * /*extra_params*/) {
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalMD(blob);
    if (ret < 0) {
        return {ret, {}};
    }
    const std::string key = makeKey(default_metadata_label, ctx_.agentName(), nixl_null_agent);
    return {NIXL_SUCCESS, [this, key, blob = std::move(blob)]() { (void)publishKey(key, blob); }};
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareSendLocalPartial(const nixl_reg_dlist_t &descs,
                                                     const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->metadataLabel.empty()) {
        NIXL_ERROR_FUNC << "metadata label is required for TCPStore send of local partial metadata";
        return {NIXL_ERR_INVALID_PARAM, {}};
    }
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, blob, extra_params);
    if (ret < 0) {
        return {ret, {}};
    }
    const std::string key = makeKey(extra_params->metadataLabel, ctx_.agentName(), nixl_null_agent);
    return {NIXL_SUCCESS, [this, key, blob = std::move(blob)]() { (void)publishKey(key, blob); }};
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareFetchRemote(const std::string &remote_name,
                                                const nixl_opt_args_t *extra_params) {
    const std::string label = (extra_params && !extra_params->metadataLabel.empty()) ?
        extra_params->metadataLabel :
        default_metadata_label;
    const std::string key = makeKey(label, remote_name, nixl_null_agent);

    // The fetch runs on the worker thread; the result lands in the agent cache
    // (observed via checkRemoteMD), matching the async model of the other backends.
    // A key that is not published yet is kept pending rather than dropped, so a
    // caller polling checkRemoteMD still converges once the peer publishes.
    return {NIXL_SUCCESS, [this, remote_name, key]() {
                if (!tryFetch(remote_name, key)) {
                    pendingFetches_[remote_name] = {
                        key, std::chrono::steady_clock::now() + fetchTimeout_};
                }
            }};
}

bool
nixlTcpStoreMetadataBackend::tryFetch(const std::string &remote_name, const std::string &key) {
    try {
        const std::optional<std::string> blob = client_->get(key);
        if (!blob) {
            NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore key not yet present: " << key;
            return false;
        }
        std::string loaded_name;
        const nixl_status_t ret = ctx_.loadRemoteMD(*blob, loaded_name);
        if (ret < 0) {
            NIXL_ERROR << "[" << ctx_.agentName() << "] failed to load metadata fetched for "
                       << remote_name << " with status " << ret;
            return true;
        }
        if (loaded_name != remote_name) {
            // A corrupted or mis-keyed store value could carry another agent's
            // metadata; reject it rather than accept it under the wrong name.
            NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore metadata for " << remote_name
                       << " embeds mismatched agent name " << loaded_name;
            return true;
        }
        NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore fetched metadata for " << remote_name;
        return true;
    }
    catch (const std::exception &e) {
        // Transport failure: the client reconnects on the next attempt, so this
        // is retried like an absent key rather than failing the fetch outright.
        NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore fetch failed for key " << key << ": "
                   << e.what();
        return false;
    }
}

void
nixlTcpStoreMetadataBackend::serviceEvents() {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = pendingFetches_.begin(); it != pendingFetches_.end();) {
        if (tryFetch(it->first, it->second.key)) {
            it = pendingFetches_.erase(it);
        } else if (now >= it->second.deadline) {
            NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore fetch for " << it->first
                       << " gave up waiting for key " << it->second.key;
            it = pendingFetches_.erase(it);
        } else {
            ++it;
        }
    }
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareInvalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    return {NIXL_SUCCESS, [this]() {
                // Hold publishedMutex_ across the deletes and the tracking reset so this
                // serializes with publishKey; lock order is always publishedMutex_ ->
                // client mutex, so there is no deadlock.
                const std::lock_guard lk(publishedMutex_);
                try {
                    for (const auto &key : publishedKeys_) {
                        client_->deleteKey(key);
                    }
                }
                catch (const std::exception &e) {
                    // Keep publishedKeys_ intact so a later invalidate retries the deletes.
                    NIXL_ERROR << "[" << ctx_.agentName()
                               << "] TCPStore invalidate failed: " << e.what();
                    return;
                }
                const std::size_t count = publishedKeys_.size();
                publishedKeys_.clear();
                NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore invalidated " << count
                           << " key(s)";
            }};
}
