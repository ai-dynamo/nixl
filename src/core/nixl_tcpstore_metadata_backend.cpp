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
#include <stdexcept>
#include <utility>

namespace {

// Key layout mirrors the ETCD namespace and the architecture doc (Sec 2.4):
// {namespace}/{label}/{src_agent}/{dst_agent | null_agent}. Full/broadcast
// metadata uses dst = nixl_null_agent (shared to all).
constexpr char namespace_prefix[] = "/nixl/agents";

// Bound the connect and each blocking store op. Overridable for slow bring-up.
constexpr long default_timeout_ms = 30000;

[[nodiscard]] std::string
makeKey(const std::string &label, const std::string &src, const std::string &dst) {
    return std::string(namespace_prefix) + "/" + label + "/" + src + "/" + dst;
}

} // namespace

nixlTcpStoreMetadataBackend::nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx) : ctx_(ctx) {
    const std::string endpoints = nixl::config::getNonEmptyString("NIXL_TCPSTORE_ENDPOINTS");

    // NIXL_TCPSTORE_ENDPOINTS is a single host:port (rfind keeps IPv4 hosts and
    // bracketless names simple; the port is the trailing field).
    const auto pos = endpoints.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= endpoints.size()) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINTS must be host:port, got: " + endpoints);
    }
    const std::string host = endpoints.substr(0, pos);
    const std::string port_str = endpoints.substr(pos + 1);
    // from_chars (not stoul) so trailing junk like "123abc" is rejected instead
    // of silently parsing to 123 and connecting to the wrong port.
    unsigned int port = 0;
    const auto [parse_end, ec] =
        std::from_chars(port_str.data(), port_str.data() + port_str.size(), port);
    if (ec != std::errc{} || parse_end != port_str.data() + port_str.size() || port == 0 ||
        port > 65535) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINTS has invalid port: " + port_str);
    }

    const long timeout_ms =
        nixl::config::getValueDefaulted<long>("NIXL_TCPSTORE_TIMEOUT_MS", default_timeout_ms);

    client_ = std::make_unique<nixlTcpStoreClient>(
        host, static_cast<std::uint16_t>(port), std::chrono::milliseconds(timeout_ms));
    NIXL_DEBUG << "[" << ctx_.getName() << "] connected TCPStore client to " << endpoints;
}

nixlTcpStoreMetadataBackend::~nixlTcpStoreMetadataBackend() = default;

nixl_status_t
nixlTcpStoreMetadataBackend::publishKey(const std::string &key, const nixl_blob_t &blob) {
    // Hold publishedMutex_ across the store write and the tracking update so a
    // concurrent invalidateLocal cannot delete this key's snapshot mid-publish,
    // which would drop the fresh value and leave publishedKeys_ out of sync.
    const std::lock_guard<std::mutex> lk(publishedMutex_);
    try {
        client_->set(key, blob);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "[" << ctx_.getName() << "] TCPStore set failed for key " << key << ": "
                   << e.what();
        return NIXL_ERR_BACKEND;
    }
    publishedKeys_.insert(key);
    NIXL_DEBUG << "[" << ctx_.getName() << "] TCPStore published key " << key;
    return NIXL_SUCCESS;
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareSendLocal(const nixl_opt_args_t * /*extra_params*/) {
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalMD(blob);
    if (ret < 0) {
        return {ret, {}};
    }
    const std::string key = makeKey(default_metadata_label, ctx_.getName(), nixl_null_agent);
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
    const std::string key = makeKey(extra_params->metadataLabel, ctx_.getName(), nixl_null_agent);
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
    return {NIXL_SUCCESS, [this, remote_name, key]() {
                try {
                    if (!client_->check(key)) {
                        NIXL_DEBUG << "[" << ctx_.getName()
                                   << "] TCPStore key not yet present: " << key;
                        return;
                    }
                    const nixl_blob_t blob = client_->get(key);
                    if (blob.empty()) {
                        // The key was deleted between check() and get() (raced invalidate);
                        // an empty read is never a valid MD blob.
                        NIXL_DEBUG << "[" << ctx_.getName()
                                   << "] TCPStore key vanished between check and get: " << key;
                        return;
                    }
                    std::string loaded_name;
                    const nixl_status_t ret = ctx_.loadRemoteMD(blob, loaded_name);
                    if (ret < 0) {
                        NIXL_ERROR << "[" << ctx_.getName()
                                   << "] failed to load metadata fetched for " << remote_name
                                   << " with status " << ret;
                        return;
                    }
                    if (loaded_name != remote_name) {
                        // A corrupted or mis-keyed store value could carry another agent's
                        // metadata; reject it rather than accept it under the wrong name.
                        NIXL_ERROR << "[" << ctx_.getName() << "] TCPStore metadata for "
                                   << remote_name << " embeds mismatched agent name "
                                   << loaded_name;
                        return;
                    }
                    NIXL_DEBUG << "[" << ctx_.getName() << "] TCPStore fetched metadata for "
                               << remote_name;
                }
                catch (const std::exception &e) {
                    NIXL_ERROR << "[" << ctx_.getName() << "] TCPStore fetch failed for key " << key
                               << ": " << e.what();
                }
            }};
}

nixlPreparedOp
nixlTcpStoreMetadataBackend::prepareInvalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    return {NIXL_SUCCESS, [this]() {
                // Hold publishedMutex_ across the deletes and the tracking reset so this
                // serializes with publishKey; lock order is always publishedMutex_ ->
                // client mutex, so there is no deadlock.
                const std::lock_guard<std::mutex> lk(publishedMutex_);
                try {
                    for (const auto &key : publishedKeys_) {
                        client_->deleteKey(key);
                    }
                }
                catch (const std::exception &e) {
                    // Keep publishedKeys_ intact so a later invalidate retries the deletes.
                    NIXL_ERROR << "[" << ctx_.getName()
                               << "] TCPStore invalidate failed: " << e.what();
                    return;
                }
                const std::size_t count = publishedKeys_.size();
                publishedKeys_.clear();
                NIXL_DEBUG << "[" << ctx_.getName() << "] TCPStore invalidated " << count
                           << " key(s)";
            }};
}
