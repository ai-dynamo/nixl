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

#include "nixl_metadata_context.h"
#include "nixl_tcpstore_client_async.h"
#include "nixl_types.h"

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <charconv>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

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
// budget so one degraded operation does not hold the worker for the window.
constexpr long default_op_timeout_ms = 5000;

constexpr std::chrono::milliseconds default_retry_delay = std::chrono::milliseconds(50);

[[nodiscard]] std::string
makeKey(const std::string &label, const std::string &src, const std::string &dst) {
    return std::string(namespace_prefix) + "/" + label + "/" + src + "/" + dst;
}

[[nodiscard]] std::chrono::milliseconds
bringUpTimeout() {
    return std::chrono::milliseconds(
        nixl::config::getValueDefaulted<long>("NIXL_TCPSTORE_TIMEOUT_MS", default_timeout_ms));
}

// Parse the endpoint and build the client (which connects on first use), so
// client_ can be a const member built in the init list.
[[nodiscard]] std::unique_ptr<nixlTcpStoreClientAsync>
makeClient(asio::io_context &io_context) {
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

    return std::make_unique<nixlTcpStoreClientAsync>(
        host, static_cast<std::uint16_t>(port), bringUpTimeout(), op_timeout, io_context);
}

} // namespace

nixlTcpStoreMetadataBackend::nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx,
                                                         const nixlMDConfig &config,
                                                         asio::io_context &io_context)
    : ctx_(ctx),
      fetchTimeout_(bringUpTimeout()),
      ioContext_(io_context),
      client_(makeClient(io_context)) {
    NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore backend ready";
}

nixlTcpStoreMetadataBackend::~nixlTcpStoreMetadataBackend() = default;

void
nixlTcpStoreMetadataBackend::start() {
    client_->connect();
}

void
nixlTcpStoreMetadataBackend::stop() {
    client_->close();
    stopCalled_ = true;
}

void
nixlTcpStoreMetadataBackend::publishKey(const std::string &key, const nixl_blob_t &blob) {
    client_->set(key, blob, [this, key = key](asio::error_code error) {
        if (!error) {
            publishedKeys_.insert(key);
            NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore published key " << key;
        } else {
            NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore set failed for key " << key << ": "
                       << error.message();
        }
    });
}

nixl_status_t
nixlTcpStoreMetadataBackend::sendLocal(const nixl_opt_args_t * /*extra_params*/) {
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalMD(blob);
    if (ret < 0) {
        return ret;
    }
    const std::string key = makeKey(default_metadata_label, ctx_.agentName(), nixl_null_agent);
    publishKey(key, blob);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpStoreMetadataBackend::sendLocalPartial(const nixl_reg_dlist_t &descs,
                                              const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->metadataLabel.empty()) {
        NIXL_ERROR_FUNC << "metadata label is required for TCPStore send of local partial metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, blob, extra_params);
    if (ret < 0) {
        return ret;
    }
    const std::string key = makeKey(extra_params->metadataLabel, ctx_.agentName(), nixl_null_agent);
    publishKey(key, blob);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpStoreMetadataBackend::fetchRemote(const std::string &remote_name,
                                         const nixl_opt_args_t *extra_params) {
    const std::string label = (extra_params && !extra_params->metadataLabel.empty()) ?
        extra_params->metadataLabel :
        default_metadata_label;
    const std::string key = makeKey(label, remote_name, nixl_null_agent);

    // The fetch runs on the worker thread; the result lands in the agent cache
    // (observed via checkRemoteMD), matching the async model of the other backends.
    // A key that is not published yet is kept pending rather than dropped, so a
    // caller polling checkRemoteMD still converges once the peer publishes.
    auto deadline = std::chrono::steady_clock::now() + fetchTimeout_;
    tryFetch(remote_name, key, deadline, default_retry_delay);
    return NIXL_SUCCESS;
}

void
nixlTcpStoreMetadataBackend::tryFetch(const std::string &remote_name,
                                      const std::string &key,
                                      std::chrono::steady_clock::time_point deadline,
                                      std::chrono::milliseconds retry_delay) {
    client_->get(
        key,
        [this, remote = remote_name, key = key, deadline, retry_delay](
            asio::error_code error, std::optional<std::string> blob) {
            if (!error && blob) {
                std::string loaded_name;
                const nixl_status_t ret = ctx_.loadRemoteMD(*blob, loaded_name);
                if (ret < 0) {
                    NIXL_ERROR << "[" << ctx_.agentName()
                               << "] failed to load metadata fetched for " << remote
                               << " with status " << ret;
                    return;
                }
                if (loaded_name != remote) {
                    // A corrupted or mis-keyed store value could carry another agent's
                    // metadata; reject it rather than accept it under the wrong name.
                    NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore metadata for " << remote
                               << " embeds mismatched agent name " << loaded_name;
                    ctx_.invalidateRemoteMD(loaded_name);
                    return;
                }
                NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore fetched metadata for "
                           << remote << ", size: " << blob->size();
            } else {
                if (std::chrono::steady_clock::now() >= deadline) {
                    NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore fetch for " << remote
                               << " gave up waiting for key " << key;
                    return;
                }
                if (!error) {
                    NIXL_DEBUG << "[" << ctx_.agentName()
                               << "] TCPStore key not yet present: " << key;
                }
                if (!stopCalled_) {
                    auto timer = std::make_unique<asio::steady_timer>(ioContext_, retry_delay);
                    timer->async_wait(
                        [this, remote, key, deadline, retry_delay, timer = std::move(timer)](
                            asio::error_code) {
                            if (!stopCalled_) {
                                tryFetch(remote, key, deadline, retry_delay);
                            }
                        });
                }
            }
        });
}

nixl_status_t
nixlTcpStoreMetadataBackend::invalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    // Access the published keys from the I/O thread, so we do not need to
    // protect it with a mutex.
    asio::post(ioContext_, [this]() {
        for (const auto &key : publishedKeys_) {
            client_->deleteKey(key, [this, key = key](asio::error_code error, bool success) {
                if (!error && success) {
                    NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore invalidated key " << key;
                } else if (error) {
                    NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore invalidate failed for key "
                               << key << ": " << error.message();
                } else {
                    NIXL_INFO << "[" << ctx_.agentName() << "] TCPStore invalidate skipped for key "
                              << key << ": key does not exist";
                }
            });
        }
        publishedKeys_.clear();
    });
    return NIXL_SUCCESS;
}
