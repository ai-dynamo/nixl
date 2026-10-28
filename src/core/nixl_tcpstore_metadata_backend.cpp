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

// Default retry delay for fetch operations.
constexpr auto default_retry_delay = std::chrono::milliseconds(50);

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

class nixlTcpStoreMetadataBackend::WriteTask {
public:
    WriteTask(nixlTcpStoreMetadataBackend &backend) : backend_(backend) {}

    virtual ~WriteTask() = default;

    virtual void
    initiate() = 0;

    nixlTcpStoreMetadataBackend &backend_;
};

nixlTcpStoreMetadataBackend::nixlTcpStoreMetadataBackend(nixlMetadataContext &ctx,
                                                         const nixlMDConfig &config,
                                                         asio::io_context &io_context)
    : fetchTimeout_(bringUpTimeout()),
      ctx_(ctx),
      ioContext_(io_context),
      client_(makeClient(io_context)),
      fetchRetryTimer_(io_context, std::chrono::steady_clock::now()) {
    NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore backend ready";
}

nixlTcpStoreMetadataBackend::~nixlTcpStoreMetadataBackend() = default;

void
nixlTcpStoreMetadataBackend::start() {
    client_->connect();
}

void
nixlTcpStoreMetadataBackend::stop() {
    asio::post(ioContext_, [this]() {
        stopCalled_ = true;
        pendingKeys_.clear();
        fetchRetryTimer_.cancel();
        if (!inflightTask_) {
            assert(writeQueue_.empty());
            client_->close();
        }
    });
}

void
nixlTcpStoreMetadataBackend::publishKey(const std::string &key, const nixl_blob_t &blob) {
    class publishKeyTask : public nixlTcpStoreMetadataBackend::WriteTask {
    public:
        publishKeyTask(nixlTcpStoreMetadataBackend &backend,
                       const std::string &key,
                       const nixl_blob_t &blob)
            : WriteTask(backend),
              key_(key),
              blob_(blob) {}

        void
        initiate() override {
            backend_.client_->set(key_, blob_, [this](asio::error_code error) {
                if (!error) {
                    NIXL_DEBUG << "[" << backend_.ctx_.agentName() << "] TCPStore published key "
                               << key_;
                    backend_.publishedKeys_.insert(key_);
                } else {
                    NIXL_ERROR << "[" << backend_.ctx_.agentName()
                               << "] TCPStore set failed for key " << key_ << ": "
                               << error.message();
                }
                backend_.writeTaskCompleted(this);
            });
        }

        std::string key_;
        nixl_blob_t blob_;
    };

    postWriteTask(std::make_unique<publishKeyTask>(*this, key, blob));
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
    auto deadline = std::chrono::steady_clock::now() + fetchTimeout_;

    // The fetch runs on the I/O thread. The result lands in the agent cache
    // (observed via checkRemoteMD), matching the async model of the other backends.
    asio::post(ioContext_, [this, remote = remote_name, key = key, deadline]() {
        if (stopCalled_) {
            return;
        }
        auto it = pendingKeys_.find(key);
        if (it != pendingKeys_.end()) {
            // Update the deadline and let the retry timer handle the next attempt.
            it->second.deadline = std::max(deadline, it->second.deadline);
            return;
        }

        pendingKeys_.try_emplace(key, pendingInfo{deadline, remote, true});
        initiateFetch(remote, key);
    });

    return NIXL_SUCCESS;
}

void
nixlTcpStoreMetadataBackend::initiateFetch(const std::string &remote_name, const std::string &key) {
    client_->get(key,
                 [this, key = key, remote = remote_name](asio::error_code error,
                                                         std::optional<std::string> blob) {
                     if (!error && blob) {
                         // Fetch succeeded.
                         pendingKeys_.erase(key);

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
                             NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore metadata for "
                                        << remote << " embeds mismatched agent name "
                                        << loaded_name;
                             ctx_.invalidateRemoteMD(loaded_name);
                             return;
                         }
                         NIXL_DEBUG << "[" << ctx_.agentName() << "] TCPStore fetched metadata for "
                                    << remote << ", size: " << blob->size();
                     } else {
                         if (!error) {
                             NIXL_DEBUG << "[" << ctx_.agentName()
                                        << "] TCPStore key not yet present: " << key;
                         }

                         auto fetch_deadline = std::chrono::steady_clock::now();
                         auto it = pendingKeys_.find(key);
                         if (it != pendingKeys_.end()) {
                             fetch_deadline = it->second.deadline;
                         }
                         if (std::chrono::steady_clock::now() >= fetch_deadline) {
                             NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore fetch for "
                                        << remote << " gave up waiting for key " << key;
                             pendingKeys_.erase(key);
                             return;
                         }

                         // A key that is not published yet is kept retrying rather than dropped, so
                         // a caller polling checkRemoteMD still converges once the peer publishes.
                         if (!stopCalled_) {
                             it->second.fetchInProgress = false;
                             if (fetchRetryTimer_.expiry() < std::chrono::steady_clock::now()) {
                                 // Arm the retry timer.
                                 fetchRetryTimer_.expires_after(default_retry_delay);
                                 fetchRetryTimer_.async_wait([this](asio::error_code error) {
                                     if (!error && !stopCalled_) {
                                         retryPendingFetches();
                                     }
                                 });
                             }
                         }
                     }
                 });
}

void
nixlTcpStoreMetadataBackend::retryPendingFetches() {
    for (auto it = pendingKeys_.begin(); it != pendingKeys_.end();) {
        auto &[key, info] = *it;
        if (info.fetchInProgress) {
            ++it;
            continue;
        }

        if (std::chrono::steady_clock::now() < info.deadline) {
            info.fetchInProgress = true;
            initiateFetch(info.remoteName, key);
            ++it;
        } else {
            NIXL_ERROR << "[" << ctx_.agentName() << "] TCPStore fetch for " << info.remoteName
                       << " gave up waiting for key " << key;
            it = pendingKeys_.erase(it);
        }
    }
}

nixl_status_t
nixlTcpStoreMetadataBackend::invalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    class invalidateLocalTask : public nixlTcpStoreMetadataBackend::WriteTask {
    public:
        invalidateLocalTask(nixlTcpStoreMetadataBackend &backend) : WriteTask(backend) {}

        void
        initiate() override {
            if (backend_.publishedKeys_.empty()) {
                backend_.writeTaskCompleted(this);
                return;
            }

            pendingCount_ = backend_.publishedKeys_.size();

            for (const auto &key : backend_.publishedKeys_) {
                backend_.client_->deleteKey(
                    key, [this, key = key](asio::error_code error, bool success) {
                        if (!error && success) {
                            NIXL_DEBUG << "[" << backend_.ctx_.agentName()
                                       << "] TCPStore invalidated key " << key;
                            backend_.publishedKeys_.erase(key);
                        } else if (error) {
                            NIXL_ERROR << "[" << backend_.ctx_.agentName()
                                       << "] TCPStore invalidate failed for key " << key << ": "
                                       << error.message();
                        } else {
                            NIXL_INFO << "[" << backend_.ctx_.agentName()
                                      << "] TCPStore invalidate skipped for key " << key
                                      << ": key does not exist";
                        }

                        if (--pendingCount_ == 0) {
                            backend_.writeTaskCompleted(this);
                        }
                    });
            }
        }

        std::size_t pendingCount_;
    };

    postWriteTask(std::make_unique<invalidateLocalTask>(*this));

    return NIXL_SUCCESS;
}

void
nixlTcpStoreMetadataBackend::postWriteTask(std::unique_ptr<WriteTask> task) {
    asio::post(ioContext_, [this, task = std::move(task)]() mutable {
        if (stopCalled_) {
            return;
        }

        if (inflightTask_) {
            writeQueue_.push_back(std::move(task));
            return;
        }

        assert(writeQueue_.empty());

        inflightTask_ = std::move(task);
        inflightTask_->initiate();
    });
}

// Must be called from the I/O thread.
void
nixlTcpStoreMetadataBackend::writeTaskCompleted(const WriteTask *completed_task) {
    assert(inflightTask_.get() == completed_task);
    inflightTask_.reset();

    if (writeQueue_.empty()) {
        if (stopCalled_) {
            client_->close();
        }
        return;
    }

    inflightTask_ = std::move(writeQueue_.front());
    writeQueue_.pop_front();
    inflightTask_->initiate();
}
