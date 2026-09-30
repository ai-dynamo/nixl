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
#if HAVE_ETCD

#include "nixl_etcd_metadata_backend.h"

#include "nixl_metadata_context.h"
#include "nixl_types.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr char default_namespace[] = "/nixl/agents/";
} // namespace

// Hand-rolled etcd client (connection + watchers). Moved from the former
// nixl_listener.cpp, with processInvalidatedAgents now driving the cache through
// nixlMetadataContext instead of a nixlAgent pointer. Nested in the backend so
// this implementation type has no linkage of its own to collide with.
class nixlEtcdMetadataBackend::etcdClient {
private:
    std::unique_ptr<etcd::SyncClient> etcd_;
    const std::string namespacePrefix_;
    const std::string agentName_;
    std::vector<std::string> invalidatedAgents_;
    std::mutex invalidatedAgentsMutex_;
    std::unordered_map<std::string, std::unique_ptr<etcd::Watcher>> agentWatchers_;
    std::chrono::microseconds watchTimeout_;
    bool agentPrefixStored_ = false;

    [[nodiscard]] std::string
    makeKey(const std::string &agent_name, const std::string &metadata_type) const {
        return namespacePrefix_ + "/" + agent_name + "/" + metadata_type;
    }

    // Announce this agent by writing its prefix key. Deferred to the first
    // publish so construction does no I/O and a store that is briefly
    // unreachable does not fail agent construction.
    nixl_status_t
    ensureAgentPrefix() {
        if (agentPrefixStored_) {
            return NIXL_SUCCESS;
        }
        const etcd::Response response = etcd_->put(makeKey(agentName_, ""), "");
        if (!response.is_ok()) {
            NIXL_ERROR << "Failed to store agent " << agentName_
                       << " prefix key in etcd: " << response.error_message();
            return NIXL_ERR_BACKEND;
        }
        agentPrefixStored_ = true;
        return NIXL_SUCCESS;
    }

public:
    // Builds the client object (no connection is established here; the gRPC
    // channel connects on first use). Throws only when the endpoint list itself
    // is unusable.
    explicit etcdClient(std::string my_agent_name, std::chrono::microseconds timeout)
        : namespacePrefix_(nixl::config::getValueDefaulted<std::string>("NIXL_ETCD_NAMESPACE",
                                                                        default_namespace)),
          agentName_(std::move(my_agent_name)),
          watchTimeout_(timeout) {
        const auto etcd_endpoints = nixl::config::getNonEmptyString("NIXL_ETCD_ENDPOINTS");
        etcd_ = std::make_unique<etcd::SyncClient>(etcd_endpoints);
        NIXL_DEBUG << "Created etcd client to endpoints: " << etcd_endpoints;
        NIXL_DEBUG << "Using etcd namespace for agents: " << namespacePrefix_;
    }

    // First store round-trip, run once at start-up: it both announces this agent
    // and reports an unreachable endpoint at bring-up instead of at the first
    // publish. Later publishes find the prefix already stored and skip it.
    nixl_status_t
    announceAgent() {
        try {
            return ensureAgentPrefix();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to announce agent " << agentName_ << " in etcd: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    nixl_status_t
    storeMetadataInEtcd(const std::string &agent_name,
                        const std::string &metadata_type,
                        const nixl_blob_t &metadata) {
        try {
            if (const nixl_status_t ret = ensureAgentPrefix(); ret != NIXL_SUCCESS) {
                return ret;
            }
            const std::string metadata_key = makeKey(agent_name, metadata_type);
            const etcd::Response response = etcd_->put(metadata_key, metadata);
            if (response.is_ok()) {
                NIXL_DEBUG << "Successfully stored " << metadata_type
                           << " in etcd with key: " << metadata_key << " (rev "
                           << response.value().modified_index() << ")";
                return NIXL_SUCCESS;
            }
            NIXL_ERROR << "Failed to store " << metadata_type
                       << " in etcd: " << response.error_message();
            return NIXL_ERR_BACKEND;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error sending " << metadata_type << " to etcd: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    nixl_status_t
    removeMetadataFromEtcd(const std::string &agent_name) {
        // Nothing published means nothing to remove, and rmdir on a prefix with
        // no keys is a client-side error rather than a no-op, so invalidating
        // twice (an explicit call, then teardown) would report a failure.
        if (agent_name == agentName_ && !agentPrefixStored_) {
            return NIXL_SUCCESS;
        }
        try {
            const std::string agent_prefix = makeKey(agent_name, "");
            const etcd::Response response = etcd_->rmdir(agent_prefix, true);
            if (agent_name == agentName_) {
                // Our own prefix key went with it, so a later publish re-announces.
                agentPrefixStored_ = false;
            }
            if (response.is_ok()) {
                NIXL_DEBUG << "Successfully removed " << response.values().size()
                           << " etcd keys for agent: " << agent_name;
                return NIXL_SUCCESS;
            }
            NIXL_ERROR << "Failed to remove etcd keys for agent: " << agent_name << " : "
                       << response.error_message();
            return NIXL_ERR_BACKEND;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Exception removing etcd keys for agent: " << agent_name << " : "
                       << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    nixl_status_t
    fetchMetadataFromEtcd(const std::string &agent_name,
                          const std::string &metadata_type,
                          nixl_blob_t &metadata) {
        const std::string metadata_key = makeKey(agent_name, metadata_type);
        try {
            const etcd::Response response = etcd_->get(metadata_key);
            if (response.is_ok()) {
                metadata = response.value().as_string();
                if (metadata.empty()) {
                    // A present-but-empty value is never a valid MD blob (e.g. the
                    // agent-prefix anchor, or a key observed mid-write). Treat as
                    // not-found so the caller watches/retries instead of handing an
                    // empty blob to loadRemoteMD.
                    NIXL_INFO << "Fetched empty value for key: " << metadata_key
                              << "; treating as not found";
                    return NIXL_ERR_NOT_FOUND;
                }
                NIXL_DEBUG << "Successfully fetched key: " << metadata_key << " (rev "
                           << response.value().modified_index() << ")";
                return NIXL_SUCCESS;
            }
            NIXL_INFO << "Failed to fetch key: " << metadata_key
                      << " from etcd: " << response.error_message();
            return NIXL_ERR_NOT_FOUND;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error fetching key: " << metadata_key << " from etcd: " << e.what();
            return NIXL_ERR_UNKNOWN;
        }
    }

    nixl_status_t
    waitForMetadataFromEtcd(const std::string &metadata_key, nixl_blob_t &remote_metadata) {
        try {
            const etcd::Response response = etcd_->get(metadata_key);
            const int64_t watch_index = response.index();
            std::promise<nixl_status_t> ret_prom;
            auto future = ret_prom.get_future();
            std::atomic<bool> promise_set{false};

            auto watcher_callback = [&](etcd::Response response) -> void {
                if (promise_set.exchange(true)) {
                    NIXL_DEBUG << "Ignoring subsequent watch event for key: " << metadata_key;
                    return;
                }
                if (!response.is_ok()) {
                    NIXL_ERROR << "Watch failed for key: " << metadata_key << " : "
                               << response.error_message();
                    ret_prom.set_value(NIXL_ERR_BACKEND);
                    return;
                }
                if (response.action() == "delete") {
                    NIXL_ERROR << "Watch response: metadata key deleted: " << metadata_key;
                    ret_prom.set_value(NIXL_ERR_INVALID_PARAM);
                    return;
                }
                remote_metadata = response.value().as_string();
                NIXL_DEBUG << "Watch response: metadata key fetched: " << metadata_key;
                ret_prom.set_value(NIXL_SUCCESS);
            };

            auto watcher = etcd::Watcher(*etcd_, metadata_key, watch_index, watcher_callback);

            const auto status = future.wait_for(watchTimeout_);
            // Cancel before returning so the callback cannot fire after the
            // stack locals it captures by reference go out of scope.
            watcher.Cancel();
            if (status == std::future_status::timeout) {
                NIXL_ERROR << "Watch timed out for key: " << metadata_key;
                return NIXL_ERR_BACKEND;
            }
            return future.get();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error watching etcd for key: " << metadata_key << " : " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    nixl_status_t
    fetchOrWaitForMetadataFromEtcd(const std::string &remote_agent,
                                   const std::string &metadata_label,
                                   nixl_blob_t &remote_metadata) {
        const nixl_status_t ret =
            fetchMetadataFromEtcd(remote_agent, metadata_label, remote_metadata);
        if (ret == NIXL_SUCCESS) {
            return NIXL_SUCCESS;
        }
        const std::string metadata_key = makeKey(remote_agent, metadata_label);
        NIXL_DEBUG << "Metadata not found, setting up watch for: " << metadata_key;
        return waitForMetadataFromEtcd(metadata_key, remote_metadata);
    }

    void
    setupAgentWatcher(const std::string &agent_name) {
        if (agentWatchers_.find(agent_name) != agentWatchers_.end()) {
            // Expected: fetchRemote re-arms after every successful fetch, and
            // this is what keeps watchers from stacking up per fetch.
            NIXL_DEBUG << "Watcher already registered for agent: " << agent_name;
            return;
        }
        // DELETE events are enqueued to be processed in serviceEvents (can't be
        // done inside the Watcher callback).
        const auto process_response = [this, agent_name](etcd::Response response) -> void {
            if (!response.is_ok()) {
                NIXL_ERROR << "Watcher failed to watch agent " << agent_name
                           << " from etcd: " << response.error_message();
                return;
            }
            NIXL_DEBUG << "Watcher received " << response.events().size() << " events from etcd";
            if (response.events().size() != 1) {
                NIXL_ERROR << "Watcher agent " << agent_name
                           << " received unexpected number of events from etcd: "
                           << response.events().size();
                return;
            }
            const auto &event = response.events()[0];
            if (event.event_type() == etcd::Event::EventType::DELETE_) {
                NIXL_DEBUG << "Watcher DELETE: " << event.kv().key() << " (rev "
                           << event.kv().modified_index() << ")";
                const std::lock_guard lock(invalidatedAgentsMutex_);
                invalidatedAgents_.push_back(agent_name);
            } else {
                NIXL_ERROR << "Watcher for " << event.kv().key()
                           << " received unexpected event from etcd: "
                           << static_cast<int>(event.event_type());
            }
        };
        const std::string agent_prefix = makeKey(agent_name, "");
        agentWatchers_[agent_name] =
            std::make_unique<etcd::Watcher>(*etcd_, agent_prefix, process_response);
    }

    void
    processInvalidatedAgents(nixlMetadataContext &ctx) {
        std::vector<std::string> tmp_invalidated_agents;
        {
            const std::lock_guard lock(invalidatedAgentsMutex_);
            tmp_invalidated_agents = std::move(invalidatedAgents_);
        }
        for (const auto &agent : tmp_invalidated_agents) {
            NIXL_DEBUG << "Invalidated agent: " << agent;
            agentWatchers_.erase(agent);
            const nixl_status_t ret = ctx.invalidateRemoteMD(agent);
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to invalidate remote metadata for agent: " << agent << ": "
                           << ret;
            } else {
                NIXL_DEBUG << "Successfully invalidated remote metadata for agent: " << agent;
            }
        }
    }
};

nixlEtcdMetadataBackend::nixlEtcdMetadataBackend(nixlMetadataContext &ctx,
                                                 const nixlMDConfig &config)
    : ctx_(ctx),
      workerDelay_(config.workerDelay),
      client_(std::make_unique<etcdClient>(ctx.agentName(), config.etcdWatchTimeout)) {}

nixlEtcdMetadataBackend::~nixlEtcdMetadataBackend() = default;

std::string_view
nixlEtcdMetadataBackend::name() const {
    return "ETCD";
}

void
nixlEtcdMetadataBackend::start() {
    worker_.start([this] { serviceEvents(); }, workerDelay_);
    // Announcing here rather than in the constructor keeps network I/O off agent
    // construction; queueing it rather than waiting for the first publish still
    // surfaces an unreachable store at bring-up.
    worker_.submit([this] { (void)client_->announceAgent(); });
}

void
nixlEtcdMetadataBackend::stop() {
    worker_.stop();
}

void
nixlEtcdMetadataBackend::serviceEvents() {
    client_->processInvalidatedAgents(ctx_);
}

nixl_status_t
nixlEtcdMetadataBackend::sendLocal(const nixl_opt_args_t * /*extra_params*/) {
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalMD(blob);
    if (ret < 0) {
        return ret;
    }
    const std::string agent = ctx_.agentName();
    worker_.submit([this, agent, blob = std::move(blob)]() {
        (void)client_->storeMetadataInEtcd(agent, default_metadata_label, blob);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::sendLocalPartial(const nixl_reg_dlist_t &descs,
                                          const nixl_opt_args_t *extra_params) {
    if (!extra_params || extra_params->metadataLabel.empty()) {
        NIXL_ERROR_FUNC << "metadata label is required for etcd send of local partial metadata";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_blob_t blob;
    const nixl_status_t ret = ctx_.getLocalPartialMD(descs, blob, extra_params);
    if (ret < 0) {
        return ret;
    }
    const std::string agent = ctx_.agentName();
    const std::string label = extra_params->metadataLabel;
    worker_.submit([this, agent, label, blob = std::move(blob)]() {
        (void)client_->storeMetadataInEtcd(agent, label, blob);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::fetchRemote(const std::string &remote_name,
                                     const nixl_opt_args_t *extra_params) {
    const std::string label = (extra_params && !extra_params->metadataLabel.empty()) ?
        extra_params->metadataLabel :
        default_metadata_label;
    worker_.submit([this, remote_name, label]() {
        nixl_blob_t blob;
        if (client_->fetchOrWaitForMetadataFromEtcd(remote_name, label, blob) != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to fetch metadata from etcd for agent: " << remote_name;
            return;
        }
        std::string loaded_name;
        const nixl_status_t ret = ctx_.loadRemoteMD(blob, loaded_name);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to load remote metadata for agent: " << remote_name << ": "
                       << ret;
            return;
        }
        if (loaded_name != remote_name) {
            NIXL_ERROR << "Metadata mismatch for agent: " << remote_name
                       << " from md: " << loaded_name;
            return;
        }
        NIXL_DEBUG << "Successfully loaded metadata for agent: " << remote_name;
        client_->setupAgentWatcher(remote_name);
    });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlEtcdMetadataBackend::invalidateLocal(const nixl_opt_args_t * /*extra_params*/) {
    const std::string agent = ctx_.agentName();
    worker_.submit([this, agent]() { (void)client_->removeMetadataFromEtcd(agent); });
    return NIXL_SUCCESS;
}

#endif // HAVE_ETCD
