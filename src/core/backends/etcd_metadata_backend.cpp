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
#include "etcd_metadata_backend.h"

#if HAVE_ETCD

#include <future>
#include <sstream>
#include <utility>

#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>

#include "agent_data.h"
#include "common/configuration.h"
#include "common/nixl_log.h"
#include "nixl.h"

namespace {

constexpr int kEtcdKeyNotFound = 100;

bool
isEtcdKeyNotFound(const etcd::Response &response) {
    return response.error_code() == kEtcdKeyNotFound;
}

} // namespace

nixlEtcdMetadataBackend::nixlEtcdMetadataBackend(std::string my_agent_name,
                                                 std::chrono::microseconds watch_timeout)
    : my_agent_name_(std::move(my_agent_name)),
      namespace_prefix_(nixl::config::getValueDefaulted<std::string>("NIXL_ETCD_NAMESPACE",
                                                                     NIXL_ETCD_NAMESPACE_DEFAULT)),
      watch_timeout_(watch_timeout) {
    const auto endpoints = nixl::config::getNonEmptyString("NIXL_ETCD_ENDPOINTS");

    try {
        etcd_ = std::make_unique<etcd::SyncClient>(endpoints);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error creating etcd client: " << e.what();
        return;
    }

    NIXL_DEBUG << "Created etcd client to endpoints: " << endpoints;
    NIXL_DEBUG << "Using etcd namespace for agents: " << namespace_prefix_;

    const std::string agent_prefix = legacyKey(my_agent_name_, "");
    etcd::Response response = etcd_->put(agent_prefix, "");
    if (!response.is_ok()) {
        throw std::runtime_error("Failed to store agent " + my_agent_name_ +
                                 " prefix key in etcd: " + response.error_message());
    }

    healthy_ = true;
}

nixlEtcdMetadataBackend::~nixlEtcdMetadataBackend() = default;

bool
nixlEtcdMetadataBackend::isHealthy() const noexcept {
    return healthy_;
}

nixl_status_t
nixlEtcdMetadataBackend::watch(const std::string &prefix, nixl_md_watch_cb_t cb) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        auto watcher = std::make_unique<etcd::Watcher>(
            *etcd_, prefix, [cb = std::move(cb)](etcd::Response response) {
                if (!response.is_ok()) {
                    NIXL_ERROR << "ETCD watch failed: " << response.error_message();
                    return;
                }
                for (const auto &event : response.events()) {
                    const nixl_watch_event_t event_type =
                        event.event_type() == etcd::Event::EventType::DELETE_ ?
                        nixl_watch_event_t::DELETE :
                        nixl_watch_event_t::PUT;
                    cb(event.kv().key(), event_type, event.kv().as_string());
                }
            });
        generic_watchers_.push_back(std::move(watcher));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error creating etcd watcher for prefix " << prefix << ": " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::fetchBatch(const std::vector<std::string> &keys,
                                    std::vector<nixl_blob_t> &out,
                                    std::vector<nixl_status_t> &per_key_status) {
    out.assign(keys.size(), nixl_blob_t{});
    per_key_status.assign(keys.size(), NIXL_SUCCESS);
    nixl_status_t worst = NIXL_SUCCESS;
    for (size_t i = 0; i < keys.size(); ++i) {
        const nixl_status_t s = fetch(keys[i], out[i]);
        per_key_status[i] = s;
        if (s != NIXL_SUCCESS && s != NIXL_ERR_NOT_FOUND) {
            worst = s;
        }
    }
    return worst;
}

std::string
nixlEtcdMetadataBackend::legacyKey(const std::string &agent_name,
                                   const std::string &metadata_type) const {
    // Agent-name-key compatibility preserves the historical double slash when
    // `namespace_prefix_` already ends with `/`.
    std::stringstream ss;
    ss << namespace_prefix_ << "/" << agent_name << "/" << metadata_type;
    return ss.str();
}

std::string
nixlEtcdMetadataBackend::uuidBackedKey(const std::string &src_uuid,
                                       const std::string &dst_name_or_null,
                                       const std::string &metadata_label) const {
    // UUID-backed keys trim a trailing slash off `namespace_prefix_` so they
    // use one separator between the namespace and `agents`.
    std::string ns = namespace_prefix_;
    if (!ns.empty() && ns.back() == '/') {
        ns.pop_back();
    }

    std::stringstream ss;
    ss << ns << "/agents/";
    if (!metadata_label.empty()) {
        ss << metadata_label << "/";
    }
    ss << src_uuid << "/" << dst_name_or_null;
    return ss.str();
}

nixl_status_t
nixlEtcdMetadataBackend::publish(const std::string &key, const nixl_blob_t &value) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->put(key, value);
        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully stored key: " << key << " (rev "
                       << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        }
        NIXL_ERROR << "Failed to store key " << key << " in etcd: " << response.error_message();
        return NIXL_ERR_BACKEND;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error publishing key " << key << " to etcd: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::fetch(const std::string &key, nixl_blob_t &value) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->get(key);
        if (response.is_ok()) {
            value = response.value().as_string();
            NIXL_DEBUG << "Successfully fetched key: " << key << " (rev "
                       << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        }
        NIXL_INFO << "Failed to fetch key: " << key << " from etcd: " << response.error_message();
        return isEtcdKeyNotFound(response) ? NIXL_ERR_NOT_FOUND : NIXL_ERR_BACKEND;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error fetching key: " << key << " from etcd: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::remove(const std::string &key) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->rmdir(key, /*recursive=*/true);
        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully removed " << response.values().size()
                       << " etcd keys at: " << key;
            return NIXL_SUCCESS;
        }
        NIXL_ERROR << "Warning: Failed to remove etcd keys at " << key << " : "
                   << response.error_message();
        return NIXL_ERR_BACKEND;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Exception removing etcd keys at " << key << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::fetchOrWait(const std::string &agent_name,
                                     const std::string &metadata_label,
                                     nixl_blob_t &value) {
    const std::string key = legacyKey(agent_name, metadata_label);
    const nixl_status_t direct = fetch(key, value);
    if (direct == NIXL_SUCCESS) {
        return NIXL_SUCCESS;
    }
    if (direct != NIXL_ERR_NOT_FOUND) {
        return direct;
    }

    NIXL_DEBUG << "Metadata not found, setting up watch for: " << key;

    try {
        etcd::Response response = etcd_->get(key);
        if (response.is_ok()) {
            value = response.value().as_string();
            NIXL_DEBUG << "Successfully fetched key before watch: " << key;
            return NIXL_SUCCESS;
        }

        const int64_t watch_index = response.index();
        std::promise<nixl_status_t> ret_prom;
        auto future = ret_prom.get_future();
        std::atomic<bool> promise_set{false};

        auto watcher_callback = [&](etcd::Response r) -> void {
            if (promise_set.exchange(true)) {
                NIXL_DEBUG << "Ignoring subsequent watch event for key: " << key;
                return;
            }
            if (!r.is_ok()) {
                NIXL_ERROR << "Watch failed for key: " << key << " : " << r.error_message();
                ret_prom.set_value(NIXL_ERR_BACKEND);
                return;
            }
            if (r.action() == "delete") {
                NIXL_ERROR << "Watch response: metadata key deleted: " << key;
                ret_prom.set_value(NIXL_ERR_INVALID_PARAM);
                return;
            }
            value = r.value().as_string();
            NIXL_DEBUG << "Watch response: metadata key fetched: " << key;
            ret_prom.set_value(NIXL_SUCCESS);
        };

        auto watcher = etcd::Watcher(*etcd_, key, watch_index, watcher_callback);

        const auto status = future.wait_for(watch_timeout_);
        if (status == std::future_status::timeout) {
            NIXL_ERROR << "Watch timed out for key: " << key;
            watcher.Cancel();
            return NIXL_ERR_BACKEND;
        }
        watcher.Cancel();
        return future.get();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error watching etcd for key: " << key << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlEtcdMetadataBackend::setupAgentInvalWatcher(const std::string &agent_name) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return;
    }
    if (agent_watchers_.find(agent_name) != agent_watchers_.end()) {
        return;
    }

    auto process_response = [this, agent_name](etcd::Response response) -> void {
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
            std::lock_guard<std::mutex> lock(invalidated_mutex_);
            invalidated_agents_.push_back(agent_name);
        } else {
            NIXL_ERROR << "Watcher for " << event.kv().key()
                       << " received unexpected event from etcd: "
                       << static_cast<int>(event.event_type());
        }
    };

    const std::string agent_prefix = legacyKey(agent_name, "");
    agent_watchers_[agent_name] =
        std::make_unique<etcd::Watcher>(*etcd_, agent_prefix, process_response);
}

void
nixlEtcdMetadataBackend::processInvalidatedAgents(nixlAgent &agent) {
    std::vector<std::string> tmp_invalidated;
    {
        std::lock_guard<std::mutex> lock(invalidated_mutex_);
        tmp_invalidated = std::move(invalidated_agents_);
    }
    for (const auto &agent_name : tmp_invalidated) {
        NIXL_DEBUG << "Invalidated agent: " << agent_name;
        agent_watchers_.erase(agent_name);
        const nixl_status_t ret = agent.invalidateRemoteMD(agent_name);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to invalidate remote metadata for agent: " << agent_name << ": "
                       << ret;
        } else {
            NIXL_DEBUG << "Successfully invalidated remote metadata for agent: " << agent_name;
        }
    }
}

#endif // HAVE_ETCD
