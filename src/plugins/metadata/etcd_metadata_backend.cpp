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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <sstream>
#include <unordered_map>
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

void
cancelWatcher(const std::unique_ptr<etcd::Watcher> &watcher) noexcept {
    if (!watcher) {
        return;
    }

    try {
        watcher->Cancel();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error cancelling etcd watcher: " << e.what();
    }
    catch (...) {
        NIXL_ERROR << "Unknown error cancelling etcd watcher";
    }
}

std::string
rangeEndForInclusiveKey(const std::string &key) {
    // etcd ranges are half-open; appending NUL includes the exact max key.
    std::string range_end = key;
    range_end.push_back('\0');
    return range_end;
}

} // namespace

namespace nixl::metadata {

struct nixlEtcdMetadataBackend::GenericWatcherState {
    explicit GenericWatcherState(nixl_md_watch_cb_t callback)
        : cb(std::move(callback)) {}

    std::atomic<bool> shutting_down{false};
    nixl_md_watch_cb_t cb;
};

struct nixlEtcdMetadataBackend::InvalidationWatcherState {
    std::atomic<bool> shutting_down{false};
    std::mutex invalidated_mutex;
    std::vector<std::string> invalidated_agents;
};

nixlEtcdMetadataBackend::nixlEtcdMetadataBackend(std::string my_agent_name,
                                                 std::chrono::microseconds watch_timeout)
    : my_agent_name_(std::move(my_agent_name)),
      namespace_prefix_(nixl::config::getValueDefaulted<std::string>("NIXL_ETCD_NAMESPACE",
                                                                     NIXL_ETCD_NAMESPACE_DEFAULT)),
      watch_timeout_(watch_timeout),
      invalidation_state_(std::make_shared<InvalidationWatcherState>()) {
    const auto endpoints = nixl::config::getNonEmptyString("NIXL_ETCD_ENDPOINTS");

    etcd_ = std::make_unique<etcd::SyncClient>(endpoints);

    NIXL_DEBUG << "Created etcd client to endpoints: " << endpoints;
    NIXL_DEBUG << "Using etcd namespace for agents: " << namespace_prefix_;

    const std::string agent_prefix = legacyKey(my_agent_name_, "");
    etcd::Response response = etcd_->put(agent_prefix, "");
    if (!response.is_ok()) {
        throw std::runtime_error("Failed to store agent " + my_agent_name_ +
                                 " prefix key in etcd: " + response.error_message());
    }

}

nixlEtcdMetadataBackend::~nixlEtcdMetadataBackend() {
    shutdownWatchers();
}

void
nixlEtcdMetadataBackend::shutdownWatchers() noexcept {
    if (invalidation_state_) {
        invalidation_state_->shutting_down.store(true, std::memory_order_release);
    }
    for (const auto &state : generic_watcher_states_) {
        state->shutting_down.store(true, std::memory_order_release);
    }

    for (const auto &agent_watcher : agent_watchers_) {
        cancelWatcher(agent_watcher.second);
    }
    for (const auto &watcher : generic_watchers_) {
        cancelWatcher(watcher);
    }

    agent_watchers_.clear();
    generic_watchers_.clear();
    generic_watcher_states_.clear();
}

nixl_status_t
nixlEtcdMetadataBackend::watch(const std::string &prefix,
                               nixl_md_watch_cb_t cb) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        auto state = std::make_shared<GenericWatcherState>(std::move(cb));
        std::weak_ptr<GenericWatcherState> weak_state = state;
        auto watcher = std::make_unique<etcd::Watcher>(
            *etcd_, prefix, [weak_state](etcd::Response response) {
                auto state = weak_state.lock();
                if (!state || state->shutting_down.load(std::memory_order_acquire)) {
                    return;
                }
                if (!response.is_ok()) {
                    NIXL_ERROR << "ETCD watch failed: " << response.error_message();
                    return;
                }
                for (const auto &event : response.events()) {
                    if (state->shutting_down.load(std::memory_order_acquire)) {
                        return;
                    }
                    const nixl_watch_event_t event_type =
                        event.event_type() == etcd::Event::EventType::DELETE_ ?
                        nixl_watch_event_t::DELETE :
                        nixl_watch_event_t::PUT;
                    state->cb(event.kv().key(), event_type, event.kv().as_string());
                }
            });
        generic_watchers_.push_back(std::move(watcher));
        generic_watcher_states_.push_back(std::move(state));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error creating etcd watcher for prefix " << prefix << ": " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlEtcdMetadataBackend::fetchBatch(const std::vector<std::string> &keys,
                                    std::vector<nixl_blob_t> &out,
                                    std::vector<nixl_status_t> &per_key_status) {
    out.assign(keys.size(), nixl_blob_t{});
    per_key_status.assign(keys.size(), NIXL_ERR_NOT_FOUND);

    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        std::fill(per_key_status.begin(), per_key_status.end(), NIXL_ERR_NOT_SUPPORTED);
        return;
    }
    if (keys.empty()) {
        return;
    }
    if (keys.size() == 1) {
        per_key_status[0] = fetch(keys[0], out[0]);
        return;
    }

    std::vector<std::string> sorted_keys = keys;
    std::sort(sorted_keys.begin(), sorted_keys.end());
    sorted_keys.erase(std::unique(sorted_keys.begin(), sorted_keys.end()), sorted_keys.end());

    std::unordered_map<std::string, std::vector<std::size_t>> requested;
    requested.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        requested[keys[i]].push_back(i);
    }

    try {
        etcd::Response response =
            etcd_->ls(sorted_keys.front(), rangeEndForInclusiveKey(sorted_keys.back()));
        if (!response.is_ok()) {
            if (isEtcdKeyNotFound(response)) {
                return;
            }
            NIXL_ERROR << "Failed to batch fetch " << keys.size()
                       << " keys from etcd: " << response.error_message();
            std::fill(per_key_status.begin(), per_key_status.end(), NIXL_ERR_BACKEND);
            return;
        }

        for (const auto &value : response.values()) {
            auto it = requested.find(value.key());
            if (it == requested.end()) {
                continue;
            }
            for (const std::size_t index : it->second) {
                out[index] = value.as_string();
                per_key_status[index] = NIXL_SUCCESS;
            }
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error batch fetching " << keys.size() << " keys from etcd: " << e.what();
        std::fill(per_key_status.begin(), per_key_status.end(), NIXL_ERR_UNKNOWN);
    }
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
        struct FetchOrWaitState {
            explicit FetchOrWaitState(std::string watched_key)
                : key(std::move(watched_key)) {}

            std::atomic<bool> done{false};
            std::promise<nixl_status_t> result;
            std::string key;
            nixl_blob_t value;
        };

        auto state = std::make_shared<FetchOrWaitState>(key);
        auto future = state->result.get_future();

        auto watcher_callback = [state](etcd::Response r) -> void {
            if (state->done.exchange(true, std::memory_order_acq_rel)) {
                NIXL_DEBUG << "Ignoring subsequent watch event for key: " << state->key;
                return;
            }
            if (!r.is_ok()) {
                NIXL_ERROR << "Watch failed for key: " << state->key << " : "
                           << r.error_message();
                state->result.set_value(NIXL_ERR_BACKEND);
                return;
            }
            if (r.action() == "delete") {
                NIXL_ERROR << "Watch response: metadata key deleted: " << state->key;
                state->result.set_value(NIXL_ERR_INVALID_PARAM);
                return;
            }
            state->value = r.value().as_string();
            NIXL_DEBUG << "Watch response: metadata key fetched: " << state->key;
            state->result.set_value(NIXL_SUCCESS);
        };

        auto watcher = etcd::Watcher(*etcd_, key, watch_index, watcher_callback);

        const auto status = future.wait_for(watch_timeout_);
        if (status == std::future_status::timeout) {
            NIXL_ERROR << "Watch timed out for key: " << key;
            state->done.store(true, std::memory_order_release);
            watcher.Cancel();
            return NIXL_ERR_BACKEND;
        }
        watcher.Cancel();
        const nixl_status_t ret = future.get();
        if (ret == NIXL_SUCCESS) {
            value = std::move(state->value);
        }
        return ret;
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

    std::weak_ptr<InvalidationWatcherState> weak_state = invalidation_state_;
    auto process_response = [weak_state, agent_name](etcd::Response response) -> void {
        auto state = weak_state.lock();
        if (!state || state->shutting_down.load(std::memory_order_acquire)) {
            return;
        }
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
            std::lock_guard<std::mutex> lock(state->invalidated_mutex);
            state->invalidated_agents.push_back(agent_name);
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
    auto state = invalidation_state_;
    if (!state || state->shutting_down.load(std::memory_order_acquire)) {
        return;
    }

    std::vector<std::string> tmp_invalidated;
    {
        std::lock_guard<std::mutex> lock(state->invalidated_mutex);
        tmp_invalidated = std::move(state->invalidated_agents);
    }
    for (const auto &agent_name : tmp_invalidated) {
        NIXL_DEBUG << "Invalidated agent: " << agent_name;
        const auto watcher = agent_watchers_.find(agent_name);
        if (watcher != agent_watchers_.end()) {
            cancelWatcher(watcher->second);
            agent_watchers_.erase(watcher);
        }
        const nixl_status_t ret = agent.invalidateRemoteMD(agent_name);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to invalidate remote metadata for agent: " << agent_name << ": "
                       << ret;
        } else {
            NIXL_DEBUG << "Successfully invalidated remote metadata for agent: " << agent_name;
        }
    }
}

} // namespace nixl::metadata

#endif // HAVE_ETCD
