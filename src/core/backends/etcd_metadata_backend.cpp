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

#include "etcd_metadata_backend.h"

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <atomic>
#include <future>
#include <stdexcept>

nixlEtcdMetadataBackend::nixlEtcdMetadataBackend(const std::string &anchor_key,
                                                 const std::chrono::microseconds &watch_timeout)
    : watchTimeout_(watch_timeout) {
    const auto etcd_endpoints = nixl::config::getNonEmptyString("NIXL_ETCD_ENDPOINTS");

    try {
        etcd_ = std::make_unique<etcd::SyncClient>(etcd_endpoints);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error creating etcd client: " << e.what();
        throw;
    }
    NIXL_DEBUG << "Created etcd client to endpoints: " << etcd_endpoints;

    etcd::Response response = etcd_->put(anchor_key, "");
    if (!response.is_ok()) {
        throw std::runtime_error("Failed to store agent prefix key " + anchor_key +
                                 " in etcd: " + response.error_message());
    }
}

nixl_status_t
nixlEtcdMetadataBackend::publish(const std::string &key, const nixl_blob_t &blob) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->put(key, blob);

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully stored key: " << key << " (rev "
                       << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Failed to store key " << key << " in etcd: " << response.error_message();
            return NIXL_ERR_BACKEND;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error sending key " << key << " to etcd: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::fetch(const std::string &key, nixl_blob_t &blob) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->get(key);

        if (response.is_ok()) {
            blob = response.value().as_string();
            NIXL_DEBUG << "Successfully fetched key: " << key << " (rev "
                       << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        } else {
            NIXL_INFO << "Failed to fetch key: " << key
                      << " from etcd: " << response.error_message();
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error fetching key: " << key << " from etcd: " << e.what();
    }

    NIXL_DEBUG << "Metadata not found, setting up watch for: " << key;
    return waitForKey(key, blob);
}

nixl_status_t
nixlEtcdMetadataBackend::waitForKey(const std::string &key, nixl_blob_t &blob) {
    try {
        // Get current index to watch from
        etcd::Response response = etcd_->get(key);
        int64_t watch_index = response.index();
        std::promise<nixl_status_t> ret_prom;
        auto future = ret_prom.get_future();
        std::atomic<bool> promise_set{false};

        // This lambda assumes lifetime only inside this method
        auto watcher_callback = [&](etcd::Response response) -> void {
            if (promise_set.exchange(true)) {
                NIXL_DEBUG << "Ignoring subsequent watch event for key: " << key;
                return;
            }

            if (!response.is_ok()) {
                NIXL_ERROR << "Watch failed for key: " << key << " : " << response.error_message();
                ret_prom.set_value(NIXL_ERR_BACKEND);
                return;
            }
            if (response.action() == "delete") {
                // Key is absent; the fetch contract reports this as NOT_FOUND.
                NIXL_DEBUG << "Watch response: metadata key deleted: " << key;
                ret_prom.set_value(NIXL_ERR_NOT_FOUND);
                return;
            }
            blob = response.value().as_string();
            NIXL_DEBUG << "Watch response: metadata key fetched: " << key;
            ret_prom.set_value(NIXL_SUCCESS);
        };

        auto watcher = etcd::Watcher(*etcd_, key, watch_index, watcher_callback);

        auto status = future.wait_for(watchTimeout_);
        if (status == std::future_status::timeout) {
            // Key never appeared within the timeout; still absent -> NOT_FOUND
            // per the fetch contract.
            NIXL_DEBUG << "Watch timed out for key: " << key;
            return NIXL_ERR_NOT_FOUND;
        }
        watcher.Cancel();
        return future.get();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error watching etcd for key: " << key << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::remove(const std::string &key) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        etcd::Response response = etcd_->rmdir(key, true);

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully removed " << response.values().size()
                       << " etcd keys for prefix: " << key;
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Warning: Failed to remove etcd keys for prefix: " << key << " : "
                       << response.error_message();
            return NIXL_ERR_BACKEND;
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Exception removing etcd keys for prefix: " << key << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlEtcdMetadataBackend::watch(const std::string &key_prefix, watch_callback_t cb) {
    if (!etcd_) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Guard the find-then-insert so concurrent watch() calls for the same
    // prefix don't both create a watcher (and to keep watchers_ consistent).
    const std::lock_guard<std::mutex> lk(watchersMutex_);
    if (watchers_.find(key_prefix) != watchers_.end()) {
        return NIXL_SUCCESS;
    }

    // The callback runs on the etcd watcher thread; it cannot call back into
    // the agent directly, so the manager-supplied cb only enqueues events.
    auto process_response = [cb, key_prefix](etcd::Response response) -> void {
        if (!response.is_ok()) {
            NIXL_ERROR << "Watcher failed to watch prefix " << key_prefix
                       << " from etcd: " << response.error_message();
            return;
        }
        NIXL_DEBUG << "Watcher received " << response.events().size() << " events from etcd";
        for (const auto &event : response.events()) {
            if (event.event_type() == etcd::Event::EventType::DELETE_) {
                NIXL_DEBUG << "Watcher DELETE: " << event.kv().key() << " (rev "
                           << event.kv().modified_index() << ")";
                cb(event.kv().key(), nixl_watch_event_t::DELETE, nixl_blob_t{});
            } else if (event.event_type() == etcd::Event::EventType::PUT) {
                cb(event.kv().key(), nixl_watch_event_t::PUT, event.kv().as_string());
            } else {
                NIXL_ERROR << "Watcher for " << event.kv().key()
                           << " received unexpected event from etcd: "
                           << static_cast<int>(event.event_type());
            }
        }
    };

    try {
        watchers_[key_prefix] =
            std::make_unique<etcd::Watcher>(*etcd_, key_prefix, process_response);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error creating watcher for prefix: " << key_prefix << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

#endif // HAVE_ETCD
