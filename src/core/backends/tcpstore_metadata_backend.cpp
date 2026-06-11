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
#include "tcpstore_metadata_backend.h"

#include "tcpstore_client.h"

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <stdexcept>

nixlTcpStoreMetadataBackend::nixlTcpStoreMetadataBackend(const std::string &anchor_key,
                                                         const std::chrono::microseconds &timeout) {
    const std::string endpoints = nixl::config::getNonEmptyString("NIXL_TCPSTORE_ENDPOINTS");

    // NIXL_TCPSTORE_ENDPOINTS is a single host:port (rfind keeps IPv4 hosts and
    // bracketless names simple; the port is the trailing field).
    const auto pos = endpoints.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= endpoints.size()) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINTS must be host:port, got: " + endpoints);
    }
    const std::string host = endpoints.substr(0, pos);
    const std::string port_str = endpoints.substr(pos + 1);
    const unsigned long port = std::stoul(port_str);
    if (port == 0 || port > 65535) {
        throw std::runtime_error("NIXL_TCPSTORE_ENDPOINTS has invalid port: " + port_str);
    }

    client_ = std::make_unique<nixlTcpStoreClient>(
        host,
        static_cast<std::uint16_t>(port),
        std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
    NIXL_DEBUG << "Connected TCPStore client to " << endpoints;

    // Anchor the agent's prefix so its presence is observable in the store.
    client_->set(anchor_key, "");
    publishedKeys_.insert(anchor_key);
}

nixlTcpStoreMetadataBackend::~nixlTcpStoreMetadataBackend() = default;

nixl_status_t
nixlTcpStoreMetadataBackend::publish(const std::string &key, const nixl_blob_t &blob) {
    try {
        client_->set(key, blob);
        {
            const std::lock_guard<std::mutex> lk(publishedMutex_);
            publishedKeys_.insert(key);
        }
        NIXL_DEBUG << "Successfully stored key: " << key;
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error storing key " << key << " in TCPStore: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlTcpStoreMetadataBackend::fetch(const std::string &key, nixl_blob_t &blob) {
    try {
        if (!client_->check(key)) {
            NIXL_DEBUG << "Key not found in TCPStore: " << key;
            return NIXL_ERR_NOT_FOUND;
        }
        blob = client_->get(key);
        NIXL_DEBUG << "Successfully fetched key: " << key;
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error fetching key " << key << " from TCPStore: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlTcpStoreMetadataBackend::remove(const std::string &key) {
    // Emulate prefix removal: delete every key this agent published under the
    // given prefix (TCPStore has no recursive delete), plus the key itself.
    std::vector<std::string> to_delete;
    {
        const std::lock_guard<std::mutex> lk(publishedMutex_);
        for (const auto &published : publishedKeys_) {
            if (published.size() >= key.size() && published.compare(0, key.size(), key) == 0) {
                to_delete.push_back(published);
            }
        }
    }
    if (std::find(to_delete.begin(), to_delete.end(), key) == to_delete.end()) {
        to_delete.push_back(key);
    }

    try {
        for (const auto &k : to_delete) {
            client_->deleteKey(k);
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error removing keys under prefix " << key << " from TCPStore: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    {
        const std::lock_guard<std::mutex> lk(publishedMutex_);
        for (const auto &k : to_delete) {
            publishedKeys_.erase(k);
        }
    }
    NIXL_DEBUG << "Removed " << to_delete.size() << " TCPStore key(s) under prefix: " << key;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpStoreMetadataBackend::watch([[maybe_unused]] const std::string &key_prefix,
                                   [[maybe_unused]] watch_callback_t cb) {
    // TCPStore has no native watch; liveness relies on republish/last-writer-wins.
    return NIXL_SUCCESS;
}

void
nixlTcpStoreMetadataBackend::fetchBatch(const std::vector<std::string> &keys,
                                        std::vector<nixl_blob_t> &blobs,
                                        std::vector<nixl_status_t> &per_key_status) {
    blobs.assign(keys.size(), nixl_blob_t{});
    per_key_status.assign(keys.size(), NIXL_ERR_NOT_FOUND);

    try {
        std::vector<std::string> present;
        std::vector<std::size_t> present_idx;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (client_->check(keys[i])) {
                present.push_back(keys[i]);
                present_idx.push_back(i);
            }
        }
        if (!present.empty()) {
            const std::vector<std::string> values = client_->multiGet(present);
            for (std::size_t j = 0; j < present_idx.size(); ++j) {
                blobs[present_idx[j]] = values[j];
                per_key_status[present_idx[j]] = NIXL_SUCCESS;
            }
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Error in TCPStore batched fetch: " << e.what();
        for (auto &status : per_key_status) {
            if (status != NIXL_SUCCESS) {
                status = NIXL_ERR_BACKEND;
            }
        }
    }
}
