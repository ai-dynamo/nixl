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
/**
 * @file etcd_metadata_backend.h
 * @brief ETCD implementation of the nixlMetadataBackend contract.
 *
 * Extracted from the comm-thread nixlEtcdClient. Keys are full and opaque
 * (the manager builds them); this class only performs etcd I/O.
 */
#ifndef NIXL_SRC_CORE_BACKENDS_ETCD_METADATA_BACKEND_H
#define NIXL_SRC_CORE_BACKENDS_ETCD_METADATA_BACKEND_H

#if HAVE_ETCD

#include "nixl_metadata_backend.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>

class nixlEtcdMetadataBackend : public nixlMetadataBackend {
public:
    // Health gate: connects to NIXL_ETCD_ENDPOINTS and puts the agent-anchor
    // key; throws on failure. anchor_key is the full {namespace}/{agent}/ key.
    nixlEtcdMetadataBackend(const std::string &anchor_key,
                            const std::chrono::microseconds &watch_timeout);

    [[nodiscard]] std::string_view
    name() const override {
        return "ETCD";
    }

    [[nodiscard]] nixl_status_t
    publish(const std::string &key, const nixl_blob_t &blob) override;

    [[nodiscard]] nixl_status_t
    fetch(const std::string &key, nixl_blob_t &blob) override;

    [[nodiscard]] nixl_status_t
    remove(const std::string &key) override;

    [[nodiscard]] nixl_status_t
    watch(const std::string &key_prefix, watch_callback_t cb) override;

private:
    [[nodiscard]] nixl_status_t
    waitForKey(const std::string &key, nixl_blob_t &blob);

    std::unique_ptr<etcd::SyncClient> etcd_;
    std::chrono::microseconds watchTimeout_;
    std::mutex watchersMutex_;
    std::unordered_map<std::string, std::unique_ptr<etcd::Watcher>> watchers_;
};

#endif // HAVE_ETCD

#endif // NIXL_SRC_CORE_BACKENDS_ETCD_METADATA_BACKEND_H
