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
 * @file tcpstore_metadata_backend.h
 * @brief TCPStore implementation of the nixlMetadataBackend contract.
 *
 * Centralized KV backend over the c10d TCPStore protocol (see
 * tcpstore_client.h). Keys are full and opaque (the manager builds them); this
 * class performs only store I/O. Unlike ETCD there is no native watch: liveness
 * relies on republish/last-writer-wins, so watch() is a no-op (the example EP
 * flow uses set/multi_get, never a subscription).
 */
#ifndef NIXL_SRC_CORE_BACKENDS_TCPSTORE_METADATA_BACKEND_H
#define NIXL_SRC_CORE_BACKENDS_TCPSTORE_METADATA_BACKEND_H

#include "nixl_metadata_backend.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

class nixlTcpStoreClient;

class nixlTcpStoreMetadataBackend : public nixlMetadataBackend {
public:
    // Health gate: reads NIXL_TCPSTORE_ENDPOINTS (host:port), connects, and
    // publishes the agent-anchor key; throws on failure. anchor_key is the full
    // {namespace}/{agent}/ key.
    nixlTcpStoreMetadataBackend(const std::string &anchor_key,
                                const std::chrono::microseconds &timeout);

    ~nixlTcpStoreMetadataBackend() override;

    [[nodiscard]] std::string_view
    name() const override {
        return "TCPStore";
    }

    [[nodiscard]] nixl_status_t
    publish(const std::string &key, const nixl_blob_t &blob) override;

    [[nodiscard]] nixl_status_t
    fetch(const std::string &key, nixl_blob_t &blob) override;

    // Deletes the key and any keys this agent published under it (TCPStore has
    // no recursive delete, so the manager's prefix-style invalidate is emulated
    // over the keys we wrote). Idempotent.
    [[nodiscard]] nixl_status_t
    remove(const std::string &key) override;

    // No-op: TCPStore has no native watch (see file header).
    [[nodiscard]] nixl_status_t
    watch(const std::string &key_prefix, watch_callback_t cb) override;

    [[nodiscard]] bool
    hasWatch() const override {
        return false;
    }

    // Batched read via the store's multi_get; reports per-key presence.
    void
    fetchBatch(const std::vector<std::string> &keys,
               std::vector<nixl_blob_t> &blobs,
               std::vector<nixl_status_t> &per_key_status) override;

private:
    std::unique_ptr<nixlTcpStoreClient> client_;
    std::mutex publishedMutex_;
    // Keys this agent has published, used to emulate prefix removal on remove().
    std::unordered_set<std::string> publishedKeys_;
};

#endif // NIXL_SRC_CORE_BACKENDS_TCPSTORE_METADATA_BACKEND_H
