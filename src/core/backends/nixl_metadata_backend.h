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
 * @file nixl_metadata_backend.h
 * @brief Southbound storage contract for nixlMDManager backends.
 *
 * Core-internal: not part of the installed public headers, so backend
 * dependencies (etcd, absl) never leak into the public API. Keys are opaque
 * to the backend; the manager owns the key format.
 */
#ifndef NIXL_SRC_CORE_BACKENDS_NIXL_METADATA_BACKEND_H
#define NIXL_SRC_CORE_BACKENDS_NIXL_METADATA_BACKEND_H

#include "nixl_types.h"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

enum class nixl_watch_event_t {
    PUT,
    DELETE,
};

// blob is valid only for PUT events.
using watch_callback_t =
    std::function<void(const std::string &key, nixl_watch_event_t event, const nixl_blob_t &blob)>;

/**
 * @class nixlMetadataBackend
 * @brief Key/value storage primitives the manager dispatches metadata ops through.
 *
 * Construction is the health gate: a backend that cannot reach its transport
 * throws from its constructor. Keys are opaque; the manager builds them.
 */
class nixlMetadataBackend {
public:
    virtual ~nixlMetadataBackend() = default;

    // Stable transport name reported by nixlMDManager::getBackend().
    [[nodiscard]] virtual std::string_view
    name() const = 0;

    // Upsert (last-writer-wins).
    [[nodiscard]] virtual nixl_status_t
    publish(const std::string &key, const nixl_blob_t &blob) = 0;

    // Returns NIXL_ERR_NOT_FOUND when the key is absent.
    [[nodiscard]] virtual nixl_status_t
    fetch(const std::string &key, nixl_blob_t &blob) = 0;

    // Idempotent.
    [[nodiscard]] virtual nixl_status_t
    remove(const std::string &key) = 0;

    // Delivers PUT/DELETE under a key prefix.
    [[nodiscard]] virtual nixl_status_t
    watch(const std::string &key_prefix, watch_callback_t cb) = 0;

    // Whether watch() actually delivers events. Watch-less backends (e.g.
    // TCPStore) return false so the manager can skip invalidation bookkeeping.
    [[nodiscard]] virtual bool
    hasWatch() const {
        return true;
    }

    // Native batched fetch; base falls back to per-key fetch and reports each
    // key's result in per_key_status.
    virtual void
    fetchBatch(const std::vector<std::string> &keys,
               std::vector<nixl_blob_t> &blobs,
               std::vector<nixl_status_t> &per_key_status) {
        blobs.resize(keys.size());
        per_key_status.resize(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            per_key_status[i] = fetch(keys[i], blobs[i]);
        }
    }
};

#endif // NIXL_SRC_CORE_BACKENDS_NIXL_METADATA_BACKEND_H
