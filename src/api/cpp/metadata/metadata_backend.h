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
 * @file metadata_backend.h
 * @brief Southbound contract for metadata transports used by nixlMetadataManager.
 *
 * The interface is intentionally key/value-shaped. ETCD fits naturally, and
 * the P2P backend models destinations as keys after the manager resolves agent
 * names to peer addresses.
 */
#ifndef NIXL_SRC_API_CPP_METADATA_METADATA_BACKEND_H
#define NIXL_SRC_API_CPP_METADATA_METADATA_BACKEND_H

#include "nixl_md.h"
#include "nixl_types.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Watch callback signature.
 *
 * Invoked by the backend when a key under a watched prefix changes.
 * `value` is empty for `nixl_watch_event_t::DELETE`.
 */
using nixl_md_watch_cb_t =
    std::function<void(const std::string &key, nixl_watch_event_t event, const nixl_blob_t &value)>;

/**
 * @class nixlMetadataBackend
 * @brief Abstract metadata transport.
 *
 * Lifetime: created and owned by `nixlMetadataManager`. All methods may be
 * called from the manager tick thread; implementations must be safe under that
 * single-threaded caller plus any backend-internal threads they own (watchers,
 * keep-alive threads).
 */
class nixlMetadataBackend {
public:
    virtual ~nixlMetadataBackend() = default;

    /**
     * @brief Store @p value at @p key.
     */
    [[nodiscard]] virtual nixl_status_t
    publish(const std::string &key, const nixl_blob_t &value) = 0;

    /**
     * @brief Read the current value at @p key into @p value.
     *
     * Returns `NIXL_ERR_NOT_FOUND` when the key is absent.
     */
    [[nodiscard]] virtual nixl_status_t
    fetch(const std::string &key, nixl_blob_t &value) = 0;

    /**
     * @brief Remove @p key (or all keys under @p key when the backend treats
     *        keys as prefix-friendly, e.g. ETCD `rmdir`-style removal).
     */
    [[nodiscard]] virtual nixl_status_t
    remove(const std::string &key) = 0;

    /**
     * @brief Subscribe to changes under @p prefix.
     *
     * Implementations that don't support watching return
     * `NIXL_ERR_NOT_SUPPORTED`.
     */
    [[nodiscard]] virtual nixl_status_t
    watch(const std::string &prefix, nixl_md_watch_cb_t cb) = 0;

    /**
     * @brief Liveness check for the backend itself (transport reachable, etc.).
     */
    [[nodiscard]] virtual bool
    isHealthy() const noexcept = 0;

    /**
     * @brief Bulk fetch.
     *
     * @p out has the same size as @p keys; entries for missing keys are left
     * as the provided default (empty string) and the per-key status is
     * returned in @p per_key_status (also same size as @p keys).
     */
    [[nodiscard]] virtual nixl_status_t
    fetchBatch(const std::vector<std::string> &keys,
               std::vector<nixl_blob_t> &out,
               std::vector<nixl_status_t> &per_key_status) = 0;
};

#endif // NIXL_SRC_API_CPP_METADATA_METADATA_BACKEND_H
