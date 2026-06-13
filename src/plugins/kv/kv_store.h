/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file kv_store.h
 * @brief Shared minimal key-value storage interface for NIXL KV backends.
 *
 * Implementations (e.g. INMEMKV example plugin, REDIS src plugin) provide
 * backend-specific storage while sharing this synchronous put/get/exists API.
 */

#ifndef NIXL_SRC_PLUGINS_KV_KV_STORE_H
#define NIXL_SRC_PLUGINS_KV_KV_STORE_H

#include "nixl_types.h"
#include <cstddef>
#include <cstdint>
#include <string_view>

/**
 * @brief Minimal key-value storage interface for KV-style NIXL backends.
 *
 * This interface intentionally keeps only the smallest synchronous API
 * needed by current KV backend flows (INMEMKV, REDIS, etc.).
 */
class iKVStore {
public:
    virtual ~iKVStore() = default;

    /**
     * @brief Stores value bytes for key.
     *
     * @param key Storage key.
     * @param data Pointer to value bytes to store.
     * @param len Number of bytes to copy from data.
     * @return NIXL_SUCCESS on success, or a backend error code on failure.
     */
    virtual nixl_status_t
    put(std::string_view key, const uint8_t *data, size_t len) = 0;

    /**
     * @brief Reads value for key into buffer.
     *
     * @param key Storage key.
     * @param buffer Output buffer for stored value bytes.
     * @param len Maximum number of bytes to read into buffer.
     * @param bytes_read Set to min(stored_len, len) on success.
     * @return NIXL_SUCCESS on success, or NIXL_ERR_NOT_FOUND when key does not exist.
     */
    virtual nixl_status_t
    get(std::string_view key, uint8_t *buffer, size_t len, size_t &bytes_read) const = 0;

    /**
     * @brief Checks whether key is present in the store.
     *
     * @param key Storage key.
     * @return true if key exists, false otherwise.
     */
    virtual bool
    exists(std::string_view key) const = 0;
};

#endif // NIXL_SRC_PLUGINS_KV_KV_STORE_H
