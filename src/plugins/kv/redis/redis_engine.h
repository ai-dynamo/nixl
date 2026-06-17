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
 * @file redis_engine.h
 * @brief REDIS KV plugin — iRedisClient interface and nixlRedisKVEngine entry.
 *
 */

#ifndef NIXL_SRC_PLUGINS_KV_REDIS_ENGINE_H
#define NIXL_SRC_PLUGINS_KV_REDIS_ENGINE_H

#include "../kv_backend.h"
#include "redis_executor.h"
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string_view>

/**
 * @class iRedisClient
 * @brief Redis client for the REDIS KV backend.
 *
 * SET/GET are async (std::promise) for postXfer/checkXfer.
 * EXISTS uses a separate synchronous hiredis connection for queryMem().
 */
class iRedisClient {
public:
    virtual ~iRedisClient() = default;

    /**
     * @brief Set the executor used to complete async promises.
     * @param executor ASIO thread pool shared with the engine implementation.
     */
    virtual void
    setExecutor(std::shared_ptr<asioThreadPoolExecutor> executor) = 0;

    /**
     * @brief Asynchronously store a value under key (Redis SET).
     * @param key Redis key.
     * @param data_ptr Pointer to value bytes.
     * @param data_len Number of bytes to write.
     * @param promise Promise set to NIXL_SUCCESS or NIXL_ERR_BACKEND on completion.
     */
    virtual void
    putKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) = 0;

    /**
     * @brief Asynchronously read a value into buffer (Redis GET).
     * @param key Redis key.
     * @param data_ptr Output buffer pointer.
     * @param data_len Maximum number of bytes to read.
     * @param promise Promise set to NIXL_SUCCESS or NIXL_ERR_BACKEND on completion.
     */
    virtual void
    getKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) = 0;

    /**
     * @brief Synchronously check whether key exists (Redis EXISTS).
     *
     * Uses a dedicated blocking hiredis connection, independent of the async
     * SET/GET path. Intended for queryMem() only.
     *
     * @param key Redis key.
     * @return true if key exists, false if not found, std::nullopt on error.
     */
    virtual std::optional<bool>
    checkKeyExistsSync(std::string_view key) = 0;
};

/**
 * @class nixlRedisKVEngine
 * @brief REDIS backend engine registered with the NIXL plugin loader.
 */
class nixlRedisKVEngine : public nixlKVEngine {
public:
    explicit nixlRedisKVEngine(const nixlBackendInitParams *init_params);

    ~nixlRedisKVEngine() override = default;
};

#endif // NIXL_SRC_PLUGINS_KV_REDIS_ENGINE_H
