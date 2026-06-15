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
 * @file client.h
 * @brief hiredis Redis client implementing iRedisClient.
 *
 * Two connections:
 *   - async (hiredis-async + libevent): SET/GET for postXfer
 *   - sync  (blocking hiredis):         EXISTS for queryMem
 */

#ifndef KV_PLUGIN_REDIS_CLIENT_H
#define KV_PLUGIN_REDIS_CLIENT_H

#include "redis_engine.h"
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include "nixl_types.h"

#ifdef HAVE_HIREDIS_ASYNC
#include <hiredis/async.h>
#include <hiredis/hiredis.h>
#include <hiredis/adapters/libevent.h>
#include <event2/event.h>
#include <event2/thread.h>
#endif

/**
 * @class hiredisAsyncClient
 * @brief iRedisClient with async SET/GET and a separate sync EXISTS connection.
 */
class hiredisAsyncClient : public iRedisClient {
public:
    /**
     * @brief Construct a client from NIXL custom params and environment variables.
     * @param custom_params Optional host/port/password/db overrides.
     * @param executor Executor used to complete async promises.
     */
    hiredisAsyncClient(nixl_b_params_t *custom_params,
                       std::shared_ptr<asioThreadPoolExecutor> executor = nullptr);

    ~hiredisAsyncClient() override;

    void
    setExecutor(std::shared_ptr<asioThreadPoolExecutor> executor) override;

    void
    putKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override;

    void
    getKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) override;

    std::optional<bool>
    checkKeyExistsSync(std::string_view key) override;

private:
    struct CallbackContext {
        std::shared_ptr<asioThreadPoolExecutor> executor;
        uintptr_t data_ptr;
        size_t data_len;
        std::shared_ptr<std::promise<nixl_status_t>> promise_ptr;
    };

    static void
    connectCallback(const redisAsyncContext *c, int status);

    static void
    disconnectCallback(const redisAsyncContext *c, int status);

    static void
    setCallback(redisAsyncContext *c, void *reply, void *privdata);

    static void
    getCallback(redisAsyncContext *c, void *reply, void *privdata);

    void
    processEventLoop();

    void
    connectSyncContext();

    std::string host_;
    int port_;
    std::string password_;
    int db_;
    redisAsyncContext *asyncContext_;
    redisContext *syncContext_;
    struct event_base *eventBase_;
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::thread eventLoopThread_;
    bool connected_;
    mutable std::mutex syncMutex_;
};

#endif // KV_PLUGIN_REDIS_CLIENT_H
