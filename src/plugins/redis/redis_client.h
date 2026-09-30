/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H
#define NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H

#include "nixl_types.h"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef HAVE_HIREDIS_ASYNC
#include <event2/event.h>
#include <event2/thread.h>
#include <hiredis/adapters/libevent.h>
#include <hiredis/async.h>
#include <hiredis/hiredis.h>
#else
struct event_base;
struct redisAsyncContext;
#endif

/** Resolved Redis connection settings. Explicit backend parameters override environment values. */
struct RedisConfig {
    std::string host = "localhost";
    int port = 6379;
    std::string username;
    std::string password;
    int db = 0;
    int pool_size = 8;

    static RedisConfig
    fromBackendParams(const nixl_b_params_t *custom_params);
};

/** Redis operations used by the NIXL REDIS backend. */
class iRedisClient {
public:
    virtual ~iRedisClient() = default;

    virtual void
    putKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) = 0;

    virtual void
    getKeyAsync(std::string_view key,
                uintptr_t data_ptr,
                size_t data_len,
                std::shared_ptr<std::promise<nixl_status_t>> promise) = 0;

    virtual std::optional<bool>
    checkKeyExistsSync(std::string_view key) = 0;
};

/**
 * Connection pool over N async TCP connections sharing one libevent thread.
 * Dispatch routes to the healthy slot with the fewest in-flight commands;
 * commands fail immediately when no healthy slot is available.
 * Resource cost: 1 OS thread + N async TCP connections + 1 shared sync TCP connection.
 */
class RedisConnectionPool : public iRedisClient {
public:
    explicit RedisConnectionPool(RedisConfig config);
    ~RedisConnectionPool() override;

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
    struct Slot;  // defined in redis_client.cpp

    bool
    scheduleOnEventLoop(std::function<void()> task);
    void
    stopEventLoop();
    Slot *
    leastLoadedHealthySlot();

    void
    initSlotAsyncCtx(Slot &slot);
    void
    startSlotAuth(Slot &slot);
    void
    startSlotSelect(Slot &slot);
    void
    completeSlotInit(Slot &slot, bool success);
    void
    freeSlotAsyncCtx(Slot &slot);
    void
    connectSyncContext();

    static void
    connectCallback(const redisAsyncContext *c, int status);
    static void
    disconnectCallback(const redisAsyncContext *c, int status);
    static void
    authCallback(redisAsyncContext *c, void *reply, void *privdata);
    static void
    selectCallback(redisAsyncContext *c, void *reply, void *privdata);
    static void
    setCallback(redisAsyncContext *c, void *reply, void *privdata);
    static void
    getCallback(redisAsyncContext *c, void *reply, void *privdata);

    event_base         *eventBase_ = nullptr;
    std::thread         eventLoopThread_;
    std::vector<std::unique_ptr<Slot>> slots_;
    redisContext       *syncCtx_ = nullptr;
    mutable std::mutex  syncMutex_;
    RedisConfig         config_;
};

#endif // NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H
