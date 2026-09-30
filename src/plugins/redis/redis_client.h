/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H
#define NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H

#include "nixl_types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#ifdef HAVE_HIREDIS_ASYNC
#include <event2/event.h>
#include <event2/thread.h>
#include <hiredis/adapters/libevent.h>
#include <hiredis/async.h>
#include <hiredis/hiredis.h>
#else
struct event_base;
struct redisAsyncContext;
struct redisContext;
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

/** Hiredis implementation with async SET/GET and a separate synchronous EXISTS connection. */
class hiredisAsyncClient : public iRedisClient {
public:
    explicit hiredisAsyncClient(RedisConfig config);
    ~hiredisAsyncClient() override;

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

    int
    inFlight() const {
        return inFlight_.load(std::memory_order_relaxed);
    }

    bool
    connected() const {
        return connected_.load(std::memory_order_relaxed);
    }

private:
    struct CallbackContext {
        uintptr_t data_ptr;
        size_t data_len;
        std::shared_ptr<std::promise<nixl_status_t>> promise_ptr;
        std::atomic<int> *inFlight;
    };

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

    void
    processEventLoop();
    void
    connectSyncContext();
    void
    startAsyncInitialization();
    void
    startAsyncSelect();
    void
    completeAsyncInitialization(bool success);
    void
    freeAsyncContextInEventLoop();
    bool
    scheduleOnEventLoop(std::function<void()> task);
    void
    stopEventLoop();

    RedisConfig config_;
    redisAsyncContext *asyncContext_;
    redisContext *syncContext_;
    struct event_base *eventBase_;
    std::thread eventLoopThread_;
    std::atomic<bool> connected_;
    std::atomic<bool> initDone_;
    std::atomic<bool> initSucceeded_;
    std::atomic<int> inFlight_{0};
    mutable std::mutex syncMutex_;
};

/**
 * Least-loaded connection pool over multiple hiredisAsyncClient slots.
 * Each slot has its own event loop thread and TCP connections. Dispatch routes
 * to the healthy slot with the fewest in-flight async commands; unhealthy
 * (disconnected) slots are skipped and commands fail immediately when no
 * healthy slot is available.
 */
class RedisConnectionPool : public iRedisClient {
public:
    explicit RedisConnectionPool(RedisConfig config);
    ~RedisConnectionPool() override = default;

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
    hiredisAsyncClient *
    leastLoadedHealthySlot();

    std::vector<std::unique_ptr<hiredisAsyncClient>> clients_;
};

#endif // NIXL_SRC_PLUGINS_REDIS_REDIS_CLIENT_H
