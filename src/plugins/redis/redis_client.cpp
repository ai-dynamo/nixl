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

#include "redis_client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

std::string
getStringSetting(const nixl_b_params_t *custom_params,
                 const char *param_name,
                 const char *env_name,
                 const char *default_value = "") {
    if (custom_params) {
        const auto it = custom_params->find(param_name);
        if (it != custom_params->end()) {
            return it->second;
        }
    }
    const char *env_value = std::getenv(env_name);
    return env_value ? std::string(env_value) : default_value;
}

std::optional<int>
parseRedisPort(const std::string &value, const char *source) {
    try {
        size_t parsed = 0;
        int port = std::stoi(value, &parsed);
        if (parsed != value.size() || port <= 0 || port > 65535) {
            NIXL_WARN << absl::StrFormat(
                "Invalid %s value '%s', using default 6379", source, value);
            return std::nullopt;
        }
        return port;
    }
    catch (const std::exception &) {
        NIXL_WARN << absl::StrFormat("Invalid %s value '%s', using default 6379", source, value);
        return std::nullopt;
    }
}

int
getRedisPort(const nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("port") > 0) {
        auto port = parseRedisPort(custom_params->at("port"), "Redis port");
        if (port) {
            return *port;
        }
    }
    const char *env_port = std::getenv("REDIS_PORT");
    if (env_port) {
        auto port = parseRedisPort(env_port, "REDIS_PORT");
        if (port) {
            return *port;
        }
    }
    return 6379;
}

std::optional<int>
parseRedisDB(const std::string &value, const char *source) {
    try {
        size_t parsed = 0;
        int val = std::stoi(value, &parsed);
        if (parsed != value.size() || val < 0) {
            NIXL_WARN << absl::StrFormat("Invalid %s value '%s', using default 0", source, value);
            return std::nullopt;
        }
        return val;
    }
    catch (const std::exception &) {
        NIXL_WARN << absl::StrFormat("Invalid %s value '%s', using default 0", source, value);
        return std::nullopt;
    }
}

int
getRedisDB(const nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("db") > 0) {
        auto val = parseRedisDB(custom_params->at("db"), "db");
        if (val) {
            return *val;
        }
    }
    const char *env_val = std::getenv("REDIS_DB");
    if (env_val) {
        auto val = parseRedisDB(env_val, "REDIS_DB");
        if (val) {
            return *val;
        }
    }
    return 0;
}

std::optional<int>
parseRedisPoolSize(const std::string &value, const char *source) {
    try {
        size_t parsed = 0;
        int val = std::stoi(value, &parsed);
        if (parsed != value.size() || val <= 0) {
            NIXL_WARN << absl::StrFormat("Invalid %s value '%s', using default 8", source, value);
            return std::nullopt;
        }
        return val;
    }
    catch (const std::exception &) {
        NIXL_WARN << absl::StrFormat("Invalid %s value '%s', using default 8", source, value);
        return std::nullopt;
    }
}

int
getRedisPoolSize(const nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("pool_size") > 0) {
        auto val = parseRedisPoolSize(custom_params->at("pool_size"), "pool_size");
        if (val) {
            return *val;
        }
    }
    const char *env_val = std::getenv("REDIS_POOL_SIZE");
    if (env_val) {
        auto val = parseRedisPoolSize(env_val, "REDIS_POOL_SIZE");
        if (val) {
            return *val;
        }
    }
    return 8;
}

} // namespace

RedisConfig
RedisConfig::fromBackendParams(const nixl_b_params_t *custom_params) {
    RedisConfig config;
    config.host = getStringSetting(custom_params, "host", "REDIS_HOST", "localhost");
    config.port = getRedisPort(custom_params);
    config.username = getStringSetting(custom_params, "username", "REDIS_USERNAME");
    config.password = getStringSetting(custom_params, "password", "REDIS_PASSWORD");
    config.db = getRedisDB(custom_params);
    config.pool_size = getRedisPoolSize(custom_params);
    if (!config.username.empty() && config.password.empty()) {
        throw std::invalid_argument("Redis username requires a password");
    }
    return config;
}

#ifdef HAVE_HIREDIS_ASYNC

namespace {

using redis_event_task_t = std::function<void()>;

void
runEventTask(evutil_socket_t, short, void *arg) {
    std::unique_ptr<redis_event_task_t> task(static_cast<redis_event_task_t *>(arg));
    (*task)();
}

bool
checkRedisReplyOk(redisReply *reply, const char *command) {
    if (!reply) {
        NIXL_ERROR << absl::StrFormat("Redis %s: no reply", command);
        return false;
    }
    if (reply->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis %s error: %s", command, reply->str);
        return false;
    }
    if (reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") != 0) {
        NIXL_ERROR << absl::StrFormat("Redis %s unexpected status: %s", command, reply->str);
        return false;
    }
    return true;
}

std::once_flag evthreadOnce;

struct CallbackContext {
    uintptr_t data_ptr;
    size_t data_len;
    std::shared_ptr<std::promise<nixl_status_t>> promise_ptr;
    std::atomic<int> *inFlight;
};

} // namespace

// Slot: one async TCP connection on the shared event loop.
struct RedisConnectionPool::Slot {
    RedisConnectionPool *pool = nullptr;
    redisAsyncContext *asyncCtx = nullptr;
    std::atomic<bool> connected{false};
    std::atomic<bool> initDone{false};
    std::atomic<bool> initSucceeded{false};
    std::atomic<int> inFlight{0};
    // asyncCtx is freed by RedisConnectionPool::stopEventLoop on the event loop thread
};

RedisConnectionPool::RedisConnectionPool(RedisConfig config) : config_(std::move(config)) {
    std::call_once(evthreadOnce, [] {
        if (evthread_use_pthreads() != 0) {
            throw std::runtime_error("evthread_use_pthreads() failed");
        }
    });

    eventBase_ = event_base_new();
    if (!eventBase_) {
        throw std::runtime_error("Failed to create event base");
    }

    const int N = config_.pool_size;
    slots_.reserve(N);
    for (int i = 0; i < N; ++i) {
        slots_.push_back(std::make_unique<Slot>());
        slots_.back()->pool = this;
    }

    // Attach all async connections to the shared event base before starting the thread.
    // On error, free whatever was created before re-throwing.
    try {
        for (auto &s : slots_) {
            initSlotAsyncCtx(*s);
        }
    }
    catch (...) {
        for (auto &s : slots_) {
            if (s->asyncCtx) {
                redisAsyncFree(s->asyncCtx);
                s->asyncCtx = nullptr;
            }
        }
        event_base_free(eventBase_);
        eventBase_ = nullptr;
        throw;
    }

    // Start the single shared event loop thread.
    // All N async connect/auth/select sequences run concurrently on this one thread.
    eventLoopThread_ = std::thread([this]() { event_base_dispatch(eventBase_); });

    // Wait for every slot to complete initialization (parallel: ~1 RTT regardless of N).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    for (auto &s : slots_) {
        while (!s->initDone.load()) {
            if (std::chrono::steady_clock::now() > deadline) {
                stopEventLoop();
                throw std::runtime_error("Redis connection pool initialization timed out");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const bool allOk =
        std::all_of(slots_.begin(), slots_.end(), [](const std::unique_ptr<Slot> &s) {
            return s->initSucceeded.load();
        });
    if (!allOk) {
        stopEventLoop();
        throw std::runtime_error("Failed to initialize one or more Redis connections in pool");
    }

    // Shared sync connection for EXISTS (queryMem is single-threaded, one connection suffices).
    connectSyncContext();

    // Worker threads perform the GET reply memcpy off the event loop thread so that
    // large transfers do not stall other pending callbacks.
    workers_.reserve(N);
    for (int i = 0; i < N; ++i) {
        workers_.emplace_back([this]() { workerLoop(); });
    }

    NIXL_INFO << absl::StrFormat(
        "Redis connection pool ready: %d connections at %s:%d (db=%d, workers=%d)",
        N,
        config_.host,
        config_.port,
        config_.db,
        N);
}

RedisConnectionPool::~RedisConnectionPool() {
    stopEventLoop(); // no new callbacks after this; no new work enters the queue
    {
        std::lock_guard<std::mutex> lock(workMutex_);
        stopWorkers_.store(true);
    }
    workCv_.notify_all();
    for (auto &w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    if (syncCtx_) {
        redisFree(syncCtx_);
        syncCtx_ = nullptr;
    }
}

void
RedisConnectionPool::initSlotAsyncCtx(Slot &slot) {
    slot.asyncCtx = redisAsyncConnect(config_.host.c_str(), config_.port);
    if (!slot.asyncCtx || slot.asyncCtx->err) {
        std::string msg;
        if (slot.asyncCtx) {
            msg = absl::StrFormat("Failed to connect to Redis: %s", slot.asyncCtx->errstr);
            redisAsyncFree(slot.asyncCtx);
            slot.asyncCtx = nullptr;
        } else {
            msg = "Failed to allocate Redis async context";
        }
        throw std::runtime_error(msg);
    }

    slot.asyncCtx->data = &slot;

    if (redisLibeventAttach(slot.asyncCtx, eventBase_) != REDIS_OK) {
        std::string msg =
            absl::StrFormat("Failed to attach Redis to event base: %s", slot.asyncCtx->errstr);
        redisAsyncFree(slot.asyncCtx);
        slot.asyncCtx = nullptr;
        throw std::runtime_error(msg);
    }

    redisAsyncSetConnectCallback(slot.asyncCtx, connectCallback);
    redisAsyncSetDisconnectCallback(slot.asyncCtx, disconnectCallback);
}

void
RedisConnectionPool::connectSyncContext() {
    struct timeval timeout = {5, 0};
    syncCtx_ = redisConnectWithTimeout(config_.host.c_str(), config_.port, timeout);
    if (!syncCtx_ || syncCtx_->err) {
        std::string err_msg = syncCtx_ ? syncCtx_->errstr : "allocation failed";
        if (syncCtx_) {
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
        }
        NIXL_WARN << absl::StrFormat(
            "Sync Redis connection failed (%s:%d): %s; queryMem will return errors",
            config_.host,
            config_.port,
            err_msg);
        return;
    }

    if (!config_.password.empty()) {
        redisReply *reply = config_.username.empty() ?
            static_cast<redisReply *>(redisCommand(syncCtx_, "AUTH %s", config_.password.c_str())) :
            static_cast<redisReply *>(redisCommand(
                syncCtx_, "AUTH %s %s", config_.username.c_str(), config_.password.c_str()));
        if (!checkRedisReplyOk(reply, "AUTH")) {
            freeReplyObject(reply);
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
            NIXL_WARN << "Sync Redis AUTH failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    if (config_.db != 0) {
        redisReply *reply =
            static_cast<redisReply *>(redisCommand(syncCtx_, "SELECT %d", config_.db));
        if (!checkRedisReplyOk(reply, "SELECT")) {
            freeReplyObject(reply);
            redisFree(syncCtx_);
            syncCtx_ = nullptr;
            NIXL_WARN << "Sync Redis SELECT failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }
}

void
RedisConnectionPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(workMutex_);
            workCv_.wait(lock, [this] { return stopWorkers_.load() || !workQueue_.empty(); });
            if (stopWorkers_.load() && workQueue_.empty()) {
                return;
            }
            task = std::move(workQueue_.front());
            workQueue_.pop();
        }
        task();
    }
}

void
RedisConnectionPool::postToWorker(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(workMutex_);
        workQueue_.push(std::move(task));
    }
    workCv_.notify_one();
}

bool
RedisConnectionPool::scheduleOnEventLoop(std::function<void()> task) {
    if (!eventBase_) {
        return false;
    }
    auto *owned_task = new redis_event_task_t(std::move(task));
    timeval immediate = {0, 0};
    if (event_base_once(eventBase_, -1, EV_TIMEOUT, runEventTask, owned_task, &immediate) != 0) {
        delete owned_task;
        return false;
    }
    return true;
}

void
RedisConnectionPool::stopEventLoop() {
    if (!eventBase_) {
        return;
    }
    if (eventLoopThread_.joinable()) {
        bool scheduled = scheduleOnEventLoop([this]() {
            for (auto &s : slots_) {
                if (s->asyncCtx) {
                    s->connected.store(false);
                    redisAsyncFree(s->asyncCtx);
                    s->asyncCtx = nullptr;
                }
            }
            event_base_loopbreak(eventBase_);
        });
        if (!scheduled) {
            event_base_loopbreak(eventBase_);
        }
        eventLoopThread_.join();
    }
    event_base_free(eventBase_);
    eventBase_ = nullptr;
}

void
RedisConnectionPool::freeSlotAsyncCtx(Slot &slot) {
    if (slot.asyncCtx) {
        redisAsyncFree(slot.asyncCtx);
        slot.asyncCtx = nullptr;
    }
}

void
RedisConnectionPool::completeSlotInit(Slot &slot, bool success) {
    slot.connected.store(success);
    slot.initSucceeded.store(success);
    slot.initDone.store(true);
}

void
RedisConnectionPool::startSlotSelect(Slot &slot) {
    if (config_.db == 0) {
        completeSlotInit(slot, true);
        return;
    }
    int ret = redisAsyncCommand(slot.asyncCtx, selectCallback, &slot, "SELECT %d", config_.db);
    if (ret != REDIS_OK) {
        NIXL_ERROR << "Failed to queue Redis SELECT command";
        completeSlotInit(slot, false);
        freeSlotAsyncCtx(slot);
    }
}

void
RedisConnectionPool::startSlotAuth(Slot &slot) {
    if (config_.password.empty()) {
        startSlotSelect(slot);
        return;
    }
    const int ret = config_.username.empty() ?
        redisAsyncCommand(slot.asyncCtx, authCallback, &slot, "AUTH %s", config_.password.c_str()) :
        redisAsyncCommand(slot.asyncCtx,
                          authCallback,
                          &slot,
                          "AUTH %s %s",
                          config_.username.c_str(),
                          config_.password.c_str());
    if (ret != REDIS_OK) {
        NIXL_ERROR << "Failed to queue Redis AUTH command";
        completeSlotInit(slot, false);
        freeSlotAsyncCtx(slot);
    }
}

void
RedisConnectionPool::connectCallback(const redisAsyncContext *c, int status) {
    auto *slot = static_cast<Slot *>(c->data);
    if (status != REDIS_OK) {
        NIXL_ERROR << absl::StrFormat("Redis connection error: %s", c->errstr);
        // hiredis frees c after this callback returns (REDIS_CONNECTED was never set,
        // so disconnectCallback is NOT called). Null asyncCtx now so stopEventLoop
        // does not call redisAsyncFree on already-freed memory.
        slot->asyncCtx = nullptr;
        slot->pool->completeSlotInit(*slot, false);
    } else {
        slot->pool->startSlotAuth(*slot);
    }
}

void
RedisConnectionPool::disconnectCallback(const redisAsyncContext *c, int status) {
    auto *slot = static_cast<Slot *>(c->data);
    if (status != REDIS_OK) {
        NIXL_WARN << absl::StrFormat("Redis disconnected with error: %s", c->errstr);
    }
    // hiredis frees c after this callback returns. Null asyncCtx now so stopEventLoop
    // does not call redisAsyncFree on already-freed memory (double-free).
    slot->asyncCtx = nullptr;
    slot->connected.store(false);
}

void
RedisConnectionPool::authCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *slot = static_cast<Slot *>(privdata);
    auto *r = static_cast<redisReply *>(reply);
    if (!checkRedisReplyOk(r, "AUTH")) {
        slot->pool->completeSlotInit(*slot, false);
        slot->pool->freeSlotAsyncCtx(*slot);
        return;
    }
    slot->pool->startSlotSelect(*slot);
}

void
RedisConnectionPool::selectCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *slot = static_cast<Slot *>(privdata);
    auto *r = static_cast<redisReply *>(reply);
    if (!checkRedisReplyOk(r, "SELECT")) {
        slot->pool->completeSlotInit(*slot, false);
        slot->pool->freeSlotAsyncCtx(*slot);
        return;
    }
    slot->pool->completeSlotInit(*slot, true);
}

void
RedisConnectionPool::setCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    bool success = false;
    if (r && r->type == REDIS_REPLY_STATUS) {
        success = (strcmp(r->str, "OK") == 0);
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis SET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    auto *inFlight = ctx->inFlight;
    delete ctx;
    if (inFlight) {
        inFlight->fetch_sub(1, std::memory_order_relaxed);
    }
    if (promise_ptr) {
        promise_ptr->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
    }
}

void
RedisConnectionPool::getCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    if (r && r->type == REDIS_REPLY_STRING) {
        const size_t reply_len = static_cast<size_t>(r->len);
        const bool size_ok = (reply_len == ctx->data_len);

        if (size_ok && ctx->data_len > 0 && ctx->data_ptr) {
            // Steal ownership of the reply buffer so the memcpy runs on a worker thread
            // instead of blocking the shared event loop. hiredis skips free() when
            // reply->str is null, so this is safe with the default hiredis allocator.
            auto *slot = static_cast<Slot *>(c->data);
            char *str = r->str;
            r->str = nullptr;
            uintptr_t dst = ctx->data_ptr;
            size_t len = ctx->data_len;
            auto promise = ctx->promise_ptr;
            auto *inFlight = ctx->inFlight;
            delete ctx;
            slot->pool->postToWorker([str, dst, len, promise, inFlight]() {
                std::memcpy(reinterpret_cast<void *>(dst), str, len);
                free(str);
                if (inFlight) {
                    inFlight->fetch_sub(1, std::memory_order_relaxed);
                }
                if (promise) {
                    promise->set_value(NIXL_SUCCESS);
                }
            });
            return;
        }

        const bool null_dst = (ctx->data_len > 0 && !ctx->data_ptr);
        if (!size_ok) {
            NIXL_ERROR << absl::StrFormat(
                "Redis GET size mismatch: expected %zu bytes, got %zu bytes",
                ctx->data_len,
                reply_len);
        } else if (null_dst) {
            NIXL_ERROR << "Redis GET: data_ptr is null with non-zero data_len";
        }
        // zero-length success: resolve on event loop (no copy needed)
        auto promise_ptr = ctx->promise_ptr;
        auto *inFlight = ctx->inFlight;
        delete ctx;
        if (inFlight) {
            inFlight->fetch_sub(1, std::memory_order_relaxed);
        }
        if (promise_ptr) {
            promise_ptr->set_value((size_ok && !null_dst) ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        }
        return;
    }

    if (r && r->type == REDIS_REPLY_NIL) {
        NIXL_WARN << "Redis GET: key not found";
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis GET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    auto *inFlight = ctx->inFlight;
    delete ctx;
    if (inFlight) {
        inFlight->fetch_sub(1, std::memory_order_relaxed);
    }
    if (promise_ptr) {
        promise_ptr->set_value(NIXL_ERR_BACKEND);
    }
}

RedisConnectionPool::Slot *
RedisConnectionPool::leastLoadedHealthySlot() {
    Slot *best = nullptr;
    for (auto &s : slots_) {
        if (!s->connected.load()) {
            continue;
        }
        if (!best ||
            s->inFlight.load(std::memory_order_relaxed) <
                best->inFlight.load(std::memory_order_relaxed)) {
            best = s.get();
        }
    }
    return best;
}

void
RedisConnectionPool::putKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    auto *slot = leastLoadedHealthySlot();
    if (!slot) {
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
        return;
    }

    slot->inFlight.fetch_add(1, std::memory_order_relaxed);

    std::string key_copy(key);
    const bool scheduled = scheduleOnEventLoop(
        [slot, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            if (!slot->connected.load() || !slot->asyncCtx) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                if (promise) {
                    promise->set_value(NIXL_ERR_BACKEND);
                }
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->data_ptr = 0;
            ctx->data_len = 0;
            ctx->promise_ptr = promise;
            ctx->inFlight = &slot->inFlight;

            const int ret = redisAsyncCommand(slot->asyncCtx,
                                              setCallback,
                                              ctx,
                                              "SET %b %b",
                                              key.data(),
                                              key.size(),
                                              reinterpret_cast<const char *>(data_ptr),
                                              data_len);
            if (ret != REDIS_OK) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                auto p = ctx->promise_ptr;
                delete ctx;
                if (p) {
                    p->set_value(NIXL_ERR_BACKEND);
                }
            }
        });

    if (!scheduled) {
        slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
    }
}

void
RedisConnectionPool::getKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    auto *slot = leastLoadedHealthySlot();
    if (!slot) {
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
        return;
    }

    slot->inFlight.fetch_add(1, std::memory_order_relaxed);

    std::string key_copy(key);
    const bool scheduled = scheduleOnEventLoop(
        [slot, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            if (!slot->connected.load() || !slot->asyncCtx) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                if (promise) {
                    promise->set_value(NIXL_ERR_BACKEND);
                }
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->data_ptr = data_ptr;
            ctx->data_len = data_len;
            ctx->promise_ptr = promise;
            ctx->inFlight = &slot->inFlight;

            const int ret = redisAsyncCommand(
                slot->asyncCtx, getCallback, ctx, "GET %b", key.data(), key.size());
            if (ret != REDIS_OK) {
                slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
                auto p = ctx->promise_ptr;
                delete ctx;
                if (p) {
                    p->set_value(NIXL_ERR_BACKEND);
                }
            }
        });

    if (!scheduled) {
        slot->inFlight.fetch_sub(1, std::memory_order_relaxed);
        if (promise) {
            promise->set_value(NIXL_ERR_BACKEND);
        }
    }
}

std::optional<bool>
RedisConnectionPool::checkKeyExistsSync(std::string_view key) {
    std::lock_guard<std::mutex> lock(syncMutex_);
    if (!syncCtx_) {
        NIXL_ERROR << "Sync Redis connection unavailable for EXISTS";
        return std::nullopt;
    }
    if (syncCtx_->err) {
        NIXL_ERROR << "Sync Redis connection unavailable for EXISTS";
        redisFree(syncCtx_);
        syncCtx_ = nullptr;
        return std::nullopt;
    }

    redisReply *reply =
        static_cast<redisReply *>(redisCommand(syncCtx_, "EXISTS %b", key.data(), key.size()));

    if (!reply) {
        NIXL_ERROR << "Redis EXISTS: no reply";
        redisFree(syncCtx_);
        syncCtx_ = nullptr;
        return std::nullopt;
    }
    if (reply->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis EXISTS error: %s", reply->str);
        freeReplyObject(reply);
        return std::nullopt;
    }
    if (reply->type != REDIS_REPLY_INTEGER) {
        NIXL_ERROR << absl::StrFormat("Redis EXISTS unexpected reply type: %d", reply->type);
        freeReplyObject(reply);
        return std::nullopt;
    }

    const bool exists = (reply->integer == 1);
    freeReplyObject(reply);
    return exists;
}

#else // HAVE_HIREDIS_ASYNC

struct RedisConnectionPool::Slot {};

RedisConnectionPool::RedisConnectionPool(RedisConfig) {
    throw std::runtime_error("hiredis-async not available");
}

RedisConnectionPool::~RedisConnectionPool() {}

void
RedisConnectionPool::putKeyAsync(std::string_view,
                                 uintptr_t,
                                 size_t,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (promise) {
        promise->set_value(NIXL_ERR_BACKEND);
    }
}

void
RedisConnectionPool::getKeyAsync(std::string_view,
                                 uintptr_t,
                                 size_t,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (promise) {
        promise->set_value(NIXL_ERR_BACKEND);
    }
}

std::optional<bool>
RedisConnectionPool::checkKeyExistsSync(std::string_view) {
    return std::nullopt;
}

#endif // HAVE_HIREDIS_ASYNC
