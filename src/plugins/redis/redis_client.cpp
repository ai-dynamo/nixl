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

#include "redis_client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
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

int
getRedisDB(const nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("db") > 0) {
        try {
            return std::stoi(custom_params->at("db"));
        }
        catch (const std::exception &) {
            NIXL_WARN << "Invalid db value, using default 0";
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

void
setPromiseStatus(const std::shared_ptr<std::promise<nixl_status_t>> &promise, bool success) {
    if (!promise) {
        return;
    }
    promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
}

} // namespace

hiredisAsyncClient::hiredisAsyncClient(RedisConfig config)
    : config_(std::move(config)),
      asyncContext_(nullptr),
      syncContext_(nullptr),
      eventBase_(nullptr),
      connected_(false),
      initDone_(false),
      initSucceeded_(false) {
    evthread_use_pthreads();

    eventBase_ = event_base_new();
    if (!eventBase_) {
        throw std::runtime_error("Failed to create event base");
    }

    asyncContext_ = redisAsyncConnect(config_.host.c_str(), config_.port);
    if (!asyncContext_ || asyncContext_->err) {
        if (asyncContext_) {
            std::string err_msg =
                absl::StrFormat("Failed to connect to Redis: %s", asyncContext_->errstr);
            redisAsyncFree(asyncContext_);
            event_base_free(eventBase_);
            throw std::runtime_error(err_msg);
        }
        event_base_free(eventBase_);
        throw std::runtime_error("Failed to allocate Redis async context");
    }

    asyncContext_->data = this;

    if (redisLibeventAttach(asyncContext_, eventBase_) != REDIS_OK) {
        std::string err_msg =
            absl::StrFormat("Failed to attach Redis to event base: %s", asyncContext_->errstr);
        redisAsyncFree(asyncContext_);
        event_base_free(eventBase_);
        throw std::runtime_error(err_msg);
    }

    redisAsyncSetConnectCallback(asyncContext_, connectCallback);
    redisAsyncSetDisconnectCallback(asyncContext_, disconnectCallback);

    eventLoopThread_ = std::thread(&hiredisAsyncClient::processEventLoop, this);

    int retries = 100;
    while (!initDone_.load() && retries-- > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!initSucceeded_.load()) {
        stopEventLoop();
        throw std::runtime_error("Failed to connect to Redis within timeout");
    }

    connectSyncContext();

    NIXL_INFO << absl::StrFormat(
        "Connected to Redis at %s:%d (db=%d)", config_.host, config_.port, config_.db);
}

void
hiredisAsyncClient::connectSyncContext() {
    struct timeval timeout = {5, 0};
    syncContext_ = redisConnectWithTimeout(config_.host.c_str(), config_.port, timeout);
    if (!syncContext_ || syncContext_->err) {
        std::string err_msg = syncContext_ ? syncContext_->errstr : "allocation failed";
        if (syncContext_) {
            redisFree(syncContext_);
            syncContext_ = nullptr;
        }
        NIXL_WARN << absl::StrFormat(
            "Sync Redis connection for EXISTS failed (%s:%d): %s; queryMem will return errors",
            config_.host,
            config_.port,
            err_msg);
        return;
    }

    if (!config_.password.empty()) {
        redisReply *reply = config_.username.empty() ?
            static_cast<redisReply *>(
                redisCommand(syncContext_, "AUTH %s", config_.password.c_str())) :
            static_cast<redisReply *>(redisCommand(
                syncContext_, "AUTH %s %s", config_.username.c_str(), config_.password.c_str()));
        if (!checkRedisReplyOk(reply, "AUTH")) {
            freeReplyObject(reply);
            redisFree(syncContext_);
            syncContext_ = nullptr;
            NIXL_WARN << "Sync Redis AUTH failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    if (config_.db != 0) {
        redisReply *reply =
            static_cast<redisReply *>(redisCommand(syncContext_, "SELECT %d", config_.db));
        if (!checkRedisReplyOk(reply, "SELECT")) {
            freeReplyObject(reply);
            redisFree(syncContext_);
            syncContext_ = nullptr;
            NIXL_WARN << "Sync Redis SELECT failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    NIXL_INFO << absl::StrFormat("Sync Redis connection ready for EXISTS at %s:%d (db=%d)",
                                 config_.host,
                                 config_.port,
                                 config_.db);
}

hiredisAsyncClient::~hiredisAsyncClient() {
    stopEventLoop();
    if (syncContext_) {
        redisFree(syncContext_);
        syncContext_ = nullptr;
    }
}

void
hiredisAsyncClient::processEventLoop() {
    event_base_dispatch(eventBase_);
}

bool
hiredisAsyncClient::scheduleOnEventLoop(std::function<void()> task) {
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
hiredisAsyncClient::freeAsyncContextInEventLoop() {
    redisAsyncContext *ctx = asyncContext_;
    asyncContext_ = nullptr;
    if (ctx) {
        redisAsyncFree(ctx);
    }
}

void
hiredisAsyncClient::stopEventLoop() {
    connected_.store(false);

    if (eventBase_ && eventLoopThread_.joinable()) {
        bool scheduled = scheduleOnEventLoop([this]() {
            freeAsyncContextInEventLoop();
            event_base_loopbreak(eventBase_);
        });
        if (!scheduled) {
            event_base_loopbreak(eventBase_);
        }
        eventLoopThread_.join();
    }

    if (eventBase_) {
        event_base_free(eventBase_);
        eventBase_ = nullptr;
    }
    asyncContext_ = nullptr;
}

void
hiredisAsyncClient::completeAsyncInitialization(bool success) {
    connected_.store(success);
    initSucceeded_.store(success);
    initDone_.store(true);
}

void
hiredisAsyncClient::startAsyncSelect() {
    if (config_.db == 0) {
        completeAsyncInitialization(true);
        return;
    }

    int ret = redisAsyncCommand(asyncContext_, selectCallback, this, "SELECT %d", config_.db);
    if (ret != REDIS_OK) {
        NIXL_ERROR << "Failed to queue Redis SELECT command";
        completeAsyncInitialization(false);
        freeAsyncContextInEventLoop();
        event_base_loopbreak(eventBase_);
    }
}

void
hiredisAsyncClient::startAsyncInitialization() {
    if (!config_.password.empty()) {
        const int ret = config_.username.empty() ?
            redisAsyncCommand(
                asyncContext_, authCallback, this, "AUTH %s", config_.password.c_str()) :
            redisAsyncCommand(asyncContext_,
                              authCallback,
                              this,
                              "AUTH %s %s",
                              config_.username.c_str(),
                              config_.password.c_str());
        if (ret != REDIS_OK) {
            NIXL_ERROR << "Failed to queue Redis AUTH command";
            completeAsyncInitialization(false);
            freeAsyncContextInEventLoop();
            event_base_loopbreak(eventBase_);
        }
        return;
    }

    // An empty password means the Redis deployment permits unauthenticated access.
    startAsyncSelect();
}

void
hiredisAsyncClient::connectCallback(const redisAsyncContext *c, int status) {
    auto *client = static_cast<hiredisAsyncClient *>(c->data);
    if (status != REDIS_OK) {
        NIXL_ERROR << absl::StrFormat("Redis connection error: %s", c->errstr);
        client->completeAsyncInitialization(false);
        client->freeAsyncContextInEventLoop();
        event_base_loopbreak(client->eventBase_);
    } else {
        client->startAsyncInitialization();
    }
}

void
hiredisAsyncClient::disconnectCallback(const redisAsyncContext *c, int status) {
    auto *client = static_cast<hiredisAsyncClient *>(c->data);
    if (status != REDIS_OK) {
        NIXL_WARN << absl::StrFormat("Redis disconnection error: %s", c->errstr);
    }
    client->connected_.store(false);
}

void
hiredisAsyncClient::authCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *client = static_cast<hiredisAsyncClient *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    if (!checkRedisReplyOk(r, "AUTH")) {
        client->completeAsyncInitialization(false);
        client->freeAsyncContextInEventLoop();
        event_base_loopbreak(client->eventBase_);
        return;
    }

    client->startAsyncSelect();
}

void
hiredisAsyncClient::selectCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *client = static_cast<hiredisAsyncClient *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    if (!checkRedisReplyOk(r, "SELECT")) {
        client->completeAsyncInitialization(false);
        client->freeAsyncContextInEventLoop();
        event_base_loopbreak(client->eventBase_);
        return;
    }

    client->completeAsyncInitialization(true);
}

void
hiredisAsyncClient::setCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    bool success = false;
    if (r && r->type == REDIS_REPLY_STATUS) {
        success = (strcmp(r->str, "OK") == 0);
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis SET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    delete ctx;
    setPromiseStatus(promise_ptr, success);
}

void
hiredisAsyncClient::getCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    bool success = false;
    if (r && r->type == REDIS_REPLY_STRING) {
        const size_t reply_len = static_cast<size_t>(r->len);
        if (reply_len != ctx->data_len) {
            NIXL_ERROR << absl::StrFormat(
                "Redis GET size mismatch: expected %zu bytes, got %zu bytes",
                ctx->data_len,
                reply_len);
        } else if (ctx->data_len == 0) {
            success = true;
        } else if (ctx->data_ptr) {
            std::memcpy(reinterpret_cast<void *>(ctx->data_ptr), r->str, ctx->data_len);
            success = true;
        }
    } else if (r && r->type == REDIS_REPLY_NIL) {
        NIXL_WARN << "Redis GET: key not found";
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis GET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    delete ctx;
    setPromiseStatus(promise_ptr, success);
}

void
hiredisAsyncClient::putKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (!connected_.load()) {
        setPromiseStatus(promise, false);
        return;
    }

    std::string key_copy(key);
    bool scheduled = scheduleOnEventLoop(
        [this, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            if (!connected_.load() || !asyncContext_) {
                setPromiseStatus(promise, false);
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->promise_ptr = promise;

            int ret = redisAsyncCommand(asyncContext_,
                                        setCallback,
                                        ctx,
                                        "SET %b %b",
                                        key.data(),
                                        key.size(),
                                        reinterpret_cast<const char *>(data_ptr),
                                        data_len);

            if (ret != REDIS_OK) {
                auto promise_ptr = ctx->promise_ptr;
                delete ctx;
                setPromiseStatus(promise_ptr, false);
            }
        });

    if (!scheduled) {
        setPromiseStatus(promise, false);
    }
}

void
hiredisAsyncClient::getKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (!connected_.load()) {
        setPromiseStatus(promise, false);
        return;
    }

    std::string key_copy(key);
    bool scheduled = scheduleOnEventLoop(
        [this, key = std::move(key_copy), data_ptr, data_len, promise]() mutable {
            if (!connected_.load() || !asyncContext_) {
                setPromiseStatus(promise, false);
                return;
            }

            auto *ctx = new CallbackContext;
            ctx->data_ptr = data_ptr;
            ctx->data_len = data_len;
            ctx->promise_ptr = promise;

            int ret = redisAsyncCommand(
                asyncContext_, getCallback, ctx, "GET %b", key.data(), key.size());

            if (ret != REDIS_OK) {
                auto promise_ptr = ctx->promise_ptr;
                delete ctx;
                setPromiseStatus(promise_ptr, false);
            }
        });

    if (!scheduled) {
        setPromiseStatus(promise, false);
    }
}

std::optional<bool>
hiredisAsyncClient::checkKeyExistsSync(std::string_view key) {
    std::lock_guard<std::mutex> lock(syncMutex_);

    if (!syncContext_ || syncContext_->err) {
        NIXL_ERROR << "Sync Redis connection unavailable for EXISTS";
        return std::nullopt;
    }

    redisReply *reply =
        static_cast<redisReply *>(redisCommand(syncContext_, "EXISTS %b", key.data(), key.size()));

    if (!reply) {
        NIXL_ERROR << "Redis EXISTS: no reply";
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

    bool exists = (reply->integer == 1);
    freeReplyObject(reply);
    return exists;
}

#else // HAVE_HIREDIS_ASYNC

hiredisAsyncClient::hiredisAsyncClient(RedisConfig config) {
    throw std::runtime_error("hiredis-async not available");
}

hiredisAsyncClient::~hiredisAsyncClient() {}

void
hiredisAsyncClient::putKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (promise) {
        promise->set_value(NIXL_ERR_BACKEND);
    }
}

void
hiredisAsyncClient::getKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (promise) {
        promise->set_value(NIXL_ERR_BACKEND);
    }
}

std::optional<bool>
hiredisAsyncClient::checkKeyExistsSync(std::string_view key) {
    return std::nullopt;
}

#endif // HAVE_HIREDIS_ASYNC

RedisConnectionPool::RedisConnectionPool(RedisConfig config) {
    clients_.reserve(static_cast<size_t>(config.pool_size));
    for (int i = 0; i < config.pool_size; ++i) {
        clients_.push_back(std::make_unique<hiredisAsyncClient>(config));
    }
}

iRedisClient &
RedisConnectionPool::nextSlot() {
    return *clients_[nextSlot_.fetch_add(1, std::memory_order_relaxed) % clients_.size()];
}

void
RedisConnectionPool::putKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    nextSlot().putKeyAsync(key, data_ptr, data_len, std::move(promise));
}

void
RedisConnectionPool::getKeyAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 std::shared_ptr<std::promise<nixl_status_t>> promise) {
    nextSlot().getKeyAsync(key, data_ptr, data_len, std::move(promise));
}

std::optional<bool>
RedisConnectionPool::checkKeyExistsSync(std::string_view key) {
    return nextSlot().checkKeyExistsSync(key);
}

size_t
RedisConnectionPool::size() const {
    return clients_.size();
}
