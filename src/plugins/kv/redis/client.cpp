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

#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef HAVE_HIREDIS_ASYNC

namespace {

std::string
getRedisHost(nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("host") > 0) {
        return custom_params->at("host");
    }
    const char *env_host = getenv("REDIS_HOST");
    return env_host ? std::string(env_host) : "localhost";
}

int
getRedisPort(nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("port") > 0) {
        try {
            return std::stoi(custom_params->at("port"));
        }
        catch (const std::exception &) {
            NIXL_WARN << "Invalid port value, using default 6379";
        }
    }
    const char *env_port = getenv("REDIS_PORT");
    if (env_port) {
        try {
            return std::stoi(env_port);
        }
        catch (const std::exception &) {
            NIXL_WARN << "Invalid REDIS_PORT, using default 6379";
        }
    }
    return 6379;
}

std::string
getRedisPassword(nixl_b_params_t *custom_params) {
    if (custom_params && custom_params->count("password") > 0) {
        return custom_params->at("password");
    }
    const char *env_password = getenv("REDIS_PASSWORD");
    return env_password ? std::string(env_password) : "";
}

int
getRedisDB(nixl_b_params_t *custom_params) {
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
setPromiseStatus(const std::shared_ptr<asioThreadPoolExecutor> &executor,
                 const std::shared_ptr<std::promise<nixl_status_t>> &promise,
                 bool success) {
    if (!promise) {
        return;
    }
    if (executor) {
        executor->post([promise, success]() {
            promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
        });
    } else {
        promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
    }
}

} // namespace

hiredisAsyncClient::hiredisAsyncClient(nixl_b_params_t *custom_params,
                                       std::shared_ptr<asioThreadPoolExecutor> executor)
    : host_(getRedisHost(custom_params)),
      port_(getRedisPort(custom_params)),
      password_(getRedisPassword(custom_params)),
      db_(getRedisDB(custom_params)),
      asyncContext_(nullptr),
      syncContext_(nullptr),
      eventBase_(nullptr),
      executor_(executor),
      connected_(false) {
    evthread_use_pthreads();

    eventBase_ = event_base_new();
    if (!eventBase_) {
        throw std::runtime_error("Failed to create event base");
    }

    asyncContext_ = redisAsyncConnect(host_.c_str(), port_);
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

    if (!password_.empty()) {
        redisAsyncCommand(asyncContext_, nullptr, nullptr, "AUTH %s", password_.c_str());
    }
    if (db_ != 0) {
        redisAsyncCommand(asyncContext_, nullptr, nullptr, "SELECT %d", db_);
    }

    eventLoopThread_ = std::thread(&hiredisAsyncClient::processEventLoop, this);

    int retries = 100;
    while (!connected_ && retries-- > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!connected_) {
        redisAsyncDisconnect(asyncContext_);
        event_base_loopbreak(eventBase_);
        if (eventLoopThread_.joinable()) {
            eventLoopThread_.join();
        }
        redisAsyncFree(asyncContext_);
        event_base_free(eventBase_);
        throw std::runtime_error("Failed to connect to Redis within timeout");
    }

    connectSyncContext();

    NIXL_INFO << absl::StrFormat("Connected to Redis at %s:%d (db=%d)", host_, port_, db_);
}

void
hiredisAsyncClient::connectSyncContext() {
    struct timeval timeout = {5, 0};
    syncContext_ = redisConnectWithTimeout(host_.c_str(), port_, timeout);
    if (!syncContext_ || syncContext_->err) {
        std::string err_msg = syncContext_ ? syncContext_->errstr : "allocation failed";
        if (syncContext_) {
            redisFree(syncContext_);
            syncContext_ = nullptr;
        }
        NIXL_WARN << absl::StrFormat(
            "Sync Redis connection for EXISTS failed (%s:%d): %s; queryMem will return errors",
            host_,
            port_,
            err_msg);
        return;
    }

    if (!password_.empty()) {
        redisReply *reply =
            static_cast<redisReply *>(redisCommand(syncContext_, "AUTH %s", password_.c_str()));
        if (!checkRedisReplyOk(reply, "AUTH")) {
            freeReplyObject(reply);
            redisFree(syncContext_);
            syncContext_ = nullptr;
            NIXL_WARN << "Sync Redis AUTH failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    if (db_ != 0) {
        redisReply *reply =
            static_cast<redisReply *>(redisCommand(syncContext_, "SELECT %d", db_));
        if (!checkRedisReplyOk(reply, "SELECT")) {
            freeReplyObject(reply);
            redisFree(syncContext_);
            syncContext_ = nullptr;
            NIXL_WARN << "Sync Redis SELECT failed; queryMem will return errors";
            return;
        }
        freeReplyObject(reply);
    }

    NIXL_INFO << absl::StrFormat(
        "Sync Redis connection ready for EXISTS at %s:%d (db=%d)", host_, port_, db_);
}

hiredisAsyncClient::~hiredisAsyncClient() {
    if (asyncContext_) {
        redisAsyncDisconnect(asyncContext_);
    }
    if (eventBase_) {
        event_base_loopbreak(eventBase_);
        if (eventLoopThread_.joinable()) {
            eventLoopThread_.join();
        }
        event_base_free(eventBase_);
    }
    if (asyncContext_) {
        redisAsyncFree(asyncContext_);
    }
    if (syncContext_) {
        redisFree(syncContext_);
    }
}

void
hiredisAsyncClient::setExecutor(std::shared_ptr<asioThreadPoolExecutor> executor) {
    executor_ = executor;
}

void
hiredisAsyncClient::processEventLoop() {
    event_base_dispatch(eventBase_);
}

void
hiredisAsyncClient::connectCallback(const redisAsyncContext *c, int status) {
    auto *client = static_cast<hiredisAsyncClient *>(c->data);
    if (status != REDIS_OK) {
        NIXL_ERROR << absl::StrFormat("Redis connection error: %s", c->errstr);
        client->connected_ = false;
    } else {
        client->connected_ = true;
    }
}

void
hiredisAsyncClient::disconnectCallback(const redisAsyncContext *c, int status) {
    auto *client = static_cast<hiredisAsyncClient *>(c->data);
    if (status != REDIS_OK) {
        NIXL_WARN << absl::StrFormat("Redis disconnection error: %s", c->errstr);
    }
    client->connected_ = false;
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
    auto executor = ctx->executor;
    delete ctx;
    setPromiseStatus(executor, promise_ptr, success);
}

void
hiredisAsyncClient::getCallback(redisAsyncContext *c, void *reply, void *privdata) {
    auto *ctx = static_cast<CallbackContext *>(privdata);
    auto *r = static_cast<redisReply *>(reply);

    bool success = false;
    if (r && r->type == REDIS_REPLY_STRING) {
        size_t copy_len = std::min(static_cast<size_t>(r->len), ctx->data_len);
        if (ctx->data_ptr && copy_len > 0) {
            std::memcpy(reinterpret_cast<void *>(ctx->data_ptr), r->str, copy_len);
            success = true;
        }
    } else if (r && r->type == REDIS_REPLY_NIL) {
        NIXL_WARN << "Redis GET: key not found";
    } else if (r && r->type == REDIS_REPLY_ERROR) {
        NIXL_ERROR << absl::StrFormat("Redis GET error: %s", r->str);
    }

    auto promise_ptr = ctx->promise_ptr;
    auto executor = ctx->executor;
    delete ctx;
    setPromiseStatus(executor, promise_ptr, success);
}

void
hiredisAsyncClient::putKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (!connected_) {
        setPromiseStatus(executor_, promise, false);
        return;
    }

    auto *ctx = new CallbackContext;
    ctx->executor = executor_;
    ctx->promise_ptr = std::move(promise);

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
        setPromiseStatus(executor_, promise_ptr, false);
    }
}

void
hiredisAsyncClient::getKeyAsync(std::string_view key,
                                uintptr_t data_ptr,
                                size_t data_len,
                                std::shared_ptr<std::promise<nixl_status_t>> promise) {
    if (!connected_) {
        setPromiseStatus(executor_, promise, false);
        return;
    }

    auto *ctx = new CallbackContext;
    ctx->executor = executor_;
    ctx->data_ptr = data_ptr;
    ctx->data_len = data_len;
    ctx->promise_ptr = std::move(promise);

    int ret = redisAsyncCommand(
        asyncContext_, getCallback, ctx, "GET %b", key.data(), key.size());

    if (ret != REDIS_OK) {
        auto promise_ptr = ctx->promise_ptr;
        delete ctx;
        setPromiseStatus(executor_, promise_ptr, false);
    }
}

std::optional<bool>
hiredisAsyncClient::checkKeyExistsSync(std::string_view key) {
    std::lock_guard<std::mutex> lock(syncMutex_);

    if (!syncContext_ || syncContext_->err) {
        NIXL_ERROR << "Sync Redis connection unavailable for EXISTS";
        return std::nullopt;
    }

    redisReply *reply = static_cast<redisReply *>(
        redisCommand(syncContext_, "EXISTS %b", key.data(), key.size()));

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

hiredisAsyncClient::hiredisAsyncClient(nixl_b_params_t *custom_params,
                                       std::shared_ptr<asioThreadPoolExecutor> executor) {
    throw std::runtime_error("hiredis-async not available");
}

hiredisAsyncClient::~hiredisAsyncClient() {}

void
hiredisAsyncClient::setExecutor(std::shared_ptr<asioThreadPoolExecutor> executor) {}

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
