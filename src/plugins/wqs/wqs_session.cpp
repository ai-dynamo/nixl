// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "wqs_session.h"

#include "common/nixl_log.h"
#include "common/backend.h"

#include <sys/un.h>

#include <climits>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
size_t
parseSize(const nixl_b_params_t &params, const std::string &name, size_t fallback) {
    size_t value;
    try {
        value = nixl::getBackendParamDefaulted(params, name, fallback);
    }
    catch (const std::exception &error) {
        throw std::invalid_argument(error.what());
    }
    if (value == 0) {
        throw std::invalid_argument("WQS parameter must be positive: " + name);
    }
    return value;
}
} // namespace

nixlWqsConfig
nixlWqsConfig::fromParams(const nixl_b_params_t &params) {
    nixlWqsConfig config;
    static const std::set<std::string, std::less<>> keys = {"unix_path",
                                                            "staging_pool_bytes",
                                                            "max_inflight",
                                                            "max_batch_entries",
                                                            "payload_bytes",
                                                            "key_prefix"};
    for (const auto &[name, value] : params) {
        if (!keys.contains(name)) {
            throw std::invalid_argument("Unknown WQS parameter: " + name);
        }
        if (name == "unix_path") {
            config.unixPath = value;
        } else if (name == "key_prefix") {
            config.keyPrefix = value;
        }
    }
    config.stagingPoolBytes = parseSize(params, "staging_pool_bytes", config.stagingPoolBytes);
    config.maxInflight = parseSize(params, "max_inflight", config.maxInflight);
    config.maxBatchEntries = parseSize(params, "max_batch_entries", config.maxBatchEntries);
    config.payloadBytes = parseSize(params, "payload_bytes", config.payloadBytes);
    if (config.unixPath.empty() || config.unixPath.front() != '/' ||
        config.unixPath.find('\0') != std::string::npos) {
        throw std::invalid_argument("WQS unix_path must be an absolute path without NUL bytes");
    }
    if (config.unixPath.size() >= sizeof(sockaddr_un{}.sun_path)) {
        throw std::invalid_argument("WQS unix_path exceeds the Unix socket path limit");
    }
    if (config.stagingPoolBytes % (size_t{2} << 20) != 0) {
        throw std::invalid_argument("WQS staging_pool_bytes must be a multiple of 2 MiB");
    }
    if (config.maxBatchEntries > INT_MAX) {
        throw std::invalid_argument("WQS max_batch_entries must not exceed INT_MAX");
    }
    if (config.payloadBytes > INT_MAX || config.payloadBytes < 64) {
        throw std::invalid_argument("WQS payload_bytes must be between 64 and INT_MAX");
    }
    return config;
}

bool
nixlWqsConfig::sameSession(const nixlWqsConfig &other) const {
    return unixPath == other.unixPath && stagingPoolBytes == other.stagingPoolBytes &&
        maxInflight == other.maxInflight && maxBatchEntries == other.maxBatchEntries &&
        payloadBytes == other.payloadBytes;
}

nixlWqsSession::nixlWqsSession(nixlWqsConfig config,
                               std::shared_ptr<iWqsClient> client,
                               std::shared_ptr<nixlWqsBufferPool> pool)
    : config_(std::move(config)),
      client_(std::move(client)),
      pool_(std::move(pool)) {
    if (!client_ || !pool_ || pool_->size() != config_.stagingPoolBytes) {
        throw std::invalid_argument("Invalid WQS session resources");
    }
}

nixlWqsOwner::nixlWqsOwner(const nixlBackendInitParams &params, size_t max_inflight)
    : progressThread(params.enableProgTh),
      syncMode(params.enableProgTh && params.syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE ?
                   nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT :
                   params.syncMode),
      mutex(syncMode),
      maxInflight_(max_inflight),
      delay_(params.pthrDelay) {
    if (progressThread) {
        worker_ = std::thread(&nixlWqsOwner::run, this);
    }
}

nixlWqsOwner::~nixlWqsOwner() {
    stop();
}

void
nixlWqsOwner::stop() {
    {
        std::unique_lock lock(mutex);
        closing = true;
        ready_.notify_one();
        idle.wait(lock, [this] { return pending == 0; });
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

nixl_status_t
nixlWqsOwner::submit(std::function<nixl_status_t()> work,
                     std::function<void(nixl_status_t)> complete) {
    wqsTask task{std::move(work), std::move(complete)};
    {
        const std::lock_guard lock(mutex);
        if (closing || inflight_ >= maxInflight_) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (progressThread) {
            queue_.push_back(std::move(task));
        }
        ++inflight_;
        ++pending;
    }
    if (progressThread) {
        ready_.notify_one();
    } else {
        execute(std::move(task));
    }
    return NIXL_SUCCESS;
}

void
nixlWqsOwner::execute(wqsTask task) {
    nixl_status_t status = NIXL_ERR_BACKEND;
    try {
        status = task.work();
    }
    catch (const std::exception &error) {
        NIXL_ERROR << "WQS worker failed: " << error.what();
    }
    catch (...) {
        NIXL_ERROR << "WQS worker failed with an unknown exception";
    }
    task.work = {};
    {
        const std::lock_guard lock(mutex);
        --inflight_;
    }
    // A completed handle can be reposted immediately, including at the queue limit.
    task.complete(status);
    const std::lock_guard lock(mutex);
    --pending;
    idle.notify_all();
}

void
nixlWqsOwner::run() {
    for (;;) {
        wqsTask task;
        {
            std::unique_lock lock(mutex);
            const auto ready = [this] { return closing || !queue_.empty(); };
            if (delay_.count() > 0) {
                ready_.wait_for(lock, delay_, ready);
            } else {
                ready_.wait(lock, ready);
            }
            if (queue_.empty()) {
                if (closing) {
                    return;
                }
                continue;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        execute(std::move(task));
    }
}
