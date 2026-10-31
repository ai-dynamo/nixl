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
    worker_ = std::thread(&nixlWqsSession::run, this);
}

nixlWqsSession::~nixlWqsSession() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_one();
    worker_.join();
}

nixl_status_t
nixlWqsSession::submit(const std::shared_ptr<nixlWqsOwner> &owner, std::function<void()> work) {
    const std::lock_guard lock(mutex_);
    const std::lock_guard owner_lock(owner->mutex);
    if (stopping_ || owner->closing) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (inflight_ >= config_.maxInflight) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    queue_.push_back(wqsTask{owner, std::move(work)});
    ++inflight_;
    ++owner->pending;
    ready_.notify_one();
    return NIXL_SUCCESS;
}

void
nixlWqsSession::run() {
    for (;;) {
        wqsTask task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            task.work();
        }
        catch (const std::exception &error) {
            NIXL_ERROR << "WQS worker failed: " << error.what();
        }
        catch (...) {
            NIXL_ERROR << "WQS worker failed with an unknown exception";
        }
        task.work = {};
        {
            const std::lock_guard lock(mutex_);
            --inflight_;
        }
        {
            const std::lock_guard lock(task.owner->mutex);
            --task.owner->pending;
            task.owner->idle.notify_all();
        }
    }
}
