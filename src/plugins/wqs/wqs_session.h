// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H
#define NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H

#include "wqs_buffer_pool.h"
#include "wqs_client.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

struct nixlWqsOwner {
    std::mutex mutex;
    std::condition_variable idle;
    size_t pending = 0;
    bool closing = false;
};

/** A bounded worker queue sharing one client and staging pool across engines. */
class nixlWqsSession {
public:
    nixlWqsSession(nixlWqsConfig config,
                   std::shared_ptr<iWqsClient> client,
                   std::shared_ptr<nixlWqsBufferPool> pool);
    ~nixlWqsSession();
    nixlWqsSession(const nixlWqsSession &) = delete;
    nixlWqsSession &
    operator=(const nixlWqsSession &) = delete;

    [[nodiscard]] nixl_status_t
    submit(const std::shared_ptr<nixlWqsOwner> &owner, std::function<void()> work);

    [[nodiscard]] const nixlWqsConfig &
    config() const noexcept {
        return config_;
    }

    [[nodiscard]] iWqsClient &
    client() noexcept {
        return *client_;
    }

    [[nodiscard]] nixlWqsBufferPool &
    pool() noexcept {
        return *pool_;
    }

private:
    struct wqsTask {
        std::shared_ptr<nixlWqsOwner> owner;
        std::function<void()> work;
    };

    void
    run();

    nixlWqsConfig config_;
    std::shared_ptr<iWqsClient> client_;
    std::shared_ptr<nixlWqsBufferPool> pool_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<wqsTask> queue_;
    size_t inflight_ = 0;
    bool stopping_ = false;
    std::thread worker_;
};

#endif // NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H
