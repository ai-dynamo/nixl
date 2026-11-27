// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H
#define NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H

#include "wqs_buffer_pool.h"
#include "wqs_client.h"
#include "backend/backend_aux.h"
#include "sync.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

struct nixlWqsOwner {
    nixlWqsOwner(const nixlBackendInitParams &params, size_t max_inflight);
    ~nixlWqsOwner();

    [[nodiscard]] nixl_status_t
    submit(std::function<nixl_status_t()> work, std::function<void(nixl_status_t)> complete);
    void
    stop();

    const bool progressThread;
    const nixl_thread_sync_t syncMode;
    nixlLock mutex;
    std::condition_variable_any idle;
    size_t pending = 0;
    bool closing = false;

private:
    struct wqsTask {
        std::function<nixl_status_t()> work;
        std::function<void(nixl_status_t)> complete;
    };

    void
    run();
    void
    execute(wqsTask task);

    const size_t maxInflight_;
    const std::chrono::microseconds delay_;
    std::condition_variable_any ready_;
    std::deque<wqsTask> queue_;
    size_t inflight_ = 0;
    std::thread worker_;
};

/** Process-wide SDK resources shared by independently configured engines. */
class nixlWqsSession {
public:
    nixlWqsSession(nixlWqsConfig config,
                   std::shared_ptr<iWqsClient> client,
                   std::shared_ptr<nixlWqsBufferPool> pool);
    nixlWqsSession(const nixlWqsSession &) = delete;
    nixlWqsSession &
    operator=(const nixlWqsSession &) = delete;

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

    [[nodiscard]] std::mutex &
    transferMutex() noexcept {
        return transferMutex_;
    }

private:
    nixlWqsConfig config_;
    std::shared_ptr<iWqsClient> client_;
    std::shared_ptr<nixlWqsBufferPool> pool_;
    // Engines can have different sync modes but still share the staging allocations.
    std::mutex transferMutex_;
};

#endif // NIXL_SRC_PLUGINS_WQS_WQS_SESSION_H
