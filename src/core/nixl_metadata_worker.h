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
/**
 * @file nixl_metadata_worker.h
 * @brief Background thread a metadata backend composes to run its own I/O.
 */
#ifndef NIXL_SRC_CORE_NIXL_METADATA_WORKER_H
#define NIXL_SRC_CORE_NIXL_METADATA_WORKER_H

#include "common/nixl_time.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

/** A unit of transport I/O produced on the caller thread, run on the worker. */
using nixl_worker_task_t = std::function<void()>;

/**
 * @class nixlMetadataWorker
 * @brief Thread that drains a task queue and calls a poll callback each pass.
 *
 * The thread management shared by the backends that need one: each owns an
 * instance rather than sharing a manager-wide thread, so a backend blocked on
 * its store cannot hold up the others. Declare it last in the owning backend so
 * it joins before the state its tasks touch is destroyed.
 *
 * A worker that was never started still accepts tasks: submit() runs them on
 * the caller thread, serialized, which is what a backend with no background
 * work needs (P2P without a listener).
 */
class nixlMetadataWorker {
public:
    using poll_t = std::function<void()>;

    nixlMetadataWorker() = default;
    ~nixlMetadataWorker();

    nixlMetadataWorker(const nixlMetadataWorker &) = delete;
    nixlMetadataWorker &
    operator=(const nixlMetadataWorker &) = delete;

    /**
     * @brief Launch the loop (no-op if already running). Each pass runs queued
     *        tasks up to a time budget, calls @p poll, then sleeps @p delay.
     */
    void
    start(poll_t poll, nixlTime::us_t delay);

    /** @brief Drain queued tasks, then stop and join. Idempotent. */
    void
    stop();

    /**
     * @brief Run @p task on the worker thread, or inline on the caller thread
     *        when this worker is not running.
     */
    void
    submit(nixl_worker_task_t task);

private:
    void
    loop();

    // Run queued tasks until @p until, leaving any remainder queued in order.
    void
    runQueuedTasks(std::chrono::steady_clock::time_point until);

    poll_t poll_;
    nixlTime::us_t delay_ = 0;
    std::deque<nixl_worker_task_t> tasks_;
    std::mutex mutex_;
    // Serializes the inline path, where callers run tasks themselves.
    std::mutex inlineMutex_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::exception_ptr exception_;
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_WORKER_H
