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

#include <chrono>
#include <condition_variable>
#include <deque>
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
     *        tasks up to a time budget and calls @p poll; @p delay is how long
     *        a pass waits for work before polling anyway.
     */
    void
    start(poll_t poll, std::chrono::microseconds delay);

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
    std::chrono::microseconds delay_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<nixl_worker_task_t> tasks_;
    // Whether a thread exists to run tasks. Held under mutex_ rather than read
    // off thread_: submit() is reachable from any thread, and inspecting
    // thread_ while stop() is inside join() would race on the thread object.
    bool started_ = false;
    bool stopping_ = false;
    // Serializes the inline path. Separate from mutex_ so running a task never
    // holds the queue lock across transport I/O, and so a task that submits
    // cannot deadlock against a non-recursive mutex.
    std::mutex inlineMutex_;
    std::thread thread_;
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_WORKER_H
