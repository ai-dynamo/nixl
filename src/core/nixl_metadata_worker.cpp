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
#include "nixl_metadata_worker.h"

#include "common/nixl_log.h"

#include <chrono>
#include <mutex>
#include <utility>

namespace {

// How long one pass spends on queued tasks before polling. A task can block on
// store I/O, so draining the queue unconditionally would hold back this
// backend's inbound servicing (P2P accepts, etcd invalidations) for as long as
// its slowest operation takes; the remainder stays queued for the next pass.
constexpr auto task_budget = std::chrono::milliseconds(100);

} // namespace

nixlMetadataWorker::~nixlMetadataWorker() {
    stop();
}

void
nixlMetadataWorker::start(poll_t poll, std::chrono::microseconds delay) {
    if (thread_.joinable()) {
        return;
    }
    poll_ = std::move(poll);
    delay_ = delay;
    {
        const std::lock_guard lk(mutex_);
        stopping_ = false;
        started_ = true;
    }
    thread_ = std::thread([this] {
        try {
            loop();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker thread died: " << e.what();
        }
        catch (...) {
            NIXL_ERROR << "Metadata worker thread died with an unknown exception";
        }
    });
}

void
nixlMetadataWorker::stop() {
    if (!thread_.joinable()) {
        return;
    }
    {
        const std::lock_guard lk(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    // The loop drains what is queued before it exits, so a send/invalidate
    // issued just before shutdown still reaches the peer/store. started_ stays
    // set until the thread is gone, so a task submitted meanwhile is drained
    // there rather than run by its caller alongside the still-running loop.
    thread_.join();
    const std::lock_guard lk(mutex_);
    started_ = false;
}

void
nixlMetadataWorker::submit(nixl_worker_task_t task) {
    {
        const std::lock_guard lk(mutex_);
        if (started_) {
            tasks_.push_back(std::move(task));
            cv_.notify_one();
            return;
        }
    }
    // Nothing would ever run a queued task here: with no thread it would sit in
    // tasks_ until the worker is destroyed. The caller pays for the I/O instead,
    // at the cost of making the call synchronous, and the lock keeps the owner's
    // promise that its transport state is touched by one thread at a time.
    const std::lock_guard lk(inlineMutex_);
    task();
}

void
nixlMetadataWorker::runQueuedTasks(std::chrono::steady_clock::time_point until) {
    while (true) {
        nixl_worker_task_t task;
        {
            const std::lock_guard lk(mutex_);
            if (tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        // Isolate each unit of work: one throwing task is logged and the worker
        // keeps running, rather than tearing down all metadata processing.
        try {
            task();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker task threw an exception: " << e.what();
        }
        catch (...) {
            NIXL_ERROR << "Metadata worker task threw an unknown exception";
        }
        if (std::chrono::steady_clock::now() >= until) {
            return;
        }
    }
}

void
nixlMetadataWorker::loop() {
    while (true) {
        {
            std::unique_lock lk(mutex_);
            // Wake on submitted work or shutdown; time out to poll anyway, which
            // is what makes delay_ the interval between polls rather than a floor
            // on how long a submitted task waits to start.
            cv_.wait_for(lk, delay_, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_) {
                break;
            }
        }
        // Spend a bounded slice on tasks, then poll: a task can block on I/O (an
        // etcd fetch waits on a watch), and draining the whole queue first would
        // stall inbound servicing behind it. At least one task runs per pass, so
        // the queue still drains.
        runQueuedTasks(std::chrono::steady_clock::now() + task_budget);
        try {
            if (poll_) {
                poll_();
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker poll threw an exception: " << e.what();
        }
        catch (...) {
            NIXL_ERROR << "Metadata worker poll threw an unknown exception";
        }
    }
    // Anything queued before shutdown, including tasks a pass deferred when its
    // budget ran out, runs before the thread leaves.
    runQueuedTasks(std::chrono::steady_clock::time_point::max());
}
