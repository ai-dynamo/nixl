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
#include <deque>
#include <iterator>
#include <mutex>
#include <thread>
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
nixlMetadataWorker::start(poll_t poll, nixlTime::us_t delay) {
    if (thread_.joinable()) {
        return;
    }
    poll_ = std::move(poll);
    delay_ = delay;
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this] {
        try {
            loop();
        }
        catch (...) {
            exception_ = std::current_exception();
        }
    });
}

void
nixlMetadataWorker::stop() {
    if (!thread_.joinable()) {
        return;
    }
    // Let the loop drain what is queued so a send/invalidate issued just before
    // shutdown still reaches the peer/store. running_ stays set until the thread
    // is gone, so a task submitted meanwhile is drained here rather than run by
    // its caller alongside the still-running loop.
    while (true) {
        {
            const std::lock_guard lk(mutex_);
            if (tasks_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stop_.store(true);
    thread_.join();
    running_.store(false);
    if (exception_) {
        try {
            std::rethrow_exception(exception_);
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Metadata worker thread threw an exception: " << e.what();
        }
        exception_ = nullptr;
    }
}

void
nixlMetadataWorker::submit(nixl_worker_task_t task) {
    if (!running_.load()) {
        // No thread to run this on, and queueing it would drop it silently. The
        // caller pays for the I/O, at the cost of making the call synchronous;
        // the lock keeps the owner's promise that its transport state is touched
        // by one thread at a time, which the loop provides in the other branch.
        const std::lock_guard lk(inlineMutex_);
        task();
        return;
    }
    const std::lock_guard lk(mutex_);
    tasks_.push_back(std::move(task));
}

void
nixlMetadataWorker::runQueuedTasks(std::chrono::steady_clock::time_point until) {
    std::deque<nixl_worker_task_t> batch;
    {
        const std::lock_guard lk(mutex_);
        batch.swap(tasks_);
    }
    while (!batch.empty()) {
        // Isolate each unit of work: one throwing task is logged and the worker
        // keeps running, rather than tearing down all metadata processing.
        try {
            batch.front()();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker task threw an exception: " << e.what();
        }
        batch.pop_front();
        if (std::chrono::steady_clock::now() >= until) {
            break;
        }
    }
    if (!batch.empty()) {
        // Put the remainder back in front of anything submitted meanwhile, so
        // tasks still run in the order they were issued.
        const std::lock_guard lk(mutex_);
        tasks_.insert(tasks_.begin(),
                      std::make_move_iterator(batch.begin()),
                      std::make_move_iterator(batch.end()));
    }
}

void
nixlMetadataWorker::loop() {
    while (!stop_.load()) {
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
        std::this_thread::sleep_for(std::chrono::microseconds(delay_));
    }
    // stop() waits for the queue to look empty, which it can while a pass still
    // holds tasks the budget deferred; run those before leaving so nothing
    // submitted before shutdown is dropped.
    runQueuedTasks(std::chrono::steady_clock::time_point::max());
}
