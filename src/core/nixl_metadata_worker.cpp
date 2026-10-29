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

// The worker whose thread is running this call, so submit() can tell a task
// adding more work from an outside caller. Not thread_.get_id(): submit() would
// then read the thread object while stop() is inside join(), which writes it.
thread_local const nixlMetadataWorker *current_worker = nullptr;

} // namespace

nixlMetadataWorker::~nixlMetadataWorker() {
    stop();
}

void
nixlMetadataWorker::start(nixl_metadata_task_t poll, std::chrono::microseconds delay) {
    // Held across the spawn so state_ and thread_ change together: a stop()
    // that sees RUNNING must find a thread to join.
    const std::lock_guard lk(mutex_);
    if (state_ != state::IDLE) {
        return;
    }
    poll_ = std::move(poll);
    delay_ = delay;
    thread_ = std::thread([this] {
        current_worker = this;
        // Only allocation and the condition variable can throw past the loop,
        // and a backend whose worker is gone stops exchanging metadata without
        // ever failing a call.
        try {
            loop();
        }
        catch (const std::exception &e) {
            NIXL_FATAL << "Metadata worker thread died: " << e.what();
        }
        catch (...) {
            NIXL_FATAL << "Metadata worker thread died with an unknown exception";
        }
    });
    // After the spawn: a thread that failed to start must not leave a state
    // claiming there is one to join.
    state_ = state::RUNNING;
}

void
nixlMetadataWorker::stop() {
    {
        std::unique_lock lk(mutex_);
        if (state_ == state::IDLE) {
            return;
        }
        if (state_ == state::STOPPING) {
            // Someone else owns the join. Returning has to mean the thread is
            // gone, since the caller tears down state the tasks reach.
            idleCv_.wait(lk, [this] { return state_ == state::IDLE; });
            return;
        }
        state_ = state::STOPPING;
    }
    cv_.notify_all();
    // The loop runs what is already queued before it exits, so a send/invalidate
    // issued just before shutdown still reaches the peer/store. That drain ends
    // because STOPPING closed the queue: were callers still able to add, a
    // steady stream of them would keep the thread, and this join, going.
    thread_.join();
    const std::lock_guard lk(mutex_);
    // Nothing is left to run: STOPPING sent later callers down the inline path
    // and the loop drained the rest. Publishing IDLE is the last touch, since a
    // stop() waiting on it may destroy us.
    state_ = state::IDLE;
    idleCv_.notify_all();
}

void
nixlMetadataWorker::submit(nixl_metadata_task_t task) {
    {
        const std::lock_guard lk(mutex_);
        // The queue is for a thread that will come back for it. Once shutdown
        // starts only the worker itself may still add, since its own drain is
        // what runs the addition; anyone else would be queueing behind a thread
        // that is leaving.
        if (state_ == state::RUNNING || current_worker == this) {
            tasks_.push_back(std::move(task));
            cv_.notify_one();
            return;
        }
    }
    // No thread will come for it, so the caller pays for the I/O instead, at the
    // cost of making the call synchronous.
    runTask(task);
}

void
nixlMetadataWorker::runQueuedTasks(std::chrono::steady_clock::time_point until) {
    while (true) {
        nixl_metadata_task_t task;
        {
            const std::lock_guard lk(mutex_);
            if (tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        runTask(task);
        if (std::chrono::steady_clock::now() >= until) {
            return;
        }
    }
}

void
nixlMetadataWorker::runTask(nixl_metadata_task_t &task) {
    const std::lock_guard lk(taskMutex_);
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
}

void
nixlMetadataWorker::loop() {
    while (true) {
        {
            std::unique_lock lk(mutex_);
            // Wake on submitted work or shutdown; time out to poll anyway, which
            // is what makes delay_ the interval between polls rather than a floor
            // on how long a submitted task waits to start.
            cv_.wait_for(
                lk, delay_, [this] { return state_ == state::STOPPING || !tasks_.empty(); });
            if (state_ == state::STOPPING) {
                break;
            }
        }
        // Spend a bounded slice on tasks, then poll: a task can block on I/O (an
        // etcd fetch waits on a watch), and draining the whole queue first would
        // stall inbound servicing behind it. At least one task runs per pass, so
        // the queue still drains.
        runQueuedTasks(std::chrono::steady_clock::now() + task_budget);
        if (poll_) {
            // Run like a task: it reaches the backend's transport state too.
            runTask(poll_);
        }
    }
    // Anything queued before shutdown, including tasks a pass deferred when its
    // budget ran out, runs before the thread leaves.
    runQueuedTasks(std::chrono::steady_clock::time_point::max());
}
