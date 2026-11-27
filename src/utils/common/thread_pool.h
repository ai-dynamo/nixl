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
#ifndef NIXL_SRC_UTILS_COMMON_THREAD_POOL_H
#define NIXL_SRC_UTILS_COMMON_THREAD_POOL_H

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "nixl_log.h"

namespace nixl {

/**
 * @brief Threads that run queued tasks in order
 *
 * Tasks must not throw. Tasks queued before destruction still run.
 */
class threadPool {
public:
    /**
     * @param num_threads Number of threads, capped at maxThreads()
     */
    explicit threadPool(size_t num_threads) {
        if (num_threads > maxThreads()) {
            NIXL_WARN << "Capping thread pool from " << num_threads << " to " << maxThreads()
                      << " threads";
            num_threads = maxThreads();
        }
        threads_.reserve(num_threads);
        try {
            for (size_t i = 0; i < num_threads; ++i) {
                threads_.emplace_back([this] { worker(); });
            }
        }
        catch (...) {
            // The destructor doesn't run, so join the threads that did start.
            stop();
            throw;
        }
    }

    threadPool(const threadPool &) = delete;
    threadPool &
    operator=(const threadPool &) = delete;

    ~threadPool() {
        stop();
    }

    /**
     * @brief Largest pool: four threads per hardware thread
     */
    [[nodiscard]] static size_t
    maxThreads() {
        return static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency())) * 4;
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return threads_.size();
    }

    /**
     * @brief Queue a task
     * @return false if the pool is stopping, in which case the task never runs
     */
    [[nodiscard]] bool
    submit(std::function<void()> task) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                NIXL_WARN << "Ignoring a task submitted after the thread pool stopped";
                return false;
            }
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
        return true;
    }

private:
    void
    stop() {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &thread : threads_) {
            thread.join();
        }
    }

    void
    worker() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_THREAD_POOL_H
