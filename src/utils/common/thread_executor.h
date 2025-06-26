/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NIXL_THREAD_EXECUTOR_H
#define _NIXL_THREAD_EXECUTOR_H

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <string>
#include "nixl_time.h"

namespace nixl {

/**
 * @brief Task execution mode
 */
enum class TaskMode {
    ONESHOT, ///< Execute task once and remove from queue
    PERIODIC ///< Execute task periodically until unregistered
};

/**
 * @brief Task information structure
 */
struct TaskInfo {
    std::function<void()> task;
    TaskMode mode;
    std::chrono::microseconds interval;
    std::chrono::steady_clock::time_point next_execution;
    bool active;
    std::string name;

    TaskInfo() : mode(TaskMode::ONESHOT), interval(std::chrono::microseconds(0)), active(false) {}

    TaskInfo(const std::function<void()> &t,
             TaskMode m,
             const std::chrono::microseconds &i,
             const std::string &n)
        : task(t),
          mode(m),
          interval(i),
          active(true),
          name(n) {
        next_execution = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Thread executor for managing periodic and oneshot tasks
 *
 * This class provides a centralized thread executor that can manage multiple
 * tasks with different execution modes. It supports both oneshot execution
 * (execute once and remove) and periodic execution (execute repeatedly at
 * specified intervals).
 *
 * This is implemented as a singleton to provide global access from anywhere
 * in the codebase.
 */
class ThreadExecutor {
private:
    // Singleton instance
    static std::unique_ptr<ThreadExecutor> instance_;
    static std::mutex instance_mutex_;
    static std::atomic<bool> initialized_;

    std::thread executor_thread_;
    mutable std::mutex tasks_mutex_;
    std::condition_variable tasks_cv_;
    std::atomic<bool> stop_requested_;
    std::atomic<bool> thread_active_;

    std::unordered_map<std::string, std::shared_ptr<TaskInfo>> tasks_;
    std::chrono::microseconds default_poll_interval_;

    explicit ThreadExecutor(std::chrono::microseconds poll_interval = std::chrono::milliseconds(1));

    void
    executorLoop();

    void
    executeTask(std::shared_ptr<TaskInfo> task_info);

    std::chrono::steady_clock::time_point
    getNextExecutionTime(const std::shared_ptr<TaskInfo> &task_info);

public:
    ~ThreadExecutor();

    static ThreadExecutor &
    getInstance();

    static void
    initialize(std::chrono::microseconds poll_interval = std::chrono::milliseconds(1));

    static void
    shutdown();

    static bool
    isInitialized() {
        return initialized_.load();
    }

    bool
    registerOneshotTask(const std::string &name, std::function<void()> task);

    bool
    registerPeriodicTask(const std::string &name,
                         std::function<void()> task,
                         std::chrono::microseconds interval);

    bool
    unregisterTask(const std::string &name);

    void
    stop();

    bool
    isRunning() const {
        return thread_active_.load();
    }

    // Disable copy constructor and assignment operator
    ThreadExecutor(const ThreadExecutor &) = delete;
    ThreadExecutor &
    operator=(const ThreadExecutor &) = delete;
};

} // namespace nixl

#endif // _NIXL_THREAD_EXECUTOR_H