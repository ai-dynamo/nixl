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
    std::function<void()> task; ///< Task function to execute
    TaskMode mode; ///< Execution mode (oneshot or periodic)
    std::chrono::microseconds interval; ///< Interval for periodic tasks
    std::chrono::steady_clock::time_point next_execution; ///< Next execution time
    bool active; ///< Whether task is active
    std::string name; ///< Task name for identification

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

    /**
     * @brief Constructor (private for singleton)
     * @param poll_interval Default polling interval for the executor thread
     */
    explicit ThreadExecutor(std::chrono::microseconds poll_interval = std::chrono::milliseconds(1));

    /**
     * @brief Main executor thread function
     */
    void
    executorLoop();

    /**
     * @brief Execute a single task
     * @param task_info Task to execute
     */
    void
    executeTask(std::shared_ptr<TaskInfo> task_info);

    /**
     * @brief Get the next execution time for a task
     * @param task_info Task information
     * @return Next execution time
     */
    std::chrono::steady_clock::time_point
    getNextExecutionTime(const std::shared_ptr<TaskInfo> &task_info);

public:
    /**
     * @brief Destructor
     */
    ~ThreadExecutor();

    // Singleton methods
    /**
     * @brief Get the singleton instance
     * @return Reference to the singleton ThreadExecutor
     */
    static ThreadExecutor &
    getInstance();

    /**
     * @brief Initialize the singleton with custom polling interval
     * @param poll_interval Polling interval for the executor
     */
    static void
    initialize(std::chrono::microseconds poll_interval = std::chrono::milliseconds(1));

    /**
     * @brief Shutdown the singleton instance
     */
    static void
    shutdown();

    /**
     * @brief Check if the singleton is initialized
     * @return true if initialized, false otherwise
     */
    static bool
    isInitialized() {
        return initialized_.load();
    }

    // Task management methods
    /**
     * @brief Register a oneshot task
     * @param name Unique name for the task
     * @param task Function to execute
     * @return true if registration successful, false otherwise
     */
    bool
    registerOneshotTask(const std::string &name, std::function<void()> task);

    /**
     * @brief Register a periodic task
     * @param name Unique name for the task
     * @param task Function to execute
     * @param interval Execution interval
     * @return true if registration successful, false otherwise
     */
    bool
    registerPeriodicTask(const std::string &name,
                         std::function<void()> task,
                         std::chrono::microseconds interval);

    /**
     * @brief Unregister a task
     * @param name Name of the task to unregister
     * @return true if unregistration successful, false if task not found
     */
    bool
    unregisterTask(const std::string &name);

    /**
     * @brief Check if a task is registered
     * @param name Name of the task to check
     * @return true if task is registered, false otherwise
     */
    bool
    isTaskRegistered(const std::string &name) const;

    /**
     * @brief Get the number of registered tasks
     * @return Number of active tasks
     */
    size_t
    getTaskCount() const;

    /**
     * @brief Stop the executor thread
     */
    void
    stop();

    /**
     * @brief Check if the executor is running
     * @return true if executor is active, false otherwise
     */
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