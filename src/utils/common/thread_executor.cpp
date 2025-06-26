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
#include "thread_executor.h"
#include "nixl_log.h"

namespace nixl {

// Static member initialization
std::unique_ptr<ThreadExecutor> ThreadExecutor::instance_;
std::mutex ThreadExecutor::instance_mutex_;
std::atomic<bool> ThreadExecutor::initialized_{false};

ThreadExecutor::ThreadExecutor(std::chrono::microseconds poll_interval)
    : stop_requested_(false),
      thread_active_(false),
      default_poll_interval_(poll_interval) {

    executor_thread_ = std::thread(&ThreadExecutor::executorLoop, this);

    // Wait for thread to become active
    while (!thread_active_.load()) {
        std::this_thread::yield();
    }

    NIXL_DEBUG << "ThreadExecutor started with poll interval: " << default_poll_interval_.count()
               << " microseconds";
}

ThreadExecutor::~ThreadExecutor() {
    stop();
}

ThreadExecutor &
ThreadExecutor::getInstance() {
    if (!initialized_.load()) {
        initialize();
    }
    return *instance_;
}

void
ThreadExecutor::initialize(std::chrono::microseconds poll_interval) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (initialized_.load()) {
        NIXL_WARN << "ThreadExecutor already initialized";
        return;
    }

    ThreadExecutor *raw_ptr = new ThreadExecutor(poll_interval);
    instance_.reset(raw_ptr);
    initialized_.store(true);

    NIXL_INFO << "ThreadExecutor initialized with poll interval: " << poll_interval.count()
              << " microseconds";
}

void
ThreadExecutor::shutdown() {
    std::lock_guard<std::mutex> lock(instance_mutex_);

    if (!initialized_.load()) {
        NIXL_WARN << "ThreadExecutor not initialized, nothing to shutdown";
        return;
    }

    if (instance_) {
        instance_->stop();
        instance_.reset();
    }

    initialized_.store(false);
    NIXL_INFO << "ThreadExecutor shutdown complete";
}

void
ThreadExecutor::executorLoop() {
    thread_active_ = true;

    while (!stop_requested_.load()) {
        std::unique_lock<std::mutex> lock(tasks_mutex_);

        if (tasks_.empty()) {
            tasks_cv_.wait_for(lock, default_poll_interval_, [this] {
                return stop_requested_.load() || !tasks_.empty();
            });
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> tasks_to_remove;
        std::vector<std::shared_ptr<TaskInfo>> tasks_to_execute;

        for (auto &[name, task_info] : tasks_) {
            if (!task_info->active) {
                tasks_to_remove.push_back(name);
                continue;
            }

            if (now >= task_info->next_execution) {
                tasks_to_execute.push_back(task_info);

                if (task_info->mode == TaskMode::ONESHOT) {
                    tasks_to_remove.push_back(name);
                } else {
                    task_info->next_execution = getNextExecutionTime(task_info);
                }
            }
        }

        for (const auto &name : tasks_to_remove) {
            tasks_.erase(name);
        }

        lock.unlock();

        for (auto &task_info : tasks_to_execute) {
            executeTask(task_info);
        }

        if (tasks_to_execute.empty()) {
            std::this_thread::sleep_for(default_poll_interval_);
        }
    }

    thread_active_ = false;
    NIXL_DEBUG << "ThreadExecutor thread stopped";
}

void
ThreadExecutor::executeTask(std::shared_ptr<TaskInfo> task_info) {
    try {
        NIXL_DEBUG << "Executing task: " << task_info->name;
        task_info->task();
        NIXL_DEBUG << "Task completed: " << task_info->name;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Exception in task '" << task_info->name << "': " << e.what();
    }
    catch (...) {
        NIXL_ERROR << "Unknown exception in task '" << task_info->name << "'";
    }
}

std::chrono::steady_clock::time_point
ThreadExecutor::getNextExecutionTime(const std::shared_ptr<TaskInfo> &task_info) {
    auto now = std::chrono::steady_clock::now();
    return now + task_info->interval;
}

bool
ThreadExecutor::registerOneshotTask(const std::string &name, std::function<void()> task) {
    if (name.empty() || !task) {
        NIXL_ERROR << "Invalid task registration: empty name or null task";
        return false;
    }

    std::lock_guard<std::mutex> lock(tasks_mutex_);

    if (tasks_.find(name) != tasks_.end()) {
        NIXL_ERROR << "Task with name '" << name << "' already registered";
        return false;
    }

    auto task_info =
        std::make_shared<TaskInfo>(task, TaskMode::ONESHOT, std::chrono::microseconds(0), name);
    tasks_[name] = task_info;

    NIXL_DEBUG << "Registered oneshot task: " << name;
    tasks_cv_.notify_one();

    return true;
}

bool
ThreadExecutor::registerPeriodicTask(const std::string &name,
                                     std::function<void()> task,
                                     std::chrono::microseconds interval) {
    if (name.empty() || !task) {
        NIXL_ERROR << "Invalid task registration: empty name or null task";
        return false;
    }

    if (interval.count() <= 0) {
        NIXL_ERROR << "Invalid interval for periodic task: " << interval.count() << " microseconds";
        return false;
    }

    std::lock_guard<std::mutex> lock(tasks_mutex_);

    if (tasks_.find(name) != tasks_.end()) {
        NIXL_ERROR << "Task with name '" << name << "' already registered";
        return false;
    }

    auto task_info = std::make_shared<TaskInfo>(task, TaskMode::PERIODIC, interval, name);
    tasks_[name] = task_info;

    NIXL_DEBUG << "Registered periodic task: " << name << " with interval: " << interval.count()
               << " microseconds";
    tasks_cv_.notify_one();

    return true;
}

bool
ThreadExecutor::unregisterTask(const std::string &name) {
    if (name.empty()) {
        NIXL_ERROR << "Cannot unregister task with empty name";
        return false;
    }

    std::lock_guard<std::mutex> lock(tasks_mutex_);

    auto it = tasks_.find(name);
    if (it == tasks_.end()) {
        NIXL_DEBUG << "Task '" << name << "' not found for unregistration";
        return false;
    }

    it->second->active = false;
    tasks_.erase(it);

    NIXL_DEBUG << "Unregistered task: " << name;
    return true;
}

void
ThreadExecutor::stop() {
    if (stop_requested_.load()) {
        return; // Already stopping or stopped
    }

    NIXL_DEBUG << "Stopping ThreadExecutor";
    stop_requested_ = true;
    tasks_cv_.notify_all();

    if (executor_thread_.joinable()) {
        executor_thread_.join();
    }

    // Clear any remaining tasks
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    tasks_.clear();

    NIXL_DEBUG << "ThreadExecutor stopped";
}

} // namespace nixl