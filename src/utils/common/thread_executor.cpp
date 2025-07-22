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

ThreadExecutor::ThreadExecutor() : stop_requested_(false), thread_active_(false) {

    io_context_ = std::make_unique<boost::asio::io_context>();
    work_guard_ = std::make_unique<boost::asio::io_context::work>(*io_context_);

    executor_thread_ = std::thread(&ThreadExecutor::executorLoop, this);

    while (!thread_active_.load()) {
        std::this_thread::yield();
    }
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
ThreadExecutor::initialize() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (initialized_.load()) {
        NIXL_WARN << "ThreadExecutor already initialized";
        return;
    }

    instance_.reset(new ThreadExecutor());
    initialized_.store(true);
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

    try {
        io_context_->run();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Exception in ThreadExecutor io_context: " << e.what();
    }
    catch (...) {
        NIXL_ERROR << "Unknown exception in ThreadExecutor io_context";
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

void
ThreadExecutor::scheduleTask(std::shared_ptr<TaskInfo> task_info) {
    if (!task_info->active || !task_info->timer) {
        return;
    }

    auto delay = std::chrono::duration_cast<std::chrono::milliseconds>(
        task_info->next_execution - std::chrono::steady_clock::now());

    if (delay.count() <= 0) {
        delay = std::chrono::milliseconds(1);
    }

    task_info->timer->expires_after(delay);
    task_info->timer->async_wait([this, task_info](const boost::system::error_code &ec) {
        handleTaskExecution(ec, task_info);
    });
}

void
ThreadExecutor::handleTaskExecution(const boost::system::error_code &ec,
                                    std::shared_ptr<TaskInfo> task_info) {
    if (ec) {
        if (ec != boost::asio::error::operation_aborted) {
            NIXL_ERROR << "Timer error for task '" << task_info->name << "': " << ec.message();
        }
        return;
    }

    if (!task_info->active) {
        return;
    }

    // Execute the task
    executeTask(task_info);

    // Handle task mode
    if (task_info->mode == TaskMode::ONESHOT) {
        // Remove oneshot task
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        tasks_.erase(task_info->name);
        NIXL_DEBUG << "Removed oneshot task: " << task_info->name;
    } else if (task_info->mode == TaskMode::PERIODIC && task_info->active) {
        // Reschedule periodic task
        task_info->next_execution = getNextExecutionTime(task_info);
        scheduleTask(task_info);
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

    auto task_info = std::make_shared<TaskInfo>(
        task, TaskMode::ONESHOT, std::chrono::microseconds(0), name, *io_context_);
    tasks_[name] = task_info;

    NIXL_DEBUG << "Registered oneshot task: " << name;

    // Schedule the task immediately
    scheduleTask(task_info);

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

    auto task_info =
        std::make_shared<TaskInfo>(task, TaskMode::PERIODIC, interval, name, *io_context_);
    tasks_[name] = task_info;

    NIXL_DEBUG << "Registered periodic task: " << name << " with interval: " << interval.count()
               << " microseconds";

    // Schedule the task
    scheduleTask(task_info);

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

    // Cancel the timer and mark as inactive
    if (it->second->timer) {
        it->second->timer->cancel();
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

    // Cancel all timers
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        for (auto &[name, task_info] : tasks_) {
            if (task_info->timer) {
                task_info->timer->cancel();
            }
            task_info->active = false;
        }
        tasks_.clear();
    }

    // Stop the io_context
    if (work_guard_) {
        work_guard_.reset();
    }

    if (io_context_) {
        io_context_->stop();
    }

    if (executor_thread_.joinable()) {
        executor_thread_.join();
    }

    NIXL_DEBUG << "ThreadExecutor stopped";
}

} // namespace nixl