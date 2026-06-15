/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_KV_REDIS_EXECUTOR_H
#define NIXL_SRC_PLUGINS_KV_REDIS_EXECUTOR_H

#include <asio.hpp>
#include <functional>
#include <thread>

/**
 * @class asioThreadPoolExecutor
 * @brief ASIO-based thread pool executor for asynchronous task execution
 * 
 * This executor uses ASIO's thread_pool to execute tasks asynchronously across
 * multiple threads. It provides a simple interface for posting tasks that will
 * be executed by worker threads in the pool.
 * 
 * ASIO Design Pattern:
 * - ASIO (Asio C++ library) provides asynchronous I/O operations
 * - thread_pool creates a pool of worker threads that process tasks
 * - post() submits tasks to the pool's work queue
 * - Tasks are executed by worker threads asynchronously (non-blocking)
 * 
 * Why use ASIO executor here?
 * - Redis async callbacks need to be executed in a separate thread context
 * - Prevents blocking the main thread or hiredis event loop thread
 * - Allows concurrent execution of multiple Redis operations
 * - Provides thread-safe task queuing and execution
 */
class asioThreadPoolExecutor {
public:
    /**
     * @brief Construct a thread pool executor with specified number of threads
     * @param num_threads Number of worker threads in the pool
     * 
     * Example: asioThreadPoolExecutor(4) creates a pool with 4 worker threads
     * These threads will continuously process tasks from the work queue
     */
    explicit asioThreadPoolExecutor(std::size_t num_threads) : pool_(num_threads) {
        // ASIO thread_pool automatically starts worker threads
        // Each thread runs an event loop that processes tasks from the queue
    }

    /**
     * @brief Stop the thread pool and wait for all threads to finish
     * 
     * This is called during Redis backend destruction to ensure all
     * pending tasks complete before cleanup.
     */
    void
    WaitUntilStopped() {
        pool_.stop();  // Signal all threads to stop processing new tasks
        pool_.join();  // Wait for all threads to finish current tasks and exit
    }

    /**
     * @brief Wait until all queued tasks are completed
     * 
     * Blocks until the work queue is empty and all tasks have been processed.
     * Useful for synchronization during shutdown.
     */
    void
    waitUntilIdle() {
        pool_.wait();  // Wait for all tasks in the queue to complete
    }

    /**
     * @brief Post a task to the thread pool for asynchronous execution
     * @param task Function object to execute asynchronously
     * 
     * ASIO post() semantics:
     * - Non-blocking: returns immediately after queuing the task
     * - Asynchronous: task executes in a worker thread, not the caller's thread
     * - Thread-safe: multiple threads can post tasks concurrently
     * - Ordering: tasks are executed in FIFO order per thread
     * 
     * Execution flow:
     * 1. Caller thread: post(task) -> returns immediately
     * 2. Task is queued in ASIO's internal work queue
     * 3. Worker thread picks up task from queue
     * 4. Worker thread executes task asynchronously
     * 
     * Example usage:
     *   executor->post([promise_ptr, success]() {
     *       promise_ptr->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
     *   });
     * 
     * This lambda will execute in a worker thread, not the caller's thread.
     */
    void
    post(std::function<void()> &&task) {
        // asio::post schedules the task for execution in the thread pool
        // The task will be executed by one of the worker threads
        asio::post(pool_, std::move(task));
    }

private:
    /**
     * ASIO thread_pool manages a pool of worker threads
     * - Automatically creates and manages worker threads
     * - Provides a work queue for task distribution
     * - Handles thread lifecycle (creation, execution, cleanup)
     * - Ensures thread-safe task execution
     */
    asio::thread_pool pool_;
};

#endif // NIXL_SRC_PLUGINS_KV_REDIS_EXECUTOR_H
