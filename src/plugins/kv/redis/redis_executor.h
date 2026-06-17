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
 * @class redisThreadPoolExecutor
 * @brief Small ASIO thread pool used to complete Redis async operations.
 */
class redisThreadPoolExecutor {
public:
    explicit redisThreadPoolExecutor(std::size_t num_threads) : pool_(num_threads) {}

    void
    WaitUntilStopped() {
        pool_.stop();
        pool_.join();
    }

    void
    waitUntilIdle() {
        pool_.wait();
    }

    void
    post(std::function<void()> &&task) {
        asio::post(pool_, std::move(task));
    }

private:
    asio::thread_pool pool_;
};

#endif // NIXL_SRC_PLUGINS_KV_REDIS_EXECUTOR_H
