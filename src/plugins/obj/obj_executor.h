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

#ifndef OBJ_EXECUTOR_H
#define OBJ_EXECUTOR_H

#include <aws/core/utils/threading/Executor.h>
#include <asio.hpp>
#include <functional>

class asioThreadPoolExecutor : public Aws::Utils::Threading::Executor {
public:
    explicit asioThreadPoolExecutor(std::size_t num_threads) : pool_(num_threads) {}

    void
    WaitUntilStopped() override {
        pool_.stop();
        pool_.join();
    }

    void
    waitUntilIdle() {
        pool_.wait();
    }

protected:
    bool
    SubmitToThread(std::function<void()> &&task) override {
        asio::post(pool_, std::move(task));
        return true;
    }

private:
    asio::thread_pool pool_;
};

#endif // OBJ_EXECUTOR_H
