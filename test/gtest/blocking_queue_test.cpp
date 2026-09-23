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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <stop_token>
#include <thread>
#include <vector>

#include "blocking_queue.h"

namespace {

struct item : nixl::blockingQueue<item>::node {
    size_t value = 0;
};

} // namespace

namespace nixl {

TEST(blockingQueueTest, FIFO) {
    blockingQueue<item> queue;
    EXPECT_EQ(queue.tryPop(), nullptr);

    std::array<item, 5> items;
    for (auto &item : items) {
        queue.push(&item);
    }
    for (auto &item : items) {
        EXPECT_EQ(queue.tryPop(), &item);
    }
    EXPECT_EQ(queue.tryPop(), nullptr);
}

TEST(blockingQueueTest, Copy) {
    blockingQueue<item> queue;
    item first;
    queue.push(&first);

    item copy = first;
    queue.push(&copy);

    EXPECT_EQ(queue.tryPop(), &first);
    EXPECT_EQ(queue.tryPop(), &copy);
    EXPECT_EQ(queue.tryPop(), nullptr);
}

TEST(blockingQueueTest, Stop) {
    blockingQueue<item> queue;
    item single;
    std::stop_source stopped;
    stopped.request_stop();

    // A stop request ends the wait, queued items are still drained first
    EXPECT_EQ(queue.pop(stopped.get_token()), nullptr);
    queue.push(&single);
    EXPECT_EQ(queue.pop(stopped.get_token()), &single);
    EXPECT_EQ(queue.pop(stopped.get_token()), nullptr);

    // A stop request wakes a blocked pop. The consumer announces that it is
    // about to block; a stop landing just before the wait returns null as well,
    // and a missed wake-up hangs the join below.
    std::atomic<bool> popping{false};
    std::jthread consumer([&](std::stop_token token) {
        popping.store(true);
        EXPECT_EQ(queue.pop(token), nullptr);
    });
    while (!popping.load()) {
        std::this_thread::yield();
    }
    consumer.request_stop();
}

TEST(blockingQueueTest, MPMC) {
    constexpr size_t num_threads = 4;
    constexpr size_t num_items = 1000;
    blockingQueue<item> queue;
    std::vector<item> items(num_threads * num_items);

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < num_items; ++i) {
                queue.push(&items[t * num_items + i]);
            }
        });
        threads.emplace_back([&]() {
            for (size_t i = 0; i < num_items; ++i) {
                ++queue.pop()->value;
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (const auto &item : items) {
        EXPECT_EQ(item.value, 1);
    }
}

} // namespace nixl
