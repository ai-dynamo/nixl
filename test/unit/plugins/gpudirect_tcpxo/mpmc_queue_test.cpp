/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "mpmc_queue.h"

namespace tcpxo {
namespace {

    TEST(MPMCQueueTest, EnqueueDequeue) {
        MPMCQueue<int> queue;
        EXPECT_TRUE(queue.IsEmpty());

        queue.Enqueue(1);
        EXPECT_FALSE(queue.IsEmpty());

        queue.Enqueue(2);
        EXPECT_FALSE(queue.IsEmpty());

        auto val1 = queue.TryDequeue();
        ASSERT_TRUE(val1.ok());
        EXPECT_EQ(*val1, 1);

        auto val2 = queue.TryDequeue();
        ASSERT_TRUE(val2.ok());
        EXPECT_EQ(*val2, 2);

        EXPECT_TRUE(queue.IsEmpty());
    }

    TEST(MPMCQueueTest, EmptyQueue) {
        MPMCQueue<int> queue;
        EXPECT_TRUE(queue.IsEmpty());

        auto val = queue.TryDequeue();
        EXPECT_FALSE(val.ok());
        EXPECT_EQ(val.status().code(), absl::StatusCode::kOutOfRange);
    }

    TEST(MPMCQueueTest, MoveSemantics) {
        MPMCQueue<std::unique_ptr<int>> queue;
        auto ptr = std::make_unique<int>(42);

        queue.Enqueue(std::move(ptr));
        EXPECT_EQ(ptr, nullptr);

        auto dequeued_ptr_or = queue.TryDequeue();
        ASSERT_TRUE(dequeued_ptr_or.ok());
        auto dequeued_ptr = std::move(*dequeued_ptr_or);
        ASSERT_NE(dequeued_ptr, nullptr);
        EXPECT_EQ(*dequeued_ptr, 42);
    }

    TEST(MPMCQueueTest, MultiThreaded) {
        static constexpr int num_producers = 4;
        static constexpr int num_consumers = 4;
        static constexpr int items_per_producer = 100000;
        static constexpr int total_items = num_producers * items_per_producer;
        MPMCQueue<int> queue;

        std::vector<absl::Duration> producer_durations(num_producers);
        std::vector<std::thread> producers;
        for (int i = 0; i < num_producers; i++) {
            producers.emplace_back([&queue, &producer_durations, i]() {
                absl::Time start = absl::Now();
                for (int j = 0; j < items_per_producer; j++) {
                    queue.Enqueue(1);
                }
                producer_durations[i] = absl::Now() - start;
            });
        }

        std::atomic<int> total_consumed{0};
        std::vector<absl::Duration> consumer_durations(num_consumers);
        std::vector<std::thread> consumers;
        for (int i = 0; i < num_consumers; i++) {
            consumers.emplace_back([&queue, &total_consumed, &consumer_durations, i]() {
                absl::Time start = absl::Now();
                while (total_consumed < total_items) {
                    auto val = queue.TryDequeue();
                    if (val.ok()) {
                        total_consumed.fetch_add(1);
                    } else {
                        std::this_thread::yield();
                    }
                }
                consumer_durations[i] = absl::Now() - start;
            });
        }

        for (auto &t : producers) {
            t.join();
        }
        for (auto &t : consumers) {
            t.join();
        }

        absl::Duration total_producer_duration = absl::ZeroDuration();
        for (const auto &d : producer_durations) {
            total_producer_duration += d;
        }

        absl::Duration total_consumer_duration = absl::ZeroDuration();
        for (const auto &d : consumer_durations) {
            total_consumer_duration += d;
        }

        std::cout << "Average Producer duration: " << total_producer_duration / num_producers
                  << ". Average time per enqueue: "
                  << total_producer_duration / num_producers / total_items << std::endl;
        std::cout << "Average Consumer duration: " << total_consumer_duration / num_consumers
                  << ". Average time per dequeue: "
                  << total_consumer_duration / num_consumers / total_consumed.load() << std::endl;

        EXPECT_EQ(total_consumed.load(), total_items);
        EXPECT_TRUE(queue.IsEmpty());
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
