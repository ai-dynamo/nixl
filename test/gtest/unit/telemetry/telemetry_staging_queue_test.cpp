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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "telemetry/telemetry_staging_queue.h"

namespace {

constexpr auto kType = nixl_telemetry_event_type_t::AGENT_TX_BYTES;

nixlTelemetryEvent
makeEvent(uint64_t value) {
    return {kType, value};
}

TEST(telemetryStagingQueueTest, ConstructorRetainsCapacityEmptyNoDrops) {
    nixlTelemetryStagingQueue queue(8);
    EXPECT_EQ(queue.capacity(), 8u);
    EXPECT_TRUE(queue.takePending().empty());
    EXPECT_EQ(queue.takeNumDropped(), 0u);
}

TEST(telemetryStagingQueueTest, SinglePushPreservesTypeAndValue) {
    nixlTelemetryStagingQueue queue(4);
    ASSERT_TRUE(queue.tryPush({nixl_telemetry_event_type_t::AGENT_XFER_TIME, 42}));
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].eventType_, nixl_telemetry_event_type_t::AGENT_XFER_TIME);
    EXPECT_EQ(pending[0].value_, 42u);
}

TEST(telemetryStagingQueueTest, ExactlyCapacitySinglePushesSucceed) {
    nixlTelemetryStagingQueue queue(3);
    for (uint64_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(queue.tryPush(makeEvent(i)));
    }
    EXPECT_EQ(queue.takeNumDropped(), 0u);
}

TEST(telemetryStagingQueueTest, PushBeyondCapacityFailsAndDropsByOne) {
    nixlTelemetryStagingQueue queue(2);
    ASSERT_TRUE(queue.tryPush(makeEvent(0)));
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    EXPECT_FALSE(queue.tryPush(makeEvent(2)));
    EXPECT_EQ(queue.takeNumDropped(), 1u);
}

TEST(telemetryStagingQueueTest, FittingBatchAppendedCompletelyInOrder) {
    nixlTelemetryStagingQueue queue(8);
    const std::vector<nixlTelemetryEvent> batch = {makeEvent(10), makeEvent(20), makeEvent(30)};
    ASSERT_TRUE(queue.tryPushBatch(batch));
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 3u);
    EXPECT_EQ(pending[0].value_, 10u);
    EXPECT_EQ(pending[1].value_, 20u);
    EXPECT_EQ(pending[2].value_, 30u);
    EXPECT_EQ(queue.takeNumDropped(), 0u);
}

TEST(telemetryStagingQueueTest, NonFittingBatchRejectedWithoutPartialInsertion) {
    nixlTelemetryStagingQueue queue(4);
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    ASSERT_TRUE(queue.tryPush(makeEvent(2)));
    const std::vector<nixlTelemetryEvent> batch = {makeEvent(3), makeEvent(4), makeEvent(5)};
    EXPECT_FALSE(queue.tryPushBatch(batch));
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 2u);
    EXPECT_EQ(pending[0].value_, 1u);
    EXPECT_EQ(pending[1].value_, 2u);
}

TEST(telemetryStagingQueueTest, RejectedBatchDropCountEqualsBatchLength) {
    nixlTelemetryStagingQueue queue(4);
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    ASSERT_TRUE(queue.tryPush(makeEvent(2)));
    const std::vector<nixlTelemetryEvent> batch = {makeEvent(3), makeEvent(4), makeEvent(5)};
    EXPECT_FALSE(queue.tryPushBatch(batch));
    EXPECT_EQ(queue.takeNumDropped(), batch.size());
}

TEST(telemetryStagingQueueTest, EmptyBatchSucceedsWithoutChangingState) {
    nixlTelemetryStagingQueue queue(4);
    ASSERT_TRUE(queue.tryPush(makeEvent(7)));
    EXPECT_TRUE(queue.tryPushBatch({}));
    EXPECT_EQ(queue.takeNumDropped(), 0u);
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(pending[0].value_, 7u);
}

TEST(telemetryStagingQueueTest, TakePendingReturnsAllInOrderAndEmpties) {
    nixlTelemetryStagingQueue queue(4);
    for (uint64_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(queue.tryPush(makeEvent(i)));
    }
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 3u);
    for (uint64_t i = 0; i < 3; ++i) {
        EXPECT_EQ(pending[i].value_, i);
    }
    EXPECT_TRUE(queue.takePending().empty());
}

TEST(telemetryStagingQueueTest, SecondDrainIsEmpty) {
    nixlTelemetryStagingQueue queue(4);
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    EXPECT_EQ(queue.takePending().size(), 1u);
    EXPECT_TRUE(queue.takePending().empty());
}

TEST(telemetryStagingQueueTest, AcceptsEventsAfterDrain) {
    nixlTelemetryStagingQueue queue(2);
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    ASSERT_TRUE(queue.tryPush(makeEvent(2)));
    (void)queue.takePending();
    ASSERT_TRUE(queue.tryPush(makeEvent(3)));
    ASSERT_TRUE(queue.tryPush(makeEvent(4)));
    auto pending = queue.takePending();
    ASSERT_EQ(pending.size(), 2u);
    EXPECT_EQ(pending[0].value_, 3u);
    EXPECT_EQ(pending[1].value_, 4u);
}

TEST(telemetryStagingQueueTest, TakeNumDroppedReturnsDeltaOnceThenZero) {
    nixlTelemetryStagingQueue queue(1);
    ASSERT_TRUE(queue.tryPush(makeEvent(1)));
    EXPECT_FALSE(queue.tryPush(makeEvent(2)));
    EXPECT_FALSE(queue.tryPush(makeEvent(3)));
    EXPECT_EQ(queue.takeNumDropped(), 2u);
    EXPECT_EQ(queue.takeNumDropped(), 0u);
}

TEST(telemetryStagingQueueTest, ConcurrentProducersConserveEventSlots) {
    constexpr size_t kProducers = 8;
    constexpr size_t kEventsPerProducer = 5000;
    constexpr size_t kBatchSize = 4;
    constexpr size_t kProduced = kProducers * kEventsPerProducer * kBatchSize;
    // Capacity below total production so both acceptance and drops occur, while
    // a concurrent consumer drains to keep the queue from staying saturated.
    nixlTelemetryStagingQueue queue(256);

    std::atomic<bool> done{false};
    std::vector<nixlTelemetryEvent> drained;
    std::thread consumer([&]() {
        while (!done.load(std::memory_order_acquire)) {
            auto pending = queue.takePending();
            drained.insert(drained.end(), pending.begin(), pending.end());
            std::this_thread::yield();
        }
        auto pending = queue.takePending();
        drained.insert(drained.end(), pending.begin(), pending.end());
    });

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (size_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (size_t i = 0; i < kEventsPerProducer; ++i) {
                const uint64_t base = (p * kEventsPerProducer + i) * kBatchSize;
                std::array<nixlTelemetryEvent, kBatchSize> batch;
                for (size_t b = 0; b < kBatchSize; ++b) {
                    batch[b] = makeEvent(base + b);
                }
                (void)queue.tryPushBatch(batch);
            }
        });
    }
    for (auto &t : producers) {
        t.join();
    }
    done.store(true, std::memory_order_release);
    consumer.join();

    const uint64_t dropped = queue.takeNumDropped();
    EXPECT_EQ(drained.size() + dropped, kProduced);

    // Every accepted identifier appears exactly once, and acceptance is
    // batch-atomic: each fully accepted batch contributes all kBatchSize members.
    std::unordered_map<uint64_t, uint64_t> counts;
    counts.reserve(drained.size());
    for (const auto &event : drained) {
        ++counts[event.value_];
    }
    for (const auto &[value, seen] : counts) {
        EXPECT_EQ(seen, 1u) << "duplicate identifier " << value;
    }

    std::unordered_map<uint64_t, size_t> batch_members;
    batch_members.reserve(drained.size());
    for (const auto &event : drained) {
        ++batch_members[event.value_ / kBatchSize];
    }
    for (const auto &[batch_id, members] : batch_members) {
        EXPECT_EQ(members, kBatchSize)
            << "partial batch " << batch_id << " with " << members << " members";
    }
}

} // namespace
