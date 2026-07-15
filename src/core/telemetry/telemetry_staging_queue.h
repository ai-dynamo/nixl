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
#ifndef NIXL_SRC_CORE_TELEMETRY_TELEMETRY_STAGING_QUEUE_H
#define NIXL_SRC_CORE_TELEMETRY_TELEMETRY_STAGING_QUEUE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "telemetry_event.h"

/**
 * @brief Bounded multi-producer/single-consumer staging queue for telemetry
 *        events.
 *
 * Owns the queue mechanics that used to live inline in nixlTelemetry: the event
 * storage and its capacity reserve, the producer/consumer mutex, capacity
 * enforcement, all-or-none single and batch insertion, the swap-drain, and the
 * staging-drop counter. Metric semantics (activation, event selection, exporter
 * dispatch, synthetic dropped-event construction) stay in nixlTelemetry.
 *
 * Drop policy is drop-newest: once the queue is full an incoming event or batch
 * is rejected and counted as dropped; already-queued events are never
 * overwritten.
 */
class nixlTelemetryStagingQueue {
public:
    /**
     * @brief Construct a queue that retains at most @p capacity events.
     *
     * Reserves storage for @p capacity events up front so the producer path
     * never reallocates. Any @p capacity is accepted (a zero capacity yields a
     * queue that drops every push); rejecting a zero telemetry buffer size is
     * the caller's responsibility (see nixlTelemetry construction).
     * @param capacity Maximum number of events retained before pushes are dropped.
     */
    explicit nixlTelemetryStagingQueue(size_t capacity);

    /**
     * @brief Append a single event if there is room.
     *
     * The capacity check and insertion happen under one mutex acquisition. On a
     * full queue the event is rejected and the staging-drop count is incremented
     * by one.
     * @return true if the event was queued, false if rejected (dropped).
     */
    bool
    tryPush(const nixlTelemetryEvent &event);

    /**
     * @brief Append a batch of events all-or-none if the whole batch fits.
     *
     * The capacity check and insertion happen under one mutex acquisition. The
     * batch is accepted completely or rejected completely; on rejection the
     * staging-drop count is incremented by @p count. A zero-length batch succeeds
     * without locking or changing drop accounting.
     * @param events Pointer to @p count contiguous events.
     * @param count Number of events in the batch.
     * @return true if the batch was queued (or empty), false if rejected.
     */
    bool
    tryPushBatch(const nixlTelemetryEvent *events, size_t count);

    /**
     * @brief Swap the live queue with an empty capacity-reserved vector and
     *        return the drained events in insertion order.
     *
     * The swap happens under the mutex; producers can write to the fresh live
     * vector while the consumer processes the returned one.
     */
    std::vector<nixlTelemetryEvent>
    takePending();

    /**
     * @brief Atomically take and reset the staging drops accumulated since the
     *        previous call.
     */
    uint64_t
    exchangeDropped() noexcept;

    [[nodiscard]] size_t
    capacity() const noexcept;

private:
    const size_t capacity_;
    std::vector<nixlTelemetryEvent> events_;
    std::mutex mutex_;
    std::atomic<uint64_t> droppedEvents_{0};
};

#endif
