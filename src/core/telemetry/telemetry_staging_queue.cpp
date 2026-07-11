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
#include "telemetry_staging_queue.h"

nixlTelemetryStagingQueue::nixlTelemetryStagingQueue(size_t capacity) : capacity_(capacity) {
    events_.reserve(capacity_);
}

bool
nixlTelemetryStagingQueue::tryPush(const nixlTelemetryEvent &event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= capacity_) {
        droppedEvents_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    events_.push_back(event);
    return true;
}

bool
nixlTelemetryStagingQueue::tryPushBatch(const nixlTelemetryEvent *events, size_t count) {
    if (count == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() + count > capacity_) {
        droppedEvents_.fetch_add(count, std::memory_order_relaxed);
        return false;
    }
    events_.insert(events_.end(), events, events + count);
    return true;
}

std::vector<nixlTelemetryEvent>
nixlTelemetryStagingQueue::takePending() {
    std::vector<nixlTelemetryEvent> pending;
    pending.reserve(capacity_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.swap(pending);
    }
    return pending;
}

uint64_t
nixlTelemetryStagingQueue::exchangeDropped() noexcept {
    return droppedEvents_.exchange(0, std::memory_order_relaxed);
}

size_t
nixlTelemetryStagingQueue::capacity() const noexcept {
    return capacity_;
}
