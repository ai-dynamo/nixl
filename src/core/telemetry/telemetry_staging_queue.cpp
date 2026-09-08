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
    live_.reserve(capacity_);
    drained_.reserve(capacity_);
}

bool
nixlTelemetryStagingQueue::tryPush(const nixlTelemetryEvent &event) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (live_.size() >= capacity_) {
        numDroppedEvents_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    live_.push_back(event);
    return true;
}

bool
nixlTelemetryStagingQueue::tryPushBatch(std::span<const nixlTelemetryEvent> events) {
    if (events.empty()) {
        return true;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (events.size() > capacity_ - live_.size()) {
        numDroppedEvents_.fetch_add(events.size(), std::memory_order_relaxed);
        return false;
    }
    live_.insert(live_.end(), events.begin(), events.end());
    return true;
}

std::span<const nixlTelemetryEvent>
nixlTelemetryStagingQueue::drainPending() {
    const std::lock_guard<std::mutex> lock(mutex_);
    drained_.clear();
    live_.swap(drained_);
    return drained_;
}

uint64_t
nixlTelemetryStagingQueue::takeNumDropped() noexcept {
    return numDroppedEvents_.exchange(0, std::memory_order_relaxed);
}

size_t
nixlTelemetryStagingQueue::capacity() const noexcept {
    return capacity_;
}
