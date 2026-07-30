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
#ifndef NIXL_SPSC_QUEUE_H
#define NIXL_SPSC_QUEUE_H

#include <atomic>
#include <optional>
#include <new>

template<typename T, size_t Capacity> class spscQueue {
private:
    T buffer_[Capacity];

    // C++17: Automatically align to the target architecture's cache line size
    // Fallback to 64 if the compiler doesn't support the macro yet
    /*#ifdef __cpp_lib_hardware_interference_size
        static constexpr size_t CacheLineSize = std::hardware_destructive_interference_size;
    #else
        static constexpr size_t CacheLineSize = 64;
    #endif*/

    /* TODO-Eyal: I commented out the above because it fails on werror */
    static constexpr size_t cacheLineSize = 64;

    alignas(cacheLineSize) std::atomic<size_t> head_{0};
    alignas(cacheLineSize) std::atomic<size_t> tail_{0};

public:
    bool
    push(const T &item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t next_head = (current_head + 1) % Capacity;

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    std::optional<T>
    tryPop() {
        size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt; // Queue is empty
        }

        T item = std::move(buffer_[current_tail]);

        tail_.store((current_tail + 1) % Capacity, std::memory_order_release);

        return item;
    }
};

static constexpr size_t spsc_size = 64;

#endif // NIXL_SPSC_QUEUE_H
