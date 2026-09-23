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
#ifndef NIXL_SRC_UTILS_COMMON_BLOCKING_QUEUE_H
#define NIXL_SRC_UTILS_COMMON_BLOCKING_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <type_traits>
#include <utility>

namespace nixl {

/**
 * @class blockingQueue
 * @brief Intrusive FIFO handing items from any number of producers to
 *        consumers, with a blocking pop that a stop token can interrupt.
 */
template<typename T> class blockingQueue {
public:
    /**
     * @brief Intrusive link
     */
    struct node {
        T *next = nullptr;
    };

    blockingQueue() = default;

    blockingQueue(const blockingQueue &) = delete;
    blockingQueue &
    operator=(const blockingQueue &) = delete;

    /**
     * @brief Append an item and wake one blocked consumer
     * @param item New item that is not queued. A single item with intrusive link
     *             can be queued into at most 1 queue.
     */
    void
    push(T *item) {
        static_assert(std::is_base_of_v<node, T>,
                      "blockingQueue items must derive from blockingQueue<T>::node");
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            item->next = nullptr;
            *ptail_ = item;
            ptail_ = &item->next;
        }
        cv_.notify_one();
    }

    /**
     * @brief Pop the oldest item without blocking
     * @return The item, or nullptr if the queue is empty
     */
    [[nodiscard]] T *
    tryPop() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return popLocked();
    }

    /**
     * @brief Pop the oldest item, blocking until one is queued or a stop is requested
     * @return The item, or nullptr if stopped with the queue empty
     */
    [[nodiscard]] T *
    pop(std::stop_token token = {}) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, std::move(token), [this] { return head_ != nullptr; });
        return popLocked();
    }

private:
    [[nodiscard]] T *
    popLocked() noexcept {
        T *const item = head_;
        if (item == nullptr) {
            return nullptr;
        }
        head_ = item->next;
        if (ptail_ == &item->next) {
            ptail_ = &head_;
        }
        return item;
    }

    std::mutex mutex_;
    std::condition_variable_any cv_;
    T *head_ = nullptr;
    T **ptail_ = &head_;
};

} // namespace nixl

#endif // NIXL_SRC_UTILS_COMMON_BLOCKING_QUEUE_H
