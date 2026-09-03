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

#ifndef POSIX_IO_QUEUE_H
#define POSIX_IO_QUEUE_H

#include <stdint.h>
#include <list>
#include <memory>
#include <vector>
#include <functional>
#include "backend_aux.h"

using nixlPosixIOQueueDoneCb = std::function<void(void *ctx, uint32_t data_size, int error)>;
using nixlPosixIOQueueCancelDoneCb = std::function<void(void *ctx)>;

class nixlPosixIOQueue {
public:
    using nixlPosixIOQueueCreateFn =
        std::function<std::unique_ptr<nixlPosixIOQueue>(uint32_t ios_pool_size,
                                                        uint32_t kernel_queue_size)>;

    nixlPosixIOQueue(uint32_t ios_pool_size, uint32_t kernel_queue_size)
        : ios_pool_size_(normalizedIOSPoolSize(ios_pool_size)),
          kernel_queue_size_(normalizedKernelQueueSize(kernel_queue_size)) {}

    virtual ~nixlPosixIOQueue() {}

    virtual nixl_status_t
    enqueue(int fd,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) = 0;
    virtual nixl_status_t
    post(void) = 0;
    virtual nixl_status_t
    poll(void) = 0;

    /** Cancel I/Os owned by @p ctx synchronously when possible; return the number of async
     * cancellations that will invoke @p clb. */
    virtual unsigned
    cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) {
        return 0;
    }

    static std::unique_ptr<nixlPosixIOQueue>
    instantiate(std::string_view io_queue_type, uint32_t ios_pool_size, uint32_t kernel_queue_size);
    static std::string_view
    getDefaultIoQueueType(void);

    static constexpr uint32_t MIN_IOS_POOL_SIZE = 64;
    static constexpr uint32_t MAX_IOS_POOL_SIZE = 1024 * 64;
    static constexpr uint32_t DEF_IOS_POOL_SIZE = MAX_IOS_POOL_SIZE;
    static constexpr uint32_t MIN_KERNEL_QUEUE_SIZE = 16;
    static constexpr uint32_t MAX_KERNEL_QUEUE_SIZE = 1024;
    static constexpr uint32_t DEF_KERNEL_QUEUE_SIZE = 256;

protected:
    static uint32_t
    normalizedIOSPoolSize(uint32_t ios_pool_size) {
        return std::clamp(ios_pool_size, MIN_IOS_POOL_SIZE, MAX_IOS_POOL_SIZE);
    }

    static uint32_t
    normalizedKernelQueueSize(uint32_t kernel_queue_size) {
        return std::clamp(kernel_queue_size, MIN_KERNEL_QUEUE_SIZE, MAX_KERNEL_QUEUE_SIZE);
    }

    uint32_t ios_pool_size_;
    uint32_t kernel_queue_size_;
};

template<typename Entry> class nixlPosixIOQueueImpl : public nixlPosixIOQueue {
public:
    nixlPosixIOQueueImpl(uint32_t ios_pool_size, uint32_t kernel_queue_size)
        : nixlPosixIOQueue(ios_pool_size, kernel_queue_size),
          ios_(ios_pool_size_) {
        for (uint32_t i = 0; i < ios_pool_size_; i++) {
            free_ios_.push_back(&ios_[i]);
        }
    }

protected:
    std::vector<Entry> ios_;
    std::list<Entry *> free_ios_;
    std::list<Entry *> ios_to_submit_;
};

#endif // POSIX_IO_QUEUE_H
