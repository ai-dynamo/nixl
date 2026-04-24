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

#include "io_queue.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>

#ifdef HAVE_POSIXAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size);
#endif
#ifdef HAVE_LIBURING
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size);
#endif
#ifdef HAVE_LINUXAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueLinuxAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size);
#endif

static const struct {
    const char *name;
    nixlPosixIOQueue::nixlPosixIOQueueCreateFn createFn;
} factories[] = {
#ifdef HAVE_POSIXAIO
    {"POSIXAIO", nixlPosixIOQueueAIOCreate},
#endif
#ifdef HAVE_LIBURING
    {"URING", nixlPosixIOQueueUringCreate},
#endif
#ifdef HAVE_LINUXAIO
    {"AIO", nixlPosixIOQueueLinuxAIOCreate},
#endif
};

const uint32_t nixlPosixIOQueue::MIN_IOS_POOL_SIZE = 64;
const uint32_t nixlPosixIOQueue::MAX_IOS_POOL_SIZE = 1024 * 64;
const uint32_t nixlPosixIOQueue::DEF_IOS_POOL_SIZE = nixlPosixIOQueue::MAX_IOS_POOL_SIZE;
const uint32_t nixlPosixIOQueue::MIN_KERNEL_QUEUE_SIZE = 16;
const uint32_t nixlPosixIOQueue::MAX_KERNEL_QUEUE_SIZE = 1024;
const uint32_t nixlPosixIOQueue::DEF_KERNEL_QUEUE_SIZE = 256;

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueue::instantiate(std::string_view io_queue_type,
                              uint32_t ios_pool_size,
                              uint32_t kernel_queue_size) {
    for (const auto &factory : factories) {
        if (io_queue_type == factory.name) {
            if (ios_pool_size == 0) {
                ios_pool_size = DEF_IOS_POOL_SIZE;
                NIXL_INFO << "Using default IO pool size: " << ios_pool_size;
            }
            if (kernel_queue_size == 0) {
                kernel_queue_size = DEF_KERNEL_QUEUE_SIZE;
                NIXL_INFO << "Using default kernel queue size: " << kernel_queue_size;
            }
            return factory.createFn(ios_pool_size, kernel_queue_size);
        }
    }
    return nullptr;
}

std::string_view
nixlPosixIOQueue::getDefaultIoQueueType(void) {
#ifdef HAVE_LINUXAIO
    return "AIO";
#elif defined(HAVE_LIBURING)
    return "URING";
#elif defined(HAVE_POSIXAIO)
    return "POSIXAIO";
#else
    // Should never reach here. At least one of the queues should be available.
    NIXL_ASSERT(false);
    return nullptr;
#endif
}
