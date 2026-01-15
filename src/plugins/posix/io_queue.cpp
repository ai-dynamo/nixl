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

#ifdef HAVE_LIBAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixAioIOQueueCreate(uint32_t max_ios);
#endif
#ifdef HAVE_LIBURING
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIoUringIOQueueCreate(uint32_t max_ios);
#endif
#ifdef HAVE_LINUXAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixLinuxAioIOQueueCreate(uint32_t max_ios);
#endif

static const struct {
    const char *name;
    nixlPosixIOQueue::nixlPosixIOQueueCreateFn createFn;
} factories[] = {
#ifdef HAVE_LIBAIO
    {"POSIXAIO", nixlPosixAioIOQueueCreate},
#endif
#ifdef HAVE_LIBURING
    {"URING", nixlPosixIoUringIOQueueCreate},
#endif
#ifdef HAVE_LINUXAIO
    {"AIO", nixlPosixLinuxAioIOQueueCreate},
#endif
};

const uint32_t nixlPosixIOQueue::MIN_IOS = 64;
const uint32_t nixlPosixIOQueue::MAX_IOS = 1024 * 64;
const uint32_t nixlPosixIOQueue::MAX_OUTSTANDING_IOS = 16;

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueue::instantiate(std::string_view io_queue_type, uint32_t max_ios) {
    for (const auto &factory : factories) {
        if (io_queue_type == factory.name) {
            return factory.createFn(max_ios);
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
#elif defined(HAVE_LIBAIO)
    return "POSIXAIO";
#else
    // Should never reach here. At least one of the queues should be available.
    NIXL_ASSERT(false);
    return nullptr;
#endif
}
