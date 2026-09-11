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

#include <limits>

#ifdef HAVE_POSIXAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueAIOCreate(uint32_t ios_pool_size,
                          uint32_t kernel_queue_size,
                          bool open_synchronous);
#endif
#ifdef HAVE_LIBURING
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t ios_pool_size,
                            uint32_t kernel_queue_size,
                            bool open_synchronous);
#endif
#ifdef HAVE_LINUXAIO
std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueLinuxAIOCreate(uint32_t ios_pool_size,
                               uint32_t kernel_queue_size,
                               bool open_synchronous);
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

nixl_status_t
nixlPosixIOQueue::registerFile(uint64_t dev_id, const std::string &meta_info) {
    const bool path_mode = nixl::parsePathMeta(meta_info).has_value();
    if (!path_mode && dev_id > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto file = files_.find(dev_id);
    if (file != files_.end()) {
        if (path_mode || file->second.pathMode) {
            return NIXL_ERR_INVALID_PARAM;
        }
        file->second.registrations++;
        return NIXL_SUCCESS;
    }

    files_.try_emplace(dev_id, dev_id, meta_info, path_mode);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueue::deregisterFile(uint64_t dev_id) {
    auto file = files_.find(dev_id);
    if (file == files_.end()) {
        return NIXL_SUCCESS;
    }
    if (--file->second.registrations == 0) {
        files_.erase(file);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueue::enqueue(uint64_t dev_id,
                          void *buf,
                          size_t len,
                          off_t offset,
                          bool read,
                          nixlPosixIOQueueDoneCb clb,
                          void *ctx) {
    auto file = files_.find(dev_id);
    if (file == files_.end()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return enqueueFd(file->second.fileFd.fd(), buf, len, offset, read, std::move(clb), ctx);
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueue::instantiate(std::string_view io_queue_type,
                              uint32_t ios_pool_size,
                              uint32_t kernel_queue_size,
                              bool open_synchronous) {
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
            return factory.createFn(ios_pool_size, kernel_queue_size, open_synchronous);
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
