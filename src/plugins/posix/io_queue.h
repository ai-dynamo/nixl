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

class nixlPosixIOQueue {
public:
    using nixlPosixIOQueueCreateFn =
        std::function<std::unique_ptr<nixlPosixIOQueue>(uint32_t max_ios)>;

    nixlPosixIOQueue(uint32_t max_ios) : max_ios_(normalizedMaxIOS(max_ios)) {}

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

    static std::unique_ptr<nixlPosixIOQueue>
    instantiate(std::string_view io_queue_type, uint32_t max_ios);
    static std::string_view
    getDefaultIoQueueType(void);

protected:
    static uint32_t
    normalizedMaxIOS(uint32_t max_ios) {
        return std::clamp(max_ios, MIN_IOS, MAX_IOS);
    }

    uint32_t max_ios_;
    static const uint32_t MIN_IOS;
    static const uint32_t MAX_IOS;
};

template<typename Entry> class nixlPosixIOQueueImpl : public nixlPosixIOQueue {
public:
    nixlPosixIOQueueImpl(uint32_t max_ios) : nixlPosixIOQueue(max_ios), ios_(max_ios_) {
        for (uint32_t i = 0; i < max_ios_; i++) {
            free_ios_.push_back(&ios_[i]);
        }
    }

protected:
    std::vector<Entry> ios_;
    std::list<Entry *> free_ios_;
    std::list<Entry *> ios_to_submit_;
};


#endif // POSIX_IO_QUEUE_H
