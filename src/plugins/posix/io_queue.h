/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <map>
#include <string>
#include <memory>
#include "backend_aux.h"

typedef void (*nixlPosixIOQueueDoneCb)(void *ctx, uint32_t data_size, int error);

class nixlPosixIOQueue {
public:
    typedef std::unique_ptr<nixlPosixIOQueue> (*nixlPosixIOQueueCreateFn)(void);

    nixlPosixIOQueue() {}

    virtual ~nixlPosixIOQueue() {}

    virtual nixl_status_t
    init(uint32_t max_ios) = 0;
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
    getApi(const std::string &api_name) {
        auto it = apis.find(api_name);
        if (it != apis.end()) {
            nixlPosixIOQueueCreateFn create_fn = it->second;
            return create_fn();
        }
        return nullptr;
    }

    static void
    registerIOQueue(const std::string &api_name, nixlPosixIOQueueCreateFn create_fn) {
        apis[api_name] = create_fn;
    }

    static uint32_t
    normalizedMaxIOS(uint32_t max_ios) {
        uint32_t m = std::max(MIN_IOS, max_ios);
        m = std::min(m, MAX_IOS);
        return m;
    }

protected:
    static std::map<std::string, nixlPosixIOQueueCreateFn> apis;
    static const uint32_t MIN_IOS;
    static const uint32_t MAX_IOS;
};

#define NIXL_POSIX_IO_QUEUE_REGISTER(io_queue_name, class_name)               \
    static std::unique_ptr<nixlPosixIOQueue> class_name##Create(void) {       \
        return std::make_unique<class_name>();                                \
    }                                                                         \
    static const bool registered = []() {                                     \
        nixlPosixIOQueue::registerIOQueue(io_queue_name, class_name##Create); \
        return true;                                                          \
    }();

#endif // POSIX_IO_QUEUE_H
