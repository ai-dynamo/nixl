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
#include "common/nixl_log.h"

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
    static const uint32_t MAX_OUTSTANDING_IOS;
};

template<typename Entry> class nixlPosixIOQueueImpl : public nixlPosixIOQueue {
public:
    nixlPosixIOQueueImpl(uint32_t max_ios)
        : nixlPosixIOQueue(max_ios),
          ios_(max_ios_),
          num_ios_outstanding_(0) {
        for (uint32_t i = 0; i < max_ios_; i++) {
            free_ios_.push_back(&ios_[i]);
        }
    }

    virtual nixl_status_t
    post(void) override {
        return submit();
    }

    virtual nixl_status_t
    poll(void) override {
        nixl_status_t status = submit();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        if (num_ios_outstanding_ == 0) {
            return NIXL_SUCCESS; // No blocks in flight
        }

        return complete();
    }

protected:
    nixl_status_t
    submit(void) {
        if (ios_to_submit_.empty()) {
            return NIXL_SUCCESS; // No blocks to submit
        }

        NIXL_ASSERT(num_ios_outstanding_ <= MAX_OUTSTANDING_IOS);

        // Calculate how many more we can submit to reach target outstanding
        size_t slots_available = MAX_OUTSTANDING_IOS - num_ios_outstanding_;
        if (slots_available == 0) {
            return NIXL_SUCCESS; // No slots available
        }

        uint32_t to_submit = std::min(ios_to_submit_.size(), slots_available);
        uint32_t submitted_ios = 0;

        nixl_status_t status = submitBatch(to_submit, submitted_ios);
        num_ios_outstanding_ += submitted_ios;

        return status;
    }

    nixl_status_t
    complete(void) {
        uint32_t completed_ios = 0;
        nixl_status_t status = checkCompleted(completed_ios);
        if (status != NIXL_SUCCESS) {
            return status;
        }

        NIXL_ASSERT(completed_ios <= num_ios_outstanding_);
        num_ios_outstanding_ -= completed_ios;

        return NIXL_SUCCESS;
    }

    virtual nixl_status_t
    submitBatch(uint32_t to_submit, uint32_t &submitted_ios) = 0;
    virtual nixl_status_t
    checkCompleted(uint32_t &completed_ios) = 0;

    std::vector<Entry> ios_;
    std::list<Entry *> free_ios_;
    std::list<Entry *> ios_to_submit_;

private:
    uint32_t num_ios_outstanding_;
};


#endif // POSIX_IO_QUEUE_H
