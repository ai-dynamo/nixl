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

#include "io_queue.h"
#include <libaio.h>
#include "common/nixl_log.h"
#include <algorithm>
#include <absl/strings/str_format.h>

struct nixlPosixLinuxAioIO {
public:
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_;
    struct iocb io_;
};

class nixlPosixLinuxAioIOQueue : public nixlPosixIOQueueImpl<nixlPosixLinuxAioIO> {
public:
    nixlPosixLinuxAioIOQueue(uint32_t max_ios);

    virtual nixl_status_t
    enqueue(int fd,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) override;
    virtual ~nixlPosixLinuxAioIOQueue() override;

protected:
    nixlPosixLinuxAioIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    submitBatch(uint32_t to_submit, uint32_t &submitted_ios) override;
    nixl_status_t
    checkCompleted(uint32_t &completed_ios) override;

private:
    io_context_t io_ctx_; // I/O context
};

nixlPosixLinuxAioIOQueue::nixlPosixLinuxAioIOQueue(uint32_t max_ios)
    : nixlPosixIOQueueImpl<nixlPosixLinuxAioIO>(max_ios) {
    int res = io_queue_init(max_ios_, &io_ctx_);
    if (res) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_queue: %s", nixl_strerror(errno)));
    }
}

nixl_status_t
nixlPosixLinuxAioIOQueue::enqueue(int fd,
                                  void *buf,
                                  size_t len,
                                  off_t offset,
                                  bool read,
                                  nixlPosixIOQueueDoneCb clb,
                                  void *ctx) {
    if (free_ios_.empty()) {
        NIXL_ERROR << "No more free blocks available";
        return NIXL_ERR_NOT_ALLOWED;
    }
    nixlPosixLinuxAioIO *io = free_ios_.front();
    free_ios_.pop_front();

    if (read) {
        io_prep_pread(&io->io_, fd, buf, len, offset);
    } else {
        io_prep_pwrite(&io->io_, fd, buf, len, offset);
    }
    io->clb_ = clb;
    io->ctx_ = ctx;
    io->io_.data = io;
    ios_to_submit_.push_back(io);

    return NIXL_SUCCESS;
}

nixlPosixLinuxAioIOQueue::~nixlPosixLinuxAioIOQueue() {
    io_queue_release(io_ctx_);
}

nixl_status_t
nixlPosixLinuxAioIOQueue::submitBatch(uint32_t to_submit, uint32_t &submitted_ios) {
    struct iocb *ios[MAX_OUTSTANDING_IOS];
    nixlPosixLinuxAioIO *ios_to_submit[MAX_OUTSTANDING_IOS];

    assert(!ios_to_submit_.empty());
    assert(ios_to_submit_.size() >= to_submit);

    for (uint32_t i = 0; i < to_submit; i++) {
        nixlPosixLinuxAioIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        ios[i] = &io->io_;
        ios_to_submit[i] = io;
    }

    int ret = io_submit(io_ctx_, to_submit, ios);
    if (ret < 0) {
        if (ret == -EAGAIN) {
            ret = 0; // 0 were submitted, we will try again later
        } else {
            NIXL_ERROR << "io_submit failed: " << nixl_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }
    }

    submitted_ios = ret;

    for (int i = to_submit - 1; i >= ret; i--) {
        // If not submitted, push back to the front of the list
        nixlPosixLinuxAioIO *io = ios_to_submit[i];
        ios_to_submit_.push_front(io);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixLinuxAioIOQueue::checkCompleted(uint32_t &completed_ios) {
    struct io_event events[MAX_OUTSTANDING_IOS];
    int rc;
    struct timespec timeout = {0, 0};

    completed_ios = 0;

    rc = io_getevents(io_ctx_, 0, MAX_OUTSTANDING_IOS, events, &timeout);
    if (rc < 0) {
        NIXL_ERROR << "io_getevents error: " << rc;
        return NIXL_ERR_BACKEND;
    }

    for (int i = 0; i < rc; i++) {
        struct iocb *iocb = events[i].obj;
        nixlPosixLinuxAioIO *io = (nixlPosixLinuxAioIO *)iocb->data;

        if (events[i].res < 0) {
            NIXL_ERROR << "AIO operation failed: " << events[i].res;
            return NIXL_ERR_BACKEND;
        }

        if (io->clb_) {
            io->clb_(io->ctx_, events[i].res, 0);
        }

        free_ios_.push_back(io);
        completed_ios++;
    }

    return NIXL_SUCCESS;
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixLinuxAioIOQueueCreate(uint32_t max_ios) {
    return std::make_unique<nixlPosixLinuxAioIOQueue>(max_ios);
}
