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
#include <libaio.h>
#include "common/nixl_log.h"
#include <algorithm>
#include <absl/strings/str_format.h>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

struct nixlPosixLinuxAioIO {
public:
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_;
    struct iocb io_;
};

class nixlPosixIOQueueLinuxAIO : public nixlPosixIOQueueImpl<nixlPosixLinuxAioIO> {
public:
    nixlPosixIOQueueLinuxAIO(uint32_t ios_pool_size, uint32_t kernel_queue_size);

    virtual nixl_status_t
    post(void) override;
    virtual nixl_status_t
    enqueue(int fd,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) override;
    virtual nixl_status_t
    poll(void) override;
    virtual ~nixlPosixIOQueueLinuxAIO() override;

protected:
    nixlPosixLinuxAioIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    doCheckCompleted(void);

private:
    io_context_t io_ctx_; // I/O context
};

nixlPosixIOQueueLinuxAIO::nixlPosixIOQueueLinuxAIO(uint32_t ios_pool_size,
                                                   uint32_t kernel_queue_size)
    : nixlPosixIOQueueImpl<nixlPosixLinuxAioIO>(ios_pool_size, kernel_queue_size) {
    int res = io_queue_init(kernel_queue_size_, &io_ctx_);
    if (res) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_queue: %s", nixl_strerror(errno)));
    }
}

nixl_status_t
nixlPosixIOQueueLinuxAIO::enqueue(int fd,
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

nixlPosixIOQueueLinuxAIO::~nixlPosixIOQueueLinuxAIO() {
    io_queue_release(io_ctx_);
}

// Note: post() must return NIXL_IN_PROG in case of success
nixl_status_t
nixlPosixIOQueueLinuxAIO::post(void) {
    struct iocb *ios[MAX_IO_SUBMIT_BATCH_SIZE];
    nixlPosixLinuxAioIO *to_submit[MAX_IO_SUBMIT_BATCH_SIZE];

    if (ios_to_submit_.empty()) {
        return NIXL_IN_PROG; // No blocks to submit
    }

    int num_ios = std::min(MAX_IO_SUBMIT_BATCH_SIZE, (int)ios_to_submit_.size());
    for (int i = 0; i < num_ios; i++) {
        nixlPosixLinuxAioIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        ios[i] = &io->io_;
        to_submit[i] = io;
    }

    int ret = io_submit(io_ctx_, num_ios, ios);
    if (ret < 0) {
        if (ret == -EAGAIN) {
            ret = 0; // 0 were submitted, we will try again later
        } else {
            NIXL_ERROR << "io_submit failed: " << nixl_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }
    }

    for (int i = num_ios - 1; i >= ret; i--) {
        // If not submitted, push back to the front of the list
        nixlPosixLinuxAioIO *io = to_submit[i];
        ios_to_submit_.push_front(io);
    }

    return NIXL_IN_PROG;
}

inline nixl_status_t
nixlPosixIOQueueLinuxAIO::doCheckCompleted(void) {
    struct io_event events[MAX_IO_CHECK_COMPLETED_BATCH_SIZE];
    std::list<nixlPosixLinuxAioIO *> completed_ios;
    int rc;
    struct timespec timeout = {0, 0};

    if (free_ios_.size() == ios_pool_size_) {
        return NIXL_SUCCESS; // All ios are free, no ios in flight
    }

    rc = io_getevents(io_ctx_, 0, MAX_IO_CHECK_COMPLETED_BATCH_SIZE, events, &timeout);
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

        completed_ios.push_back(io);
    }

    if (!completed_ios.empty()) {
        free_ios_.splice(free_ios_.end(), completed_ios);
    }

    if (free_ios_.size() == ios_pool_size_) {
        return NIXL_SUCCESS; // All ios are free now
    }

    return NIXL_IN_PROG; // Some blocks are in flight, need to check again
}

nixl_status_t
nixlPosixIOQueueLinuxAIO::poll(void) {
    nixl_status_t status = post();
    if (status < 0) {
        return status;
    }

    return doCheckCompleted();
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueLinuxAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueLinuxAIO>(ios_pool_size, kernel_queue_size);
}
