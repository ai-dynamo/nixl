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
#include <liburing.h>
#include <absl/strings/str_format.h>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

struct nixlPosixIoUringIO {
public:
    int fd;
    void *buf_;
    size_t len_;
    off_t offset_;
    bool read_;
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_;
    struct io_uring_sqe *sqe_;
};

class nixlPosixIoUringIOQueue : public nixlPosixIOQueueImpl<nixlPosixIoUringIO> {
public:
    nixlPosixIoUringIOQueue(uint32_t max_ios);

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
    virtual ~nixlPosixIoUringIOQueue() override;

protected:
    nixlPosixIoUringIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    doSubmit(void);
    nixl_status_t
    doCheckCompleted(void);

private:
    struct io_uring uring; // The io_uring instance for async I/O operations
};

nixlPosixIoUringIOQueue::nixlPosixIoUringIOQueue(uint32_t max_ios) : nixlPosixIOQueueImpl<nixlPosixIoUringIO>(max_ios) {
    io_uring_params params = {};
    if (io_uring_queue_init_params(max_ios_, &uring, &params) < 0) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(errno)));
    }
}

nixl_status_t
nixlPosixIoUringIOQueue::doSubmit(void) {
    if (ios_to_submit_.empty()) {
        return NIXL_SUCCESS; // No blocks to submit
    }

    int num_ios = std::min(MAX_IO_SUBMIT_BATCH_SIZE, (int)ios_to_submit_.size());
    for (int i = 0; i < num_ios; i++) {
        nixlPosixIoUringIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        struct io_uring_sqe *sqe = io_uring_get_sqe(&uring);
        if (!sqe) {
            NIXL_ERROR << "Failed to get io_uring submission queue entry";
            return NIXL_ERR_BACKEND;
        }

        if (io->read_) {
            io_uring_prep_read(sqe, io->fd, io->buf_, io->len_, io->offset_);
        } else {
            io_uring_prep_write(sqe, io->fd, io->buf_, io->len_, io->offset_);
        }

        io_uring_sqe_set_data(sqe, io);
    }

    int ret = io_uring_submit(&uring);
    if (ret < 0) {
        NIXL_ERROR << "io_uring_submit failed: " << nixl_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    return ios_to_submit_.empty() ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlPosixIoUringIOQueue::doCheckCompleted(void) {
    struct io_uring_cqe *cqe;
    unsigned head;
    int count = 0;
    io_uring_for_each_cqe(&uring, head, cqe) {
        int res = cqe->res;
        nixlPosixIoUringIO *io = reinterpret_cast<nixlPosixIoUringIO *>(io_uring_cqe_get_data(cqe));
        if (io->clb_) {
            io->clb_(io->ctx_, res, 0);
        }
        free_ios_.push_back(io);
        if (res < 0) {
            NIXL_ERROR << absl::StrFormat("IO operation failed: %s", nixl_strerror(-res));
            return NIXL_ERR_BACKEND;
        }
        count++;
        if (count == MAX_IO_CHECK_COMPLETED_BATCH_SIZE) {
            break;
        }
    }

    // Mark all seen
    io_uring_cq_advance(&uring, count);

    if (free_ios_.size() == max_ios_) {
        return NIXL_SUCCESS; // All ios are free now
    }

    return NIXL_IN_PROG; // Some ios are in flight, need to check again
}

nixl_status_t
nixlPosixIoUringIOQueue::enqueue(int fd,
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

    nixlPosixIoUringIO *io = free_ios_.front();
    free_ios_.pop_front();
    io->fd = fd;
    io->buf_ = buf;
    io->len_ = len;
    io->offset_ = offset;
    io->read_ = read;
    io->clb_ = clb;
    io->ctx_ = ctx;

    ios_to_submit_.push_back(io);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIoUringIOQueue::post(void) {
    nixl_status_t status = doSubmit();
    if (status < 0) {
        return status;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlPosixIoUringIOQueue::poll(void) {
    nixl_status_t status = post();
    if (status < 0) {
        return status;
    }

    return doCheckCompleted();
}

nixlPosixIoUringIOQueue::~nixlPosixIoUringIOQueue() {
    io_uring_queue_exit(&uring);
}

std::unique_ptr<nixlPosixIOQueue> nixlPosixIoUringIOQueueCreate(uint32_t max_ios) {
    return std::make_unique<nixlPosixIoUringIOQueue>(max_ios);
}
