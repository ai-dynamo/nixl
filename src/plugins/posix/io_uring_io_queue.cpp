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

class nixlPosixIOQueueUring : public nixlPosixIOQueueImpl<nixlPosixIoUringIO> {
public:
    nixlPosixIOQueueUring(uint32_t max_ios);

    virtual nixl_status_t
    enqueue(int fd,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) override;
    virtual ~nixlPosixIOQueueUring() override;

protected:
    nixlPosixIoUringIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    submitBatch(uint32_t to_submit, uint32_t &submitted_ios) override;
    nixl_status_t
    checkCompleted(uint32_t &completed_ios) override;

private:
    struct io_uring uring; // The io_uring instance for async I/O operations
};

nixlPosixIOQueueUring::nixlPosixIOQueueUring(uint32_t max_ios)
    : nixlPosixIOQueueImpl<nixlPosixIoUringIO>(max_ios) {
    io_uring_params params = {};
    if (io_uring_queue_init_params(max_ios_, &uring, &params) < 0) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(errno)));
    }
}

nixl_status_t
nixlPosixIOQueueUring::submitBatch(uint32_t to_submit, uint32_t &submitted_ios) {
    NIXL_ASSERT(!ios_to_submit_.empty());
    NIXL_ASSERT(ios_to_submit_.size() >= to_submit);

    for (uint32_t i = 0; i < to_submit; i++) {
        nixlPosixIoUringIO *io = ios_to_submit_.front();

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

        ios_to_submit_.pop_front();
        submitted_ios++;
    }

    int ret = io_uring_submit(&uring);
    if (ret < 0) {
        NIXL_ERROR << "io_uring_submit failed: " << nixl_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::checkCompleted(uint32_t &completed_ios) {
    struct io_uring_cqe *cqe;
    unsigned head;

    completed_ios = 0;

    io_uring_for_each_cqe(&uring, head, cqe) {
        int res = cqe->res;
        nixlPosixIoUringIO *io = reinterpret_cast<nixlPosixIoUringIO *>(io_uring_cqe_get_data(cqe));
        if (io->clb_) {
            io->clb_(io->ctx_, res, 0);
        }
        completed_ios++;
        free_ios_.push_back(io);
        if (res < 0) {
            NIXL_ERROR << absl::StrFormat("IO operation failed: %s", nixl_strerror(-res));
            return NIXL_ERR_BACKEND;
        }
        if (completed_ios == MAX_IO_CHECK_COMPLETED_BATCH_SIZE) {
            break;
        }
    }

    // Mark all seen
    io_uring_cq_advance(&uring, completed_ios);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::enqueue(int fd,
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

nixlPosixIOQueueUring::~nixlPosixIOQueueUring() {
    io_uring_queue_exit(&uring);
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t max_ios) {
    return std::make_unique<nixlPosixIOQueueUring>(max_ios);
}
