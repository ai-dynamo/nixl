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
#include <aio.h>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

struct nixlPosixAioIO {
public:
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_;
    struct aiocb aio_;
    bool read_;
};

class nixlPosixIOQueueAIO : public nixlPosixIOQueueImpl<nixlPosixAioIO> {
public:
    nixlPosixIOQueueAIO(uint32_t ios_pool_size, uint32_t kernel_queue_size)
        : nixlPosixIOQueueImpl<nixlPosixAioIO>(ios_pool_size, kernel_queue_size) {}

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
    virtual ~nixlPosixIOQueueAIO() override;

protected:
    nixl_status_t
    doCheckCompleted(void);

    std::list<nixlPosixAioIO *> ios_in_flight_;
};

nixlPosixIOQueueAIO::~nixlPosixIOQueueAIO() {
    for (auto &io : ios_) {
        if (io.aio_.aio_fildes != 0) {
            aio_cancel(io.aio_.aio_fildes, &io.aio_);
        }
    }
}

nixl_status_t
nixlPosixIOQueueAIO::enqueue(int fd,
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

    nixlPosixAioIO *io = free_ios_.front();
    free_ios_.pop_front();

    io->clb_ = clb;
    io->ctx_ = ctx;
    io->read_ = read;
    io->aio_.aio_fildes = fd;
    io->aio_.aio_buf = buf;
    io->aio_.aio_nbytes = len;
    io->aio_.aio_offset = offset;

    ios_to_submit_.push_back(io);

    return NIXL_SUCCESS;
}

// Note: post() must return NIXL_IN_PROG in case of success
nixl_status_t
nixlPosixIOQueueAIO::post(void) {
    if (ios_to_submit_.empty()) {
        return NIXL_IN_PROG; // No blocks to submit
    }

    int num_ios = std::min(MAX_IO_SUBMIT_BATCH_SIZE, (int)ios_to_submit_.size());
    for (int i = 0; i < num_ios; i++) {
        nixlPosixAioIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        int ret;
        if (io->read_) {
            ret = aio_read(&io->aio_);
        } else {
            ret = aio_write(&io->aio_);
        }

        if (ret < 0) {
            NIXL_ERROR << "aio_submit failed: " << nixl_strerror(-ret);
            ios_to_submit_.push_front(io);
            return NIXL_ERR_BACKEND;
        }

        ios_in_flight_.push_back(io);
    }

    return NIXL_IN_PROG;
}

inline nixl_status_t
nixlPosixIOQueueAIO::doCheckCompleted(void) {
    if (ios_in_flight_.empty()) {
        return NIXL_SUCCESS; // No blocks in flight
    }

    int num_ios = std::min(MAX_IO_CHECK_COMPLETED_BATCH_SIZE, (int)ios_in_flight_.size());
    for (auto it = ios_in_flight_.begin(); it != ios_in_flight_.end();) {
        nixlPosixAioIO *io = *it;
        int status = aio_error(&io->aio_);
        if (status == 0) {
            ssize_t ret = aio_return(&io->aio_);
            if (ret < 0 || ret != static_cast<ssize_t>(io->aio_.aio_nbytes)) {
                NIXL_ERROR << "aio_return failed: " << nixl_strerror(-ret);
                ios_in_flight_.push_front(io);
                return NIXL_ERR_BACKEND;
            }
            if (io->clb_) {
                io->clb_(io->ctx_, ret, 0);
            }
            it = ios_in_flight_.erase(it);
            free_ios_.push_back(io);
        } else if (status == EINPROGRESS) {
            return NIXL_IN_PROG;
        } else {
            NIXL_ERROR << "aio_error failed: " << nixl_strerror(-status);
            ios_in_flight_.push_front(io);
            return NIXL_ERR_BACKEND;
        }

        it++;

        num_ios--;
        if (num_ios == 0) {
            break;
        }
    }

    return ios_in_flight_.empty() ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlPosixIOQueueAIO::poll(void) {
    nixl_status_t status = post();
    if (status < 0) {
        return status;
    }

    return doCheckCompleted();
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueAIO>(ios_pool_size, kernel_queue_size);
}
