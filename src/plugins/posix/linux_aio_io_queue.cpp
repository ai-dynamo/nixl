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
#include <cerrno>
#include <absl/strings/str_format.h>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

struct nixlPosixLinuxAioIO {
public:
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_;
    struct iocb io_;
    bool in_flight_ = false;
    bool force_error_ = false;
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
    virtual nixl_status_t
    cancel(void *ctx) override;
    virtual bool
    hasPendingCleanup(void) const override {
        return cancellations_pending_ != 0;
    }

    virtual ~nixlPosixIOQueueLinuxAIO() override;

protected:
    nixlPosixLinuxAioIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    doCheckCompleted(void);

private:
    io_context_t io_ctx_; // I/O context
    unsigned cancellations_pending_ = 0;
};

nixlPosixIOQueueLinuxAIO::nixlPosixIOQueueLinuxAIO(uint32_t ios_pool_size,
                                                   uint32_t kernel_queue_size)
    : nixlPosixIOQueueImpl<nixlPosixLinuxAioIO>(ios_pool_size, kernel_queue_size) {
    int res = io_queue_init(kernel_queue_size_, &io_ctx_);
    if (res) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_queue: %s", nixl_strerror(-res)));
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
    io->in_flight_ = false;
    io->force_error_ = false;
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
    bool submission_failed = false;
    if (ret < 0) {
        if (ret == -EAGAIN || ret == -EINTR) {
            ret = 0; // 0 were submitted, we will try again later
        } else {
            NIXL_ERROR << "io_submit failed: " << nixl_strerror(-ret);
            ret = 0;
            submission_failed = true;
        }
    }

    for (int i = 0; i < ret; i++) {
        to_submit[i]->in_flight_ = true;
    }
    for (int i = num_ios - 1; i >= ret; i--) {
        // If not submitted, push back to the front of the list
        nixlPosixLinuxAioIO *io = to_submit[i];
        ios_to_submit_.push_front(io);
    }

    return submission_failed ? NIXL_ERR_BACKEND : NIXL_IN_PROG;
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

        // io_event.res is unsigned long. Interpret it as signed before checking
        // for a negative errno, and reject short completions as well.
        long res = static_cast<long>(events[i].res);
        bool was_cancellation_pending = io->force_error_;
        int error = was_cancellation_pending || res < 0 ||
                    static_cast<unsigned long>(res) != iocb->u.c.nbytes;
        if (error) {
            NIXL_DEBUG << absl::StrFormat(
                "AIO operation incomplete: result %ld, expected %lu", res, iocb->u.c.nbytes);
        }
        if (io->clb_) {
            io->clb_(io->ctx_, error ? 0 : static_cast<uint32_t>(res), error);
        }
        io->in_flight_ = false;
        io->force_error_ = false;
        if (was_cancellation_pending) {
            NIXL_ASSERT(cancellations_pending_ > 0);
            cancellations_pending_--;
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
nixlPosixIOQueueLinuxAIO::cancel(void *ctx) {
    if (!ctx) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // I/Os that have not reached the kernel can be failed and reclaimed now.
    for (auto it = ios_to_submit_.begin(); it != ios_to_submit_.end();) {
        nixlPosixLinuxAioIO *io = *it;
        if (io->ctx_ != ctx) {
            ++it;
            continue;
        }
        if (io->clb_) {
            io->clb_(io->ctx_, 0, 1);
        }
        io->force_error_ = false;
        it = ios_to_submit_.erase(it);
        free_ios_.push_back(io);
    }

    for (auto &io : ios_) {
        if (!io.in_flight_ || io.ctx_ != ctx) {
            continue;
        }

        if (io.force_error_) {
            continue;
        }

        // A sibling I/O failed, so report this I/O as failed even if it wins
        // the race with io_cancel() and completes successfully.
        io.force_error_ = true;
        cancellations_pending_++;
        struct io_event event = {};
        int ret = io_cancel(io_ctx_, &io.io_, &event);
        if (ret == 0) {
            if (io.clb_) {
                io.clb_(io.ctx_, 0, 1);
            }
            io.in_flight_ = false;
            io.force_error_ = false;
            NIXL_ASSERT(cancellations_pending_ > 0);
            cancellations_pending_--;
            free_ios_.push_back(&io);
        } else {
            // The request may already be completing or may not be cancelable.
            // Keep it alive until io_getevents() reaps and accounts for it.
            if (ret != -EAGAIN && ret != -EINVAL) {
                NIXL_DEBUG << "io_cancel failed: " << nixl_strerror(-ret);
            }
        }
    }

    return hasPendingCleanup() ? NIXL_IN_PROG : NIXL_SUCCESS;
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
