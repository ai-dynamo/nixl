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
#include <cerrno>

#define MAX_IO_SUBMIT_BATCH_SIZE 64
#define MAX_IO_CHECK_COMPLETED_BATCH_SIZE 64

enum class nixlPosixIoUringCQEKind {
    IO,
    CANCEL,
};

struct nixlPosixIoUringCQEData {
    explicit nixlPosixIoUringCQEData(nixlPosixIoUringCQEKind kind) : kind_(kind) {}

    nixlPosixIoUringCQEKind kind_;
    void *ctx_ = nullptr;
};

struct nixlPosixIoUringIO : public nixlPosixIoUringCQEData {
    nixlPosixIoUringIO() : nixlPosixIoUringCQEData(nixlPosixIoUringCQEKind::IO) {}

    int fd;
    void *buf_;
    size_t len_;
    off_t offset_;
    bool read_;
    nixlPosixIOQueueDoneCb clb_;
    bool in_flight_ = false; // owned by the ring, not yet reaped
    bool cancel_pending_ = false; // cancellation is queued or its CQE is pending
};

struct nixlPosixIoUringCancel : public nixlPosixIoUringCQEData {
    nixlPosixIoUringCancel() : nixlPosixIoUringCQEData(nixlPosixIoUringCQEKind::CANCEL) {}

    nixlPosixIoUringIO *io_ = nullptr;
    nixlPosixIOQueueCancelDoneCb clb_;
};

class nixlPosixIOQueueUring : public nixlPosixIOQueueImpl<nixlPosixIoUringIO> {
public:
    nixlPosixIOQueueUring(uint32_t ios_pool_size, uint32_t kernel_queue_size);

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
    virtual unsigned
    cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) override;
    virtual ~nixlPosixIOQueueUring() override;

protected:
    nixl_status_t
    doCheckCompleted(void);

private:
    nixl_status_t
    driveSubmissions(void);
    void
    failQueuedIOs(void *ctx);
    void
    prepareSQEs(void);
    void
    releaseIOIfIdle(nixlPosixIoUringIO *io);

    struct io_uring uring; // The io_uring instance for async I/O operations
    bool terminal_error_ = false;
    std::list<nixlPosixIoUringIO *> cancels_to_submit_;
    std::vector<nixlPosixIoUringCancel> cancels_;
};

nixlPosixIOQueueUring::nixlPosixIOQueueUring(uint32_t ios_pool_size, uint32_t kernel_queue_size)
    : nixlPosixIOQueueImpl<nixlPosixIoUringIO>(ios_pool_size, kernel_queue_size),
      cancels_(ios_.size()) {
    for (size_t i = 0; i < ios_.size(); i++) {
        cancels_[i].io_ = &ios_[i];
    }

    io_uring_params params = {};
    int ret = io_uring_queue_init_params(kernel_queue_size_, &uring, &params);
    if (ret < 0) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(-ret)));
    }
}

// Prepare pending cancellation SQEs before normal I/O SQEs, without submitting them.
void
nixlPosixIOQueueUring::prepareSQEs(void) {
    int num_sqes = 0;
    while (num_sqes < MAX_IO_SUBMIT_BATCH_SIZE) {
        if (cancels_to_submit_.empty() && ios_to_submit_.empty()) {
            break;
        }

        struct io_uring_sqe *sqe = io_uring_get_sqe(&uring);
        if (!sqe) {
            break;
        }

        if (!cancels_to_submit_.empty()) {
            nixlPosixIoUringIO *io = cancels_to_submit_.front();
            cancels_to_submit_.pop_front();
            size_t index = static_cast<size_t>(io - ios_.data());
            auto *io_data = static_cast<nixlPosixIoUringCQEData *>(io);
            io_uring_prep_cancel(sqe, io_data, 0);
            io_uring_sqe_set_data(sqe, static_cast<nixlPosixIoUringCQEData *>(&cancels_[index]));
            NIXL_ASSERT(io->cancel_pending_);
        } else {
            nixlPosixIoUringIO *io = ios_to_submit_.front();
            ios_to_submit_.pop_front();
            if (io->read_) {
                io_uring_prep_read(sqe, io->fd, io->buf_, io->len_, io->offset_);
            } else {
                io_uring_prep_write(sqe, io->fd, io->buf_, io->len_, io->offset_);
            }
            io_uring_sqe_set_data(sqe, static_cast<nixlPosixIoUringCQEData *>(io));
            io->in_flight_ = true;
        }
        num_sqes++;
    }
}

void
nixlPosixIOQueueUring::failQueuedIOs(void *ctx) {
    for (auto it = ios_to_submit_.begin(); it != ios_to_submit_.end();) {
        nixlPosixIoUringIO *io = *it;
        if (io->ctx_ != ctx) {
            ++it;
            continue;
        }
        if (io->clb_) {
            io->clb_(io->ctx_, 0, 1);
        }
        it = ios_to_submit_.erase(it);
        free_ios_.push_back(io);
    }
}

void
nixlPosixIOQueueUring::releaseIOIfIdle(nixlPosixIoUringIO *io) {
    if (!io->in_flight_ && !io->cancel_pending_) {
        free_ios_.push_back(io);
    }
}

nixl_status_t
nixlPosixIOQueueUring::post(void) {
    return driveSubmissions();
}

// Prepare I/O SQEs and submit every ring-ready SQE.
nixl_status_t
nixlPosixIOQueueUring::driveSubmissions(void) {
    if (terminal_error_) {
        return NIXL_IN_PROG;
    }

    prepareSQEs();

    int ret = io_uring_submit(&uring);
    if (ret >= 0 || ret == -EAGAIN || ret == -EBUSY || ret == -EINTR) {
        return NIXL_IN_PROG;
    }

    NIXL_ERROR << "io_uring_submit failed: " << nixl_strerror(-ret);
    terminal_error_ = true;
    return NIXL_IN_PROG;
}

inline nixl_status_t
nixlPosixIOQueueUring::doCheckCompleted(void) {
    struct io_uring_cqe *cqe;
    unsigned head;
    int count = 0;
    io_uring_for_each_cqe(&uring, head, cqe) {
        int res = cqe->res;
        auto *data = static_cast<nixlPosixIoUringCQEData *>(io_uring_cqe_get_data(cqe));
        NIXL_ASSERT(data);
        nixlPosixIoUringIO *io;
        if (data->kind_ == nixlPosixIoUringCQEKind::CANCEL) {
            auto *cancel = static_cast<nixlPosixIoUringCancel *>(data);
            io = cancel->io_;
            NIXL_ASSERT(io && io->cancel_pending_);
            io->cancel_pending_ = false;
            if (cancel->clb_) {
                cancel->clb_(cancel->ctx_);
            }
            cancel->clb_ = nullptr;
            cancel->ctx_ = nullptr;
        } else {
            io = static_cast<nixlPosixIoUringIO *>(data);
            int error = res < 0 || static_cast<size_t>(res) != io->len_;
            if (error) {
                NIXL_DEBUG << absl::StrFormat(
                    "IO operation incomplete: result %d, expected %zu", res, io->len_);
            }
            if (io->clb_) {
                io->clb_(io->ctx_, error ? 0 : static_cast<uint32_t>(res), error);
            }
            io->in_flight_ = false;
        }
        releaseIOIfIdle(io);
        if (++count == MAX_IO_CHECK_COMPLETED_BATCH_SIZE) {
            break;
        }
    }

    // Mark all seen
    io_uring_cq_advance(&uring, count);

    if (free_ios_.size() == ios_pool_size_) {
        return NIXL_SUCCESS; // All ios and cancellation cleanup are done
    }

    return NIXL_IN_PROG; // Some ios or cancellation SQEs still need to drain
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
    io->in_flight_ = false;
    io->cancel_pending_ = false;

    ios_to_submit_.push_back(io);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::poll(void) {
    nixl_status_t completion_status = doCheckCompleted();
    if (completion_status == NIXL_SUCCESS) {
        return NIXL_SUCCESS;
    }
    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }

    return driveSubmissions();
}

unsigned
nixlPosixIOQueueUring::cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) {
    if (!ctx) {
        return 0;
    }

    failQueuedIOs(ctx);

    unsigned cancels_requested = 0;
    for (auto &io : ios_) {
        if (io.in_flight_ && io.ctx_ == ctx && !io.cancel_pending_) {
            size_t index = static_cast<size_t>(&io - ios_.data());
            io.cancel_pending_ = true;
            cancels_[index].clb_ = clb;
            cancels_[index].ctx_ = ctx;
            cancels_to_submit_.push_back(&io);
            cancels_requested++;
        }
    }

    if (cancels_requested != 0) {
        // Best-effort cancellation blocks only its owning request until callbacks are invoked.
        driveSubmissions();
    }
    return cancels_requested;
}

nixlPosixIOQueueUring::~nixlPosixIOQueueUring() {
    io_uring_queue_exit(&uring);
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueUring>(ios_pool_size, kernel_queue_size);
}
