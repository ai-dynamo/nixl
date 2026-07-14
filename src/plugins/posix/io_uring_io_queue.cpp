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
    bool in_flight_ = false; // owned by the ring, not yet reaped
    bool force_error_ = false; // fail the owning request after a queue-wide submit failure
    bool cancel_queued_ = false; // a cancellation SQE has been prepared for this io
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
    virtual nixl_status_t
    cancel(void *ctx) override;

    virtual bool
    hasPendingCleanup(void) const override {
        return submission_failure_draining_ || cancel_sqes_outstanding_ != 0;
    }
    virtual ~nixlPosixIOQueueUring() override;

protected:
    nixlPosixIoUringIO *
    getBufInfo(struct iocb *io);
    nixl_status_t
    reapCompletions(void);

private:
    nixl_status_t
    driveSubmissions(void);
    void
    beginSubmissionFailureDrain(void);
    void
    prepareIOSQEs(void);
    bool
    prepareCancelSQEs(void *ctx);

    struct io_uring uring; // The io_uring instance for async I/O operations
    bool submission_failure_draining_;
    unsigned cancel_sqes_outstanding_;
};

nixlPosixIOQueueUring::nixlPosixIOQueueUring(uint32_t ios_pool_size, uint32_t kernel_queue_size)
    : nixlPosixIOQueueImpl<nixlPosixIoUringIO>(ios_pool_size, kernel_queue_size),
      submission_failure_draining_(false),
      cancel_sqes_outstanding_(0) {
    io_uring_params params = {};
    int ret = io_uring_queue_init_params(kernel_queue_size_, &uring, &params);
    if (ret < 0) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(-ret)));
    }
}

// Prepare normal I/O SQEs without submitting them.
void
nixlPosixIOQueueUring::prepareIOSQEs(void) {
    int num_ios = std::min(MAX_IO_SUBMIT_BATCH_SIZE, (int)ios_to_submit_.size());
    for (int i = 0; i < num_ios; i++) {
        // Keep the io queued until an SQE is available. A previous short
        // submit can leave the ring full of SQEs that still need a retry.
        struct io_uring_sqe *sqe = io_uring_get_sqe(&uring);
        if (!sqe) {
            break;
        }

        nixlPosixIoUringIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();

        if (io->read_) {
            io_uring_prep_read(sqe, io->fd, io->buf_, io->len_, io->offset_);
        } else {
            io_uring_prep_write(sqe, io->fd, io->buf_, io->len_, io->offset_);
        }

        io_uring_sqe_set_data(sqe, io);
        io->in_flight_ = true;
    }
}

// Prepare cancellation SQEs without submitting them. A null context selects every
// in-flight io for the private queue-wide cancellation path.
bool
nixlPosixIOQueueUring::prepareCancelSQEs(void *ctx) {
    bool prepared = false;
    for (auto &io : ios_) {
        if (!io.in_flight_ || io.cancel_queued_ || (ctx && io.ctx_ != ctx)) {
            continue;
        }

        struct io_uring_sqe *sqe = io_uring_get_sqe(&uring);
        if (!sqe) {
            break;
        }

        io_uring_prep_cancel(sqe, &io, 0);
        io_uring_sqe_set_data(sqe, nullptr);
        io.cancel_queued_ = true;
        cancel_sqes_outstanding_++;
        prepared = true;
    }
    return prepared;
}

void
nixlPosixIOQueueUring::beginSubmissionFailureDrain(void) {
    if (submission_failure_draining_) {
        return;
    }

    submission_failure_draining_ = true;

    // These ios never reached the ring and can be failed and reclaimed now.
    while (!ios_to_submit_.empty()) {
        nixlPosixIoUringIO *io = ios_to_submit_.front();
        ios_to_submit_.pop_front();
        if (io->clb_) {
            io->clb_(io->ctx_, 0, 1);
        }
        free_ios_.push_back(io);
    }

    // Ring-owned ios must stay alive until their CQEs are reaped. Force their
    // callbacks to report failure even if an operation wins the cancel race.
    for (auto &io : ios_) {
        if (io.in_flight_) {
            io.force_error_ = true;
        }
    }

    prepareCancelSQEs(nullptr);
}

nixl_status_t
nixlPosixIOQueueUring::post(void) {
    return driveSubmissions();
}

// Prepare eligible I/O or cancellation SQEs, then submit every ring-ready SQE.
// This is the only method that calls io_uring_submit() and returns NIXL_IN_PROG on success.
nixl_status_t
nixlPosixIOQueueUring::driveSubmissions(void) {
    if (!submission_failure_draining_) {
        prepareIOSQEs();
    } else {
        prepareCancelSQEs(nullptr);
    }

    // io_uring_submit() can consume only part of the SQ. Liburing keeps the
    // remainder ready in the ring, so retry it even when ios_to_submit_ is empty.
    // An empty submit normally stays in userspace, but may enter the kernel to
    // flush completion-side state such as CQ overflow.
    int ret = io_uring_submit(&uring);
    if (ret >= 0 || ret == -EAGAIN || ret == -EBUSY || ret == -EINTR) {
        return NIXL_IN_PROG;
    }

    NIXL_ERROR << "io_uring_submit failed: " << nixl_strerror(-ret);
    beginSubmissionFailureDrain();
    return NIXL_ERR_BACKEND;
}

inline nixl_status_t
nixlPosixIOQueueUring::reapCompletions(void) {
    struct io_uring_cqe *cqe;
    unsigned head;
    int count = 0;
    io_uring_for_each_cqe(&uring, head, cqe) {
        int res = cqe->res;
        nixlPosixIoUringIO *io = reinterpret_cast<nixlPosixIoUringIO *>(io_uring_cqe_get_data(cqe));
        // cancel SQEs carry a null sentinel user_data; reap the completion but
        // do not treat it as an io.
        if (!io) {
            NIXL_ASSERT(cancel_sqes_outstanding_ > 0);
            cancel_sqes_outstanding_--;
            count++;
            if (count == MAX_IO_CHECK_COMPLETED_BATCH_SIZE) {
                break;
            }
            continue;
        }
        int error = io->force_error_ || res < 0 || static_cast<size_t>(res) != io->len_;
        if (error) {
            NIXL_DEBUG << absl::StrFormat(
                "IO operation incomplete: result %d, expected %zu", res, io->len_);
        }
        if (io->clb_) {
            io->clb_(io->ctx_, error ? 0 : static_cast<uint32_t>(res), error);
        }
        io->in_flight_ = false;
        io->force_error_ = false;
        io->cancel_queued_ = false;
        free_ios_.push_back(io);
        count++;
        if (count == MAX_IO_CHECK_COMPLETED_BATCH_SIZE) {
            break;
        }
    }

    // Mark all seen
    io_uring_cq_advance(&uring, count);

    // Submitted cancel SQEs are ordered before any future work, and their null
    // CQEs are harmless. Reuse is safe once no user io, cancellation CQE, or
    // ready SQE remains.
    if (submission_failure_draining_ && free_ios_.size() == ios_pool_size_ &&
        cancel_sqes_outstanding_ == 0 && io_uring_sq_ready(&uring) == 0) {
        submission_failure_draining_ = false;
    }

    if (free_ios_.size() == ios_pool_size_ && cancel_sqes_outstanding_ == 0 &&
        !submission_failure_draining_) {
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
    if (submission_failure_draining_) {
        return NIXL_ERR_BACKEND;
    }
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
    io->force_error_ = false;
    io->cancel_queued_ = false;

    ios_to_submit_.push_back(io);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::poll(void) {
    nixl_status_t submit_status = driveSubmissions();
    nixl_status_t completion_status = reapCompletions();

    return submit_status < 0 ? submit_status : completion_status;
}

nixl_status_t
nixlPosixIOQueueUring::cancel(void *ctx) {
    // Null is reserved internally for queue-wide submission-failure cleanup.
    if (!ctx) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Drop this transfer's unsubmitted ios and fail their callbacks so the
    // owning request can account for every operation.
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

    // Best-effort: cancel this transfer's ring-owned ios. Cancel SQEs carry a
    // null sentinel and are tracked until their CQEs are reaped.
    if (prepareCancelSQEs(ctx)) {
        return driveSubmissions();
    }
    return NIXL_SUCCESS;
}

nixlPosixIOQueueUring::~nixlPosixIOQueueUring() {
    io_uring_queue_exit(&uring);
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueUring>(ios_pool_size, kernel_queue_size);
}
