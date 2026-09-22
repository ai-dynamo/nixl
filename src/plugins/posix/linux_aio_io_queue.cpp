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
    void *
    currentBuf() const {
        return static_cast<char *>(buf_) + completed_;
    }

    off_t
    currentOffset() const {
        return offset_ + static_cast<off_t>(completed_);
    }

    size_t
    remaining() const {
        return len_ - completed_;
    }

    void
    advance(size_t completed) {
        NIXL_ASSERT(completed <= remaining());
        completed_ += completed;
    }

    int fd_ = -1;
    void *buf_ = nullptr;
    off_t offset_ = 0;
    bool read_ = false;
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_ = nullptr;
    size_t len_ = 0;
    size_t completed_ = 0;
    bool in_flight_ = false;
    bool cancel_requested_ = false;
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
    virtual unsigned
    cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) override;
    virtual ~nixlPosixIOQueueLinuxAIO() override;

protected:
    nixl_status_t
    doCheckCompleted(void);

private:
    void
    queueForSubmission(nixlPosixLinuxAioIO *io);
    bool
    finishIO(nixlPosixLinuxAioIO *io, int64_t result, bool error);
    bool
    handleIOCompletion(nixlPosixLinuxAioIO *io, int64_t result);
    void
    failIO(nixlPosixLinuxAioIO *io);
    void
    requeueFrom(nixlPosixLinuxAioIO *const *ios, int first, int count);
    void
    failQueuedIOs(void *ctx);
    bool
    isTerminalSubmitError(int error);
    void
    enterTerminalError();

    io_context_t io_ctx_{}; // I/O context
    bool io_ctx_active_ = false;
    bool terminal_error_ = false;
};

nixlPosixIOQueueLinuxAIO::nixlPosixIOQueueLinuxAIO(uint32_t ios_pool_size,
                                                   uint32_t kernel_queue_size)
    : nixlPosixIOQueueImpl<nixlPosixLinuxAioIO>(ios_pool_size, kernel_queue_size) {
    int res = io_queue_init(kernel_queue_size_, &io_ctx_);
    if (res) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_queue: %s", nixl_strerror(-res)));
    }
    io_ctx_active_ = true;
}

nixl_status_t
nixlPosixIOQueueLinuxAIO::enqueue(int fd,
                                  void *buf,
                                  size_t len,
                                  off_t offset,
                                  bool read,
                                  nixlPosixIOQueueDoneCb clb,
                                  void *ctx) {
    NIXL_ASSERT(buf != nullptr);

    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }
    if (free_ios_.empty()) {
        NIXL_ERROR << "No more free blocks available";
        return NIXL_ERR_NOT_ALLOWED;
    }
    nixlPosixLinuxAioIO *io = free_ios_.front();
    free_ios_.pop_front();

    io->fd_ = fd;
    io->buf_ = buf;
    io->offset_ = offset;
    io->read_ = read;
    io->clb_ = clb;
    io->ctx_ = ctx;
    io->len_ = len;
    io->completed_ = 0;
    io->in_flight_ = false;
    io->cancel_requested_ = false;
    queueForSubmission(io);

    return NIXL_SUCCESS;
}

void
nixlPosixIOQueueLinuxAIO::queueForSubmission(nixlPosixLinuxAioIO *io) {
    if (io->read_) {
        io_prep_pread(&io->io_, io->fd_, io->currentBuf(), io->remaining(), io->currentOffset());
    } else {
        io_prep_pwrite(&io->io_, io->fd_, io->currentBuf(), io->remaining(), io->currentOffset());
    }
    io->io_.data = io;
    io->in_flight_ = false;
    ios_to_submit_.push_back(io);
}

nixlPosixIOQueueLinuxAIO::~nixlPosixIOQueueLinuxAIO() {
    if (io_ctx_active_) {
        io_queue_release(io_ctx_);
    }
}

// Note: post() must return NIXL_IN_PROG in case of success.
nixl_status_t
nixlPosixIOQueueLinuxAIO::post(void) {
    struct iocb *ios[MAX_IO_SUBMIT_BATCH_SIZE];
    nixlPosixLinuxAioIO *to_submit[MAX_IO_SUBMIT_BATCH_SIZE];

    if (terminal_error_ || ios_to_submit_.empty()) {
        return NIXL_IN_PROG;
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
        if (ret == -EAGAIN || ret == -EINTR) {
            requeueFrom(to_submit, 0, num_ios);
            return NIXL_IN_PROG;
        }

        if (isTerminalSubmitError(ret)) {
            NIXL_ERROR << "io_submit terminal error: " << nixl_strerror(-ret);
            requeueFrom(to_submit, 0, num_ios);
            enterTerminalError();
            return NIXL_IN_PROG;
        }

        // A negative return rejects the first IOCB without accepting any of the batch.
        NIXL_ERROR << "io_submit rejected I/O: " << nixl_strerror(-ret);
        failIO(to_submit[0]);
        requeueFrom(to_submit, 1, num_ios);
        return NIXL_IN_PROG;
    }

    for (int i = 0; i < ret; i++) {
        to_submit[i]->in_flight_ = true;
    }
    requeueFrom(to_submit, ret, num_ios);

    return NIXL_IN_PROG;
}

void
nixlPosixIOQueueLinuxAIO::requeueFrom(nixlPosixLinuxAioIO *const *ios, int first, int count) {
    for (int i = count - 1; i >= first; i--) {
        ios_to_submit_.push_front(ios[i]);
    }
}

bool
nixlPosixIOQueueLinuxAIO::finishIO(nixlPosixLinuxAioIO *io, int64_t result, bool error) {
    if (error) {
        NIXL_DEBUG << absl::StrFormat(
            "AIO operation incomplete: result %ld, %zu bytes remaining", result, io->remaining());
        failIO(io);
        return true;
    }

    io->in_flight_ = false;
    if (io->clb_) {
        io->clb_(io->ctx_, static_cast<uint32_t>(io->len_), 0);
    }
    free_ios_.push_back(io);
    return false;
}

bool
nixlPosixIOQueueLinuxAIO::handleIOCompletion(nixlPosixLinuxAioIO *io, int64_t result) {
    NIXL_ASSERT(io->in_flight_);
    if (result < 0) {
        return finishIO(io, result, true);
    }

    const size_t completed = static_cast<size_t>(result);

    if (completed == 0 && io->remaining() != 0) {
        return finishIO(io, result, true);
    }

    io->advance(completed);

    if (io->remaining() == 0) {
        return finishIO(io, result, false);
    }

    if (io->cancel_requested_) {
        return finishIO(io, result, true);
    }

    NIXL_DEBUG << absl::StrFormat(
        "AIO operation completed partially: %zu bytes completed, %zu remaining; resubmitting "
        "remainder",
        completed,
        io->remaining());
    queueForSubmission(io);
    return false;
}

void
nixlPosixIOQueueLinuxAIO::failIO(nixlPosixLinuxAioIO *io) {
    io->in_flight_ = false;
    if (io->clb_) {
        io->clb_(io->ctx_, 0, 1);
    }
    free_ios_.push_back(io);
}

inline nixl_status_t
nixlPosixIOQueueLinuxAIO::doCheckCompleted(void) {
    struct io_event events[MAX_IO_CHECK_COMPLETED_BATCH_SIZE];
    struct timespec timeout = {0, 0};

    if (free_ios_.size() == ios_pool_size_) {
        return NIXL_SUCCESS;
    }

    int rc = io_getevents(io_ctx_, 0, MAX_IO_CHECK_COMPLETED_BATCH_SIZE, events, &timeout);
    if (rc < 0) {
        if (rc == -EINTR) {
            return NIXL_IN_PROG;
        }
        NIXL_ERROR << "io_getevents error: " << nixl_strerror(-rc);
        return NIXL_ERR_BACKEND;
    }

    for (int i = 0; i < rc; i++) {
        auto *io = static_cast<nixlPosixLinuxAioIO *>(events[i].obj->data);
        void *ctx = io->ctx_;
        if (handleIOCompletion(io, events[i].res)) {
            failQueuedIOs(ctx);
        }
    }

    return free_ios_.size() == ios_pool_size_ ? NIXL_SUCCESS : NIXL_IN_PROG;
}

bool
nixlPosixIOQueueLinuxAIO::isTerminalSubmitError(int error) {
    if (error == -ENOSYS) {
        return true;
    }
    if (error != -EINVAL) {
        return false;
    }

    // EINVAL is ambiguous: it can describe either the AIO context or the first IOCB.
    // Probe the context without reaping completions before declaring it terminal.
    // min_nr == nr == 0 makes this return immediately even with a null timeout.
    int rc;
    do {
        rc = io_getevents(io_ctx_, 0, 0, nullptr, nullptr);
    } while (rc == -EINTR);

    return rc < 0;
}

void
nixlPosixIOQueueLinuxAIO::failQueuedIOs(void *ctx) {
    for (auto it = ios_to_submit_.begin(); it != ios_to_submit_.end();) {
        nixlPosixLinuxAioIO *io = *it;
        if (ctx && io->ctx_ != ctx) {
            ++it;
            continue;
        }

        it = ios_to_submit_.erase(it);
        failIO(io);
    }
}

void
nixlPosixIOQueueLinuxAIO::enterTerminalError() {
    terminal_error_ = true;
    failQueuedIOs(nullptr);

    if (io_ctx_active_) {
        int ret = io_queue_release(io_ctx_);
        if (ret < 0) {
            NIXL_ERROR << "io_queue_release failed: " << nixl_strerror(-ret);
        }
        io_ctx_active_ = false;
    }

    for (auto &io : ios_) {
        if (io.in_flight_) {
            failIO(&io);
        }
    }
}

unsigned
nixlPosixIOQueueLinuxAIO::cancel(void *ctx, nixlPosixIOQueueCancelDoneCb) {
    if (!ctx) {
        return 0;
    }

    failQueuedIOs(ctx);

    for (auto &io : ios_) {
        if (!io.in_flight_ || io.ctx_ != ctx) {
            continue;
        }

        io.cancel_requested_ = true;
        struct io_event event{};
        if (io_cancel(io_ctx_, &io.io_, &event) == 0) {
            // io_cancel returns the canceled operation's completion synchronously.
            handleIOCompletion(&io, event.res);
        }
    }

    // No asynchronous cancellations were requested.
    return 0;
}

nixl_status_t
nixlPosixIOQueueLinuxAIO::poll(void) {
    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }

    nixl_status_t completion_status = doCheckCompleted();
    if (completion_status == NIXL_SUCCESS) {
        return NIXL_SUCCESS;
    }
    if (completion_status < 0) {
        enterTerminalError();
        return NIXL_IN_PROG;
    }

    return post();
}

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueLinuxAIOCreate(uint32_t ios_pool_size, uint32_t kernel_queue_size) {
    return std::make_unique<nixlPosixIOQueueLinuxAIO>(ios_pool_size, kernel_queue_size);
}
