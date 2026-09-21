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

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <unistd.h>

namespace {

struct fileState;
struct ioSlot;

ioSlot *
completeOpen(void *owner, int result);
ioSlot *
completeData(void *owner, int result);
ioSlot *
completeCancel(void *owner, int result);
ioSlot *
completeClose(void *owner, int result);

struct completion {
    using handler_t = ioSlot *(*)(void *, int);

    handler_t handler;
    void *owner;
};

struct fileState {
    enum class status_t { PENDING_OPEN, OPENING, OPEN, FAILED, PENDING_CLOSE, CLOSING, CLOSED };

    explicit fileState(nixl::PathSpec path_spec)
        : path(std::move(path_spec.path)),
          flags(path_spec.flags),
          mode(path_spec.mode),
          completionData{completeOpen, this} {}

    std::string path;
    int flags;
    mode_t mode;
    status_t status = status_t::PENDING_OPEN;
    int fd = -1;
    int openError = 0;
    size_t activeIos = 0;
    bool deregistered = false;
    completion completionData;
};

struct ioSlot {
    enum class state_t { FREE, QUEUED, IN_FLIGHT };

    ioSlot() : data_completion_{completeData, this}, cancel_completion_{completeCancel, this} {}

    int fd = -1;
    void *buf_ = nullptr;
    size_t len_ = 0;
    off_t offset_ = 0;
    bool read_ = false;
    nixlPosixIOQueueDoneCb clb_;
    void *ctx_ = nullptr;
    std::shared_ptr<fileState> file_;
    state_t state_ = state_t::FREE;
    bool cancel_pending_ = false;
    bool cancel_submitted_ = false;
    nixlPosixIOQueueCancelDoneCb cancel_clb_;
    completion data_completion_;
    completion cancel_completion_;
};

class nixlPosixIOQueueUring : public nixlPosixIOQueueImpl<ioSlot> {
public:
    nixlPosixIOQueueUring(uint32_t ios_pool_size,
                          uint32_t kernel_queue_size,
                          bool open_synchronous);

    nixl_status_t
    post(void) override;
    nixl_status_t
    enqueue(uint64_t dev_id,
            void *buf,
            size_t len,
            off_t offset,
            bool read,
            nixlPosixIOQueueDoneCb clb,
            void *ctx) override;
    nixl_status_t
    registerFile(uint64_t dev_id, const std::string &meta_info) override;
    nixl_status_t
    deregisterFile(uint64_t dev_id) override;

    nixl_status_t
    poll(void) override;
    unsigned
    cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) override;
    ~nixlPosixIOQueueUring() override;

private:
    nixl_status_t
    enqueueFd(int fd,
              void *buf,
              size_t len,
              off_t offset,
              bool read,
              nixlPosixIOQueueDoneCb clb,
              void *ctx) override;
    nixl_status_t
    enqueueIO(int fd,
              std::shared_ptr<fileState> file,
              void *buf,
              size_t len,
              off_t offset,
              bool read,
              nixlPosixIOQueueDoneCb clb,
              void *ctx);
    void
    doCheckCompleted(void);
    nixl_status_t
    submitPrepared(unsigned prepared);
    void
    retireFile(uint64_t dev_id);
    void
    completeQueuedIO(ioSlot *io, int error);

    struct io_uring uring_{};

    uint32_t cq_capacity_ = 0;
    size_t in_flight_cqes_ = 0;
    unsigned pending_sqes_ = 0;
    bool terminal_error_ = false;
    bool open_supported_ = false;
    bool open_synchronous_;
    std::unordered_map<uint64_t, std::shared_ptr<fileState>> path_files_;
};

nixlPosixIOQueueUring::nixlPosixIOQueueUring(uint32_t ios_pool_size,
                                             uint32_t kernel_queue_size,
                                             bool open_synchronous)
    : nixlPosixIOQueueImpl<ioSlot>(ios_pool_size, kernel_queue_size),
      open_synchronous_(open_synchronous) {
    io_uring_params params = {};
    int ret = io_uring_queue_init_params(kernel_queue_size_, &uring_, &params);
    if (ret < 0) {
        throw std::runtime_error(
            absl::StrFormat("Failed to initialize io_uring instance: %s", nixl_strerror(-ret)));
    }
    cq_capacity_ = params.cq_entries;

    // Probe the running kernel because vendors routinely backport io_uring features.
    io_uring_probe *probe = io_uring_get_probe_ring(&uring_);
    open_supported_ = probe && io_uring_opcode_supported(probe, IORING_OP_OPENAT);
    if (probe) {
        io_uring_free_probe(probe);
    }
}

nixl_status_t
nixlPosixIOQueueUring::enqueueFd(int fd,
                                 void *buf,
                                 size_t len,
                                 off_t offset,
                                 bool read,
                                 nixlPosixIOQueueDoneCb clb,
                                 void *ctx) {
    return enqueueIO(fd, {}, buf, len, offset, read, std::move(clb), ctx);
}

nixl_status_t
nixlPosixIOQueueUring::enqueue(uint64_t dev_id,
                               void *buf,
                               size_t len,
                               off_t offset,
                               bool read,
                               nixlPosixIOQueueDoneCb clb,
                               void *ctx) {
    auto file = path_files_.find(dev_id);
    if (file == path_files_.end()) {
        return nixlPosixIOQueue::enqueue(dev_id, buf, len, offset, read, std::move(clb), ctx);
    }
    return enqueueIO(-1, file->second, buf, len, offset, read, std::move(clb), ctx);
}

nixl_status_t
nixlPosixIOQueueUring::enqueueIO(int fd,
                                 std::shared_ptr<fileState> file,
                                 void *buf,
                                 size_t len,
                                 off_t offset,
                                 bool read,
                                 nixlPosixIOQueueDoneCb clb,
                                 void *ctx) {
    if (file &&
        (file->status == fileState::status_t::PENDING_CLOSE ||
         file->status == fileState::status_t::CLOSING ||
         file->status == fileState::status_t::CLOSED)) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (free_ios_.empty()) {
        NIXL_ERROR << "No more free blocks available";
        return NIXL_ERR_NOT_ALLOWED;
    }
    ioSlot *io = free_ios_.front();
    free_ios_.pop_front();
    io->fd = fd;
    io->buf_ = buf;
    io->len_ = len;
    io->offset_ = offset;
    io->read_ = read;
    io->clb_ = std::move(clb);
    io->ctx_ = ctx;
    io->file_ = std::move(file);
    io->state_ = ioSlot::state_t::QUEUED;
    io->cancel_pending_ = false;
    io->cancel_submitted_ = false;
    io->cancel_clb_ = {};
    if (io->file_) {
        ++io->file_->activeIos;
    }
    return NIXL_SUCCESS;
}

void
queueClose(fileState *file) {
    if (file->fd < 0 || file->status == fileState::status_t::PENDING_CLOSE ||
        file->status == fileState::status_t::CLOSING ||
        file->status == fileState::status_t::CLOSED) {
        return;
    }
    file->status = fileState::status_t::PENDING_CLOSE;
}

void
nixlPosixIOQueueUring::retireFile(uint64_t dev_id) {
    auto file_it = path_files_.find(dev_id);
    NIXL_ASSERT(file_it != path_files_.end());
    const auto &file = file_it->second;
    file->deregistered = true;
    if (file->status == fileState::status_t::PENDING_OPEN) {
        file->status = fileState::status_t::CLOSED;
    } else if (file->status == fileState::status_t::OPEN) {
        queueClose(file.get());
    } else if (file->status == fileState::status_t::FAILED) {
        file->status = fileState::status_t::CLOSED;
    }
    if (file->status == fileState::status_t::CLOSED) {
        path_files_.erase(file_it);
    }
}

nixl_status_t
nixlPosixIOQueueUring::deregisterFile(uint64_t dev_id) {
    doCheckCompleted();
    auto file_it = path_files_.find(dev_id);
    if (file_it == path_files_.end()) {
        return nixlPosixIOQueue::deregisterFile(dev_id);
    }
    const auto file = file_it->second;
    if (file->activeIos != 0) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    retireFile(dev_id);
    while (file->status != fileState::status_t::CLOSED) {
        nixl_status_t status = poll();
        if (status < 0) {
            return status;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::registerFile(uint64_t dev_id, const std::string &meta_info) {
    auto path_spec = nixl::parsePathMeta(meta_info);
    if (!path_spec) {
        if (path_files_.contains(dev_id)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        return nixlPosixIOQueue::registerFile(dev_id, meta_info);
    }

    if (!open_supported_) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (path_files_.contains(dev_id) || files_.contains(dev_id)) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    auto file = std::make_shared<fileState>(std::move(*path_spec));
    path_files_.emplace(dev_id, file);

    nixl_status_t status = post();
    if (status < 0) {
        retireFile(dev_id);
        return status;
    }
    if (!open_synchronous_) {
        return NIXL_SUCCESS;
    }

    while (file->status == fileState::status_t::PENDING_OPEN ||
           file->status == fileState::status_t::OPENING) {
        status = poll();
        if (status < 0) {
            retireFile(dev_id);
            return status;
        }
    }

    if (file->status == fileState::status_t::FAILED) {
        retireFile(dev_id);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlPosixIOQueueUring::submitPrepared(unsigned prepared) {
    pending_sqes_ += prepared;
    while (pending_sqes_ > 0) {
        int ret = io_uring_submit(&uring_);
        if (ret == -EAGAIN || ret == -EBUSY || ret == -EINTR) {
            return NIXL_IN_PROG;
        }
        if (ret < 0) {
            NIXL_ERROR << "io_uring_submit failed: " << nixl_strerror(-ret);
            terminal_error_ = true;
            return NIXL_ERR_BACKEND;
        }
        if (ret == 0) {
            return NIXL_IN_PROG;
        }
        const unsigned submitted = std::min(pending_sqes_, static_cast<unsigned>(ret));
        pending_sqes_ -= submitted;
        in_flight_cqes_ += submitted;
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlPosixIOQueueUring::post(void) {
    if (terminal_error_) {
        return NIXL_ERR_BACKEND;
    }
    if (pending_sqes_ > 0) {
        return submitPrepared(0);
    }
    std::erase_if(path_files_, [](const auto &entry) {
        return entry.second->status == fileState::status_t::CLOSED;
    });
    bool failed_io = false;
    for (auto &io : ios_) {
        if (io.state_ == ioSlot::state_t::QUEUED && io.file_ &&
            io.file_->status == fileState::status_t::FAILED) {
            completeQueuedIO(&io, io.file_->openError);
            failed_io = true;
        }
    }
    if (failed_io) {
        return NIXL_IN_PROG;
    }

    const size_t occupied_cqes = in_flight_cqes_ + pending_sqes_;
    const size_t available_cqes = cq_capacity_ > occupied_cqes ? cq_capacity_ - occupied_cqes : 0;
    if (available_cqes == 0) {
        return NIXL_IN_PROG;
    }

    unsigned prepared = 0;
    for (auto &io : ios_) {
        if (prepared >= available_cqes || io_uring_sq_space_left(&uring_) < 1) {
            break;
        }
        if (!io.cancel_pending_ || io.cancel_submitted_) {
            continue;
        }
        io_uring_sqe *sqe = io_uring_get_sqe(&uring_);
        io_uring_prep_cancel(sqe, &io.data_completion_, 0);
        io_uring_sqe_set_data(sqe, &io.cancel_completion_);
        io.cancel_submitted_ = true;
        ++prepared;
    }

    for (const auto &[dev_id, file] : path_files_) {
        if (prepared >= available_cqes || io_uring_sq_space_left(&uring_) < 1) {
            break;
        }
        if (file->status != fileState::status_t::PENDING_CLOSE) {
            continue;
        }
        io_uring_sqe *sqe = io_uring_get_sqe(&uring_);
        io_uring_prep_close(sqe, file->fd);
        file->completionData.handler = completeClose;
        io_uring_sqe_set_data(sqe, &file->completionData);
        file->status = fileState::status_t::CLOSING;
        ++prepared;
    }

    // Prioritize I/O unblocked by completed opens over more registration opens.
    for (auto &io : ios_) {
        if (prepared >= available_cqes || io_uring_sq_space_left(&uring_) < 1) {
            break;
        }
        if (io.state_ != ioSlot::state_t::QUEUED ||
            (io.file_ && io.file_->status != fileState::status_t::OPEN)) {
            continue;
        }
        io_uring_sqe *data_sqe = io_uring_get_sqe(&uring_);
        const int fd = io.file_ ? io.file_->fd : io.fd;
        if (io.read_) {
            io_uring_prep_read(data_sqe, fd, io.buf_, io.len_, io.offset_);
        } else {
            io_uring_prep_write(data_sqe, fd, io.buf_, io.len_, io.offset_);
        }
        io.state_ = ioSlot::state_t::IN_FLIGHT;
        io_uring_sqe_set_data(data_sqe, &io.data_completion_);
        ++prepared;
    }

    for (const auto &[dev_id, file] : path_files_) {
        if (prepared >= available_cqes || io_uring_sq_space_left(&uring_) < 1) {
            break;
        }
        if (file->status != fileState::status_t::PENDING_OPEN) {
            continue;
        }
        io_uring_sqe *open_sqe = io_uring_get_sqe(&uring_);
        io_uring_prep_openat(open_sqe, AT_FDCWD, file->path.c_str(), file->flags, file->mode);
        io_uring_sqe_set_data(open_sqe, &file->completionData);
        file->status = fileState::status_t::OPENING;
        ++prepared;
    }

    return submitPrepared(prepared);
}

void
nixlPosixIOQueueUring::completeQueuedIO(ioSlot *io, int error) {
    if (io->clb_) {
        io->clb_(io->ctx_, 0, error);
    }
    if (io->file_ && io->file_->activeIos > 0) {
        --io->file_->activeIos;
    }
    io->file_.reset();
    io->state_ = ioSlot::state_t::FREE;
    free_ios_.push_back(io);
}

ioSlot *
completeOpen(void *owner, int result) {
    auto *file = static_cast<fileState *>(owner);
    if (result < 0) {
        file->status = fileState::status_t::FAILED;
        file->openError = -result;
        if (file->deregistered) {
            file->status = fileState::status_t::CLOSED;
        }
        NIXL_ERROR << absl::StrFormat(
            "io_uring open failed for %s: %s", file->path, nixl_strerror(-result));
        return nullptr;
    }

    file->fd = result;
    file->status = fileState::status_t::OPEN;
    if (file->deregistered) {
        queueClose(file);
    }
    return nullptr;
}

ioSlot *
completeData(void *owner, int result) {
    auto *io = static_cast<ioSlot *>(owner);
    const auto file = io->file_;
    const int error = result < 0 ? -result : static_cast<size_t>(result) != io->len_;
    if (io->clb_) {
        io->clb_(io->ctx_, error ? 0 : static_cast<uint32_t>(result), error);
    }
    if (error) {
        NIXL_DEBUG << absl::StrFormat(
            "IO operation incomplete: result %d, expected %zu", result, io->len_);
    }
    if (file) {
        --file->activeIos;
    }
    io->file_.reset();
    io->state_ = ioSlot::state_t::FREE;
    if (io->cancel_pending_ && !io->cancel_submitted_) {
        io->cancel_pending_ = false;
        if (io->cancel_clb_) {
            io->cancel_clb_(io->ctx_);
        }
        io->cancel_clb_ = {};
    }
    return io->cancel_pending_ ? nullptr : io;
}

ioSlot *
completeCancel(void *owner, int) {
    auto *io = static_cast<ioSlot *>(owner);
    io->cancel_pending_ = false;
    io->cancel_submitted_ = false;
    if (io->cancel_clb_) {
        io->cancel_clb_(io->ctx_);
    }
    io->cancel_clb_ = {};
    return io->state_ == ioSlot::state_t::FREE ? io : nullptr;
}

ioSlot *
completeClose(void *owner, int result) {
    auto *file = static_cast<fileState *>(owner);
    if (result < 0) {
        NIXL_ERROR << absl::StrFormat(
            "io_uring close failed for %s: %s", file->path, nixl_strerror(-result));
        ::close(file->fd);
    }
    file->fd = -1;
    file->status = fileState::status_t::CLOSED;
    return nullptr;
}

void
nixlPosixIOQueueUring::doCheckCompleted(void) {
    io_uring_cqe *cqe;
    unsigned head;
    unsigned count = 0;
    io_uring_for_each_cqe(&uring_, head, cqe) {
        auto *completion_data = reinterpret_cast<completion *>(io_uring_cqe_get_data(cqe));
        if (completion_data) {
            if (auto *io = completion_data->handler(completion_data->owner, cqe->res)) {
                free_ios_.push_back(io);
            }
        }
        ++count;
    }

    if (count > 0) {
        io_uring_cq_advance(&uring_, count);
        in_flight_cqes_ -= std::min(in_flight_cqes_, static_cast<size_t>(count));
    }
}

unsigned
nixlPosixIOQueueUring::cancel(void *ctx, nixlPosixIOQueueCancelDoneCb clb) {
    if (!ctx) {
        return 0;
    }

    for (auto &io : ios_) {
        if (io.state_ == ioSlot::state_t::QUEUED && io.ctx_ == ctx) {
            completeQueuedIO(&io, 1);
        }
    }

    unsigned requested = 0;
    for (auto &io : ios_) {
        if (io.state_ != ioSlot::state_t::IN_FLIGHT || io.ctx_ != ctx || io.cancel_pending_) {
            continue;
        }
        io.cancel_pending_ = true;
        io.cancel_clb_ = clb;
        ++requested;
    }

    if (requested > 0) {
        post();
    }
    return requested;
}

nixl_status_t
nixlPosixIOQueueUring::poll(void) {
    doCheckCompleted();
    nixl_status_t post_status = post();
    if (post_status < 0) {
        return post_status;
    }
    return free_ios_.size() == ios_pool_size_ ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixlPosixIOQueueUring::~nixlPosixIOQueueUring() {
    // Drain outstanding operations before ring teardown.
    while (!terminal_error_) {
        doCheckCompleted();
        const nixl_status_t status = post();
        if (status < 0) {
            break;
        }
        if (pending_sqes_ == 0 && in_flight_cqes_ == 0) {
            break;
        }
        if (in_flight_cqes_ > 0) {
            io_uring_cqe *cqe = nullptr;
            const int ret = io_uring_wait_cqe(&uring_, &cqe);
            if (ret < 0) {
                NIXL_ERROR << "io_uring wait during shutdown failed: " << nixl_strerror(-ret);
                break;
            }
        }
    }
    doCheckCompleted();

    io_uring_queue_exit(&uring_);
    for (const auto &[dev_id, file] : path_files_) {
        if (file->fd >= 0) {
            ::close(file->fd);
        }
    }
}

} // namespace

std::unique_ptr<nixlPosixIOQueue>
nixlPosixIOQueueUringCreate(uint32_t ios_pool_size,
                            uint32_t kernel_queue_size,
                            bool open_synchronous) {
    return std::make_unique<nixlPosixIOQueueUring>(
        ios_pool_size, kernel_queue_size, open_synchronous);
}
