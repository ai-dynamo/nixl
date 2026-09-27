/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <liburing.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "io_queue.h"
#include "posix_backend.h"

#ifndef LIBURING_NOEXCEPT
#define LIBURING_NOEXCEPT
#endif

namespace {
constexpr int request_count = 32, ring_entries = 16, max_poll_iterations = 2000;
constexpr size_t block_size = 4096;
constexpr size_t partial_io_offset = 123;
constexpr size_t partial_read_size = block_size / 4;
constexpr size_t partial_write_size = 3 * block_size / 8;
constexpr auto poll_pause = std::chrono::microseconds(50);
using buffers_t = std::array<std::array<char, block_size>, request_count>;

enum class submit_mode_t {
    PARTIAL_ONLY,
    TRANSIENT_ERRORS,
    PARTIAL_READ,
    PARTIAL_WRITE,
    PARTIAL_CANCEL,
    PASS_THROUGH
};
submit_mode_t submit_mode = submit_mode_t::PASS_THROUGH;
int submit_calls = 0, transient_submit_errors = 0, cancel_completions = 0;
unsigned first_ready = 0, first_submitted = 0;
int partial_io_submissions = 0;
struct io_uring *last_submitted_ring = nullptr;
std::array<unsigned, 3> partial_io_lengths{};
std::array<size_t, 3> partial_io_offsets{};
std::array<uintptr_t, 3> partial_io_buffers{};

struct completionState {
    int count = 0, errors = 0;
};

void
completionCallback(void *ctx, uint32_t, int error) {
    auto *state = static_cast<completionState *>(ctx);
    state->count++;
    state->errors += error != 0;
}

void
cancelCompletionCallback(void *) {
    cancel_completions++;
}

struct uringTest {
    int fd = -1;
    buffers_t buffers{};
    std::unique_ptr<nixlPosixIOQueue> queue;

    explicit uringTest(submit_mode_t mode)
        : queue(nixlPosixIOQueue::instantiate("URING", 64, ring_entries)) {
        submit_mode = mode;
        submit_calls = transient_submit_errors = cancel_completions = first_ready =
            first_submitted = 0;
        partial_io_submissions = 0;
        last_submitted_ring = nullptr;
        char path[] = "/tmp/nixl_uring_test_XXXXXX";
        if ((fd = mkstemp(path)) < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        unlink(path);
        for (size_t i = 0; i < buffers.size(); i++) {
            std::memset(buffers[i].data(), static_cast<int>(i + 1), buffers[i].size());
        }
    }

    ~uringTest() {
        queue.reset();
        close(fd);
    }

    bool
    enqueue(completionState &state, int start, int count) {
        for (int i = start; i < start + count; i++) {
            if (queue->enqueue(fd,
                               buffers[i].data(),
                               block_size,
                               i * block_size,
                               false,
                               completionCallback,
                               &state) != NIXL_SUCCESS) {
                return false;
            }
        }
        return true;
    }

    nixl_status_t
    drain() {
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
            status = queue->poll();
            std::this_thread::sleep_for(poll_pause);
        }
        return status;
    }
};

struct uringRequest {
    nixl_meta_dlist_t local{DRAM_SEG};
    nixl_meta_dlist_t remote{FILE_SEG};
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixlPosixBackendReqH request;

    uringRequest(uringTest &test,
                 nixlPosixFileMD &file_md,
                 int index,
                 nixl_xfer_op_t op = NIXL_WRITE,
                 size_t offset_bias = 0)
        : local([&] {
              nixl_meta_dlist_t list(DRAM_SEG);
              list.addDesc(nixlMetaDesc(
                  reinterpret_cast<uintptr_t>(test.buffers[index].data()), block_size, 0, nullptr));
              return list;
          }()),
          remote([&] {
              nixl_meta_dlist_t list(FILE_SEG);
              list.addDesc(
                  nixlMetaDesc(offset_bias + index * block_size, block_size, test.fd, &file_md));
              return list;
          }()),
          operation(op),
          request(operation, local, remote, test.queue) {}
};

nixl_status_t
waitFor(uringRequest &request, nixl_status_t status = NIXL_IN_PROG) {
    for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
        status = request.request.checkXfer();
        std::this_thread::sleep_for(poll_pause);
    }
    return status;
}

#define URING_CHECK(condition)                                                        \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::cerr << "URING_CHECK failed at line " << __LINE__ << ": " #condition \
                      << std::endl;                                                   \
            return 1;                                                                 \
        }                                                                             \
    } while (false)
} // namespace

extern "C" int
io_uring_submit(struct io_uring *ring) LIBURING_NOEXCEPT {
    using submit_fn = int (*)(struct io_uring *);
    static auto real_submit = reinterpret_cast<submit_fn>(dlsym(RTLD_NEXT, "io_uring_submit"));
    if (!real_submit) {
        return -EINVAL;
    }
    last_submitted_ring = ring;
    if (submit_mode == submit_mode_t::TRANSIENT_ERRORS && transient_submit_errors == 0) {
        transient_submit_errors++;
        return -EAGAIN;
    }

    const unsigned ready = io_uring_sq_ready(ring);
    submit_calls += ready > 0;
    if (submit_mode == submit_mode_t::PARTIAL_CANCEL && ready > 0 && partial_io_submissions == 0) {
        struct io_uring_sqe *const sqe = &ring->sq.sqes[ring->sq.sqe_head & *ring->sq.kring_mask];
        sqe->len /= 2;
        partial_io_submissions++;
        return real_submit(ring);
    }
    if (ready > 0 &&
        (submit_mode == submit_mode_t::PARTIAL_READ ||
         submit_mode == submit_mode_t::PARTIAL_WRITE)) {
        struct io_uring_sqe *const sqe = &ring->sq.sqes[ring->sq.sqe_head & *ring->sq.kring_mask];
        if (partial_io_submissions < static_cast<int>(partial_io_lengths.size())) {
            partial_io_lengths[partial_io_submissions] = sqe->len;
            partial_io_offsets[partial_io_submissions] = sqe->off;
            partial_io_buffers[partial_io_submissions] = sqe->addr;
        }
        if (partial_io_submissions++ < 2) {
            sqe->len =
                submit_mode == submit_mode_t::PARTIAL_READ ? partial_read_size : partial_write_size;
        }
        return real_submit(ring);
    }
    if (ready == 0 || submit_mode == submit_mode_t::PASS_THROUGH || submit_calls != 1 ||
        ready < 2) {
        return real_submit(ring);
    }

    const unsigned original_tail = ring->sq.sqe_tail;
    ring->sq.sqe_tail = ring->sq.sqe_head + ready / 2;
    const int ret = real_submit(ring);
    ring->sq.sqe_tail = original_tail;
    first_ready = ready;
    first_submitted = ret > 0 ? static_cast<unsigned>(ret) : 0;
    return ret;
}

int
main() {
    io_uring probe_ring{};
    io_uring_params probe_params{};
    int probe_ret = io_uring_queue_init_params(ring_entries, &probe_ring, &probe_params);
    if (probe_ret < 0) {
        std::cerr << "io_uring backend test requires a usable ring: " << std::strerror(-probe_ret)
                  << " (" << probe_ret << ")" << std::endl;
        return 1;
    }
    io_uring_queue_exit(&probe_ring);

    {
        uringTest test(submit_mode_t::PARTIAL_ONLY);
        completionState state;
        URING_CHECK(test.enqueue(state, 0, request_count));
        URING_CHECK(test.queue->post() == NIXL_IN_PROG);
        URING_CHECK(first_submitted > 0 && first_submitted < first_ready);
        URING_CHECK(test.drain() == NIXL_SUCCESS);
        URING_CHECK(state.count == request_count && !state.errors && submit_calls > 1);
    }
    {
        uringTest test(submit_mode_t::TRANSIENT_ERRORS);
        completionState state;
        URING_CHECK(test.enqueue(state, 0, request_count));
        URING_CHECK(test.queue->post() == NIXL_IN_PROG);
        URING_CHECK(test.drain() == NIXL_SUCCESS && transient_submit_errors == 1);
        URING_CHECK(state.count == request_count && !state.errors);
    }
    for (const bool read : {true, false}) {
        const size_t partial_size = read ? partial_read_size : partial_write_size;
        uringTest test(read ? submit_mode_t::PARTIAL_READ : submit_mode_t::PARTIAL_WRITE);
        std::array<char, partial_io_offset> prefix;
        std::array<char, block_size> expected;
        std::memset(prefix.data(), 0x3c, prefix.size());
        std::memset(expected.data(), read ? 0x5a : 0x1, expected.size());
        URING_CHECK(pwrite(test.fd, prefix.data(), prefix.size(), 0) ==
                    static_cast<ssize_t>(prefix.size()));
        if (read) {
            URING_CHECK(pwrite(test.fd, expected.data(), expected.size(), partial_io_offset) ==
                        static_cast<ssize_t>(expected.size()));
            std::memset(test.buffers[0].data(), 0, block_size);
        }
        nixlPosixFileMD file_md(test.fd, "");
        uringRequest transfer(test, file_md, 0, read ? NIXL_READ : NIXL_WRITE, partial_io_offset);
        URING_CHECK(transfer.request.postXfer() == NIXL_IN_PROG);
        URING_CHECK(waitFor(transfer) == NIXL_SUCCESS && partial_io_submissions == 3);
        const uintptr_t buffer = reinterpret_cast<uintptr_t>(test.buffers[0].data());
        for (size_t i = 0; i < partial_io_lengths.size(); i++) {
            const size_t completed = i * partial_size;
            URING_CHECK(partial_io_lengths[i] == block_size - completed &&
                        partial_io_offsets[i] == partial_io_offset + completed &&
                        partial_io_buffers[i] == buffer + completed);
        }
        std::array<char, partial_io_offset> actual_prefix{};
        std::array<char, block_size> actual{};
        URING_CHECK(pread(test.fd, actual_prefix.data(), actual_prefix.size(), 0) ==
                        static_cast<ssize_t>(actual_prefix.size()) &&
                    std::memcmp(actual_prefix.data(), prefix.data(), prefix.size()) == 0);
        if (read) {
            std::memcpy(actual.data(), test.buffers[0].data(), block_size);
        } else {
            URING_CHECK(pread(test.fd, actual.data(), actual.size(), partial_io_offset) ==
                        static_cast<ssize_t>(actual.size()));
        }
        URING_CHECK(std::memcmp(actual.data(), expected.data(), block_size) == 0);
    }
    for (const size_t file_size : {size_t{0}, block_size / 2}) {
        uringTest test(submit_mode_t::PASS_THROUGH);
        std::array<char, block_size> expected;
        std::memset(expected.data(), 0x6b, expected.size());
        if (file_size) {
            URING_CHECK(pwrite(test.fd, expected.data(), expected.size(), 0) ==
                            static_cast<ssize_t>(expected.size()) &&
                        ftruncate(test.fd, file_size) == 0);
        }
        std::memset(test.buffers[0].data(), 0, block_size);
        nixlPosixFileMD file_md(test.fd, "");
        uringRequest read(test, file_md, 0, NIXL_READ);
        URING_CHECK(read.request.postXfer() == NIXL_IN_PROG);
        URING_CHECK(waitFor(read) == NIXL_ERR_BACKEND && submit_calls == (file_size ? 2 : 1));
        URING_CHECK(!file_size ||
                    std::memcmp(test.buffers[0].data(), expected.data(), file_size) == 0);
    }
    for (const bool read : {true, false}) {
        uringTest test(submit_mode_t::PARTIAL_CANCEL);
        if (read) {
            URING_CHECK(pwrite(test.fd, test.buffers[0].data(), block_size, 0) ==
                        static_cast<ssize_t>(block_size));
        }
        completionState cancelled;
        URING_CHECK(test.queue->enqueue(test.fd,
                                        test.buffers[0].data(),
                                        block_size,
                                        0,
                                        read,
                                        completionCallback,
                                        &cancelled) == NIXL_SUCCESS);
        URING_CHECK(test.queue->post() == NIXL_IN_PROG);
        for (int i = 0; i < max_poll_iterations &&
             (!last_submitted_ring || io_uring_cq_ready(last_submitted_ring) == 0);
             i++) {
            std::this_thread::sleep_for(poll_pause);
        }
        URING_CHECK(last_submitted_ring && io_uring_cq_ready(last_submitted_ring) > 0);
        URING_CHECK(test.queue->cancel(&cancelled, cancelCompletionCallback) == 1);
        URING_CHECK(test.drain() == NIXL_SUCCESS && partial_io_submissions == 1);
        URING_CHECK(cancelled.count == 1 && cancelled.errors == 1 && cancel_completions == 1);
    }
    {
        uringTest test(submit_mode_t::PASS_THROUGH);
        nixlPosixFileMD file_md(test.fd, "");
        uringRequest cancelled(test, file_md, 0), unrelated(test, file_md, 1);
        nixl_status_t cancelled_status = cancelled.request.postXfer();
        URING_CHECK(cancelled_status >= NIXL_IN_PROG);
        URING_CHECK(test.queue->cancel(&cancelled.request, cancelCompletionCallback) == 1);

        URING_CHECK(unrelated.request.postXfer() >= NIXL_IN_PROG);
        URING_CHECK(waitFor(unrelated) == NIXL_SUCCESS);
        cancelled_status = waitFor(cancelled, cancelled_status);
        URING_CHECK(cancelled_status == NIXL_SUCCESS || cancelled_status == NIXL_ERR_BACKEND);
        URING_CHECK(test.drain() == NIXL_SUCCESS && cancel_completions == 1);
    }
    return 0;
}
