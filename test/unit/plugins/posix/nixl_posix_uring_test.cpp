/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <dirent.h>
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

int
runPosixUringPathAgentTest();

namespace {
constexpr int request_count = 32, ring_entries = 16, max_poll_iterations = 2000;
constexpr size_t block_size = 4096;
constexpr auto poll_pause = std::chrono::microseconds(50);
using buffers_t = std::array<std::array<char, block_size>, request_count>;

enum class submit_mode_t { PARTIAL_ONLY, TRANSIENT_ERRORS, PASS_THROUGH };
submit_mode_t submit_mode = submit_mode_t::PASS_THROUGH;
int submit_calls = 0, transient_submit_errors = 0, cancel_completions = 0;
unsigned first_ready = 0, first_submitted = 0;

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
        char path[] = "/tmp/nixl_uring_test_XXXXXX";
        if ((fd = mkstemp(path)) < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        unlink(path);
        if (queue->registerFile(fd, "") != NIXL_SUCCESS) {
            close(fd);
            throw std::runtime_error("file registration failed");
        }
        for (size_t i = 0; i < buffers.size(); i++) {
            std::memset(buffers[i].data(), static_cast<int>(i + 1), buffers[i].size());
        }
    }

    ~uringTest() {
        queue->deregisterFile(fd);
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

    uringRequest(uringTest &test, nixlPosixFileMD &file_md, int index)
        : local([&] {
              nixl_meta_dlist_t list(DRAM_SEG);
              list.addDesc(nixlMetaDesc(
                  reinterpret_cast<uintptr_t>(test.buffers[index].data()), block_size, 0, nullptr));
              return list;
          }()),
          remote([&] {
              nixl_meta_dlist_t list(FILE_SEG);
              list.addDesc(nixlMetaDesc(index * block_size, block_size, test.fd, &file_md));
              return list;
          }()),
          request(operation, local, remote, test.queue) {}
};

int
countProcessFds();

bool
runEagerPathModeTest() {
    char path[] = "/tmp/nixl_uring_eager_path_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        return false;
    }

    std::array<char, block_size> expected;
    expected.fill(0x5a);
    bool success =
        write(fd, expected.data(), expected.size()) == static_cast<ssize_t>(expected.size());
    close(fd);
    if (!success) {
        unlink(path);
        return false;
    }

    auto queue = nixlPosixIOQueue::instantiate("URING", 64, ring_entries);
    if (!queue) {
        unlink(path);
        return false;
    }
    const int fds_before_open = countProcessFds();
    if (fds_before_open < 0) {
        unlink(path);
        return false;
    }

    constexpr int dev_id = 1;
    std::array<char, block_size> buffer{};
    completionState state;
    // A transfer posted before the open CQE is reaped must remain deferred until poll().
    success = queue->registerFile(dev_id, std::string("ro:") + path) == NIXL_SUCCESS &&
        queue->enqueue(dev_id, buffer.data(), buffer.size(), 0, true, completionCallback, &state) ==
            NIXL_SUCCESS &&
        queue->deregisterFile(dev_id) == NIXL_ERR_NOT_ALLOWED && queue->post() == NIXL_IN_PROG &&
        state.count == 0;

    for (int i = 0; success && state.count == 0 && i < max_poll_iterations; ++i) {
        success = queue->poll() >= 0;
        std::this_thread::sleep_for(poll_pause);
    }

    success = success && state.count == 1 && state.errors == 0 && buffer == expected;

    if (queue->deregisterFile(dev_id) != NIXL_SUCCESS) {
        success = false;
    }
    success = success && countProcessFds() == fds_before_open;

    unlink(path);
    return success;
}

int
countProcessFds() {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) {
        return -1;
    }
    int count = 0;
    while (readdir(dir)) {
        ++count;
    }
    closedir(dir);
    return count;
}

bool
runEagerOpenShutdownTest() {
    const int before = countProcessFds();
    if (before < 0) {
        return false;
    }

    {
        auto queue = nixlPosixIOQueue::instantiate("URING", 1, ring_entries);
        if (!queue) {
            return false;
        }

        for (int i = 0; i < request_count; ++i) {
            if (queue->registerFile(i, "ro:/dev/null") != NIXL_SUCCESS) {
                return false;
            }
        }
        // Queue destruction must reap pending OPENAT CQEs and close every resulting fd.
    }

    return countProcessFds() == before;
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
    if (submit_mode == submit_mode_t::TRANSIENT_ERRORS && transient_submit_errors == 0) {
        transient_submit_errors++;
        return -EAGAIN;
    }

    const unsigned ready = io_uring_sq_ready(ring);
    if (ready == 0 || submit_mode == submit_mode_t::PASS_THROUGH || ++submit_calls != 1 ||
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
    int probe_ret = io_uring_queue_init(ring_entries, &probe_ring, 0);
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
    {
        uringTest test(submit_mode_t::PASS_THROUGH);
        nixlPosixFileMD file_md(test.fd);
        uringRequest cancelled(test, file_md, 0), unrelated(test, file_md, 1);
        nixl_status_t cancelled_status = cancelled.request.postXfer();
        URING_CHECK(cancelled_status >= NIXL_IN_PROG);
        URING_CHECK(test.queue->cancel(&cancelled.request, cancelCompletionCallback) == 1);

        nixl_status_t status = unrelated.request.postXfer();
        URING_CHECK(status >= NIXL_IN_PROG);
        for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
            status = unrelated.request.checkXfer();
            std::this_thread::sleep_for(poll_pause);
        }
        URING_CHECK(status == NIXL_SUCCESS);

        for (int i = 0; i < max_poll_iterations && cancelled_status == NIXL_IN_PROG; i++) {
            cancelled_status = cancelled.request.checkXfer();
            std::this_thread::sleep_for(poll_pause);
        }
        URING_CHECK(cancelled_status == NIXL_SUCCESS || cancelled_status == NIXL_ERR_BACKEND);
        URING_CHECK(test.drain() == NIXL_SUCCESS && cancel_completions == 1);
    }

    URING_CHECK(runEagerPathModeTest());
    URING_CHECK(runEagerOpenShutdownTest());
    URING_CHECK(runPosixUringPathAgentTest() == 0);
    std::cout << "io_uring eager path opens: regular-fd=tested" << std::endl;

    return 0;
}
