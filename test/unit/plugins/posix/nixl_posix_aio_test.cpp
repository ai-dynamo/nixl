/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <libaio.h>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

#include "io_queue.h"
#include "posix_backend.h"

namespace {
constexpr int request_count = 32, max_poll_iterations = 2000;
constexpr size_t block_size = 4096;
constexpr size_t partial_io_offset = 123;
constexpr size_t partial_read_size = block_size / 4;
constexpr size_t partial_write_size = 3 * block_size / 8;
constexpr auto poll_pause = std::chrono::microseconds(50);

enum class submitMode {
    PASS_THROUGH,
    PARTIAL_ONCE,
    TRANSIENT_ONCE,
    PARTIAL_THEN_ERROR,
    COMPLETION_ERROR,
    PARTIAL_READ,
    PARTIAL_WRITE,
    PARTIAL_CANCEL,
    TERMINAL_CONTEXT
};
submitMode submit_mode = submitMode::PASS_THROUGH;
int submit_calls = 0;
int completion_deferrals = 0;
int context_probe_calls = 0;
bool context_probe_requested_events = false;
long first_requested = 0;
int first_submitted = 0;
bool completion_error_injected = false;
int submissions_after_completion_error = 0;
std::vector<size_t> partial_io_lengths;
std::vector<size_t> partial_io_offsets;
std::vector<uintptr_t> partial_io_buffers;

struct completionState {
    int completions = 0;
    int errors = 0;
    int cancel_completions = 0;
};

[[noreturn]] void
failDlvsym(const char *symbol, const char *version) {
    const char *error = dlerror();
    std::cerr << "Failed to resolve " << symbol << "@" << version << ": "
              << (error ? error : "unknown error") << std::endl;
    std::abort();
}

void
completionCallback(void *ctx, uint32_t, int error) {
    auto *state = static_cast<completionState *>(ctx);
    state->completions++;
    state->errors += error != 0;
}

void
cancelCompletionCallback(void *ctx) {
    static_cast<completionState *>(ctx)->cancel_completions++;
}

struct aioTest {
    int fd = -1;
    std::vector<std::array<char, block_size>> buffers;
    std::unique_ptr<nixlPosixIOQueue> queue;

    explicit aioTest(size_t buffer_count = request_count)
        : buffers(buffer_count),
          queue(nixlPosixIOQueue::instantiate("AIO", 128, 16)) {
        char path[] = "/tmp/nixl_linux_aio_test_XXXXXX";
        if ((fd = mkstemp(path)) < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        unlink(path);
        for (size_t i = 0; i < buffers.size(); i++) {
            std::memset(buffers[i].data(), static_cast<int>(i + 1), buffers[i].size());
        }
    }

    ~aioTest() {
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

struct aioRequest {
    nixl_meta_dlist_t local{DRAM_SEG};
    nixl_meta_dlist_t remote{FILE_SEG};
    nixlPosixBackendReqH request;

    aioRequest(aioTest &test,
               nixlPosixFileMD &file_md,
               int first,
               int count,
               nixl_xfer_op_t op = NIXL_WRITE,
               size_t offset_bias = 0)
        : local([&] {
              nixl_meta_dlist_t list(DRAM_SEG);
              for (int i = first; i < first + count; i++) {
                  list.addDesc(nixlMetaDesc(
                      reinterpret_cast<uintptr_t>(test.buffers[i].data()), block_size, 0, nullptr));
              }
              return list;
          }()),
          remote([&] {
              nixl_meta_dlist_t list(FILE_SEG);
              for (int i = first; i < first + count; i++) {
                  list.addDesc(
                      nixlMetaDesc(offset_bias + i * block_size, block_size, test.fd, &file_md));
              }
              return list;
          }()),
          request(op, local, remote, test.queue) {}
};

nixl_status_t
waitFor(aioRequest &request, nixl_status_t status = NIXL_IN_PROG) {
    for (int i = 0; i < max_poll_iterations && status == NIXL_IN_PROG; i++) {
        status = request.request.checkXfer();
        std::this_thread::sleep_for(poll_pause);
    }
    return status;
}

void
setSubmitMode(submitMode mode) {
    submit_mode = mode;
    submit_calls = 0;
    completion_deferrals = 0;
    context_probe_calls = 0;
    context_probe_requested_events = false;
    first_requested = 0;
    first_submitted = 0;
    completion_error_injected = false;
    submissions_after_completion_error = 0;
    partial_io_lengths.clear();
    partial_io_offsets.clear();
    partial_io_buffers.clear();
}

#define AIO_CHECK(condition)                                                                      \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            std::cerr << "AIO_CHECK failed at line " << __LINE__ << ": " #condition << std::endl; \
            return 1;                                                                             \
        }                                                                                         \
    } while (false)
} // namespace

extern "C" int
io_submit(io_context_t ctx, long nr, struct iocb **iocbpp) {
    using submit_fn_t = int (*)(io_context_t, long, struct iocb **);
    static auto real_submit =
        reinterpret_cast<submit_fn_t>(dlvsym(RTLD_NEXT, "io_submit", "LIBAIO_0.1"));
    if (!real_submit) {
        failDlvsym("io_submit", "LIBAIO_0.1");
    }

    if (completion_error_injected) {
        submissions_after_completion_error++;
    }
    submit_calls++;
    if (nr > 0 &&
        (submit_mode == submitMode::PARTIAL_READ || submit_mode == submitMode::PARTIAL_WRITE)) {
        struct iocb *const iocb = iocbpp[0];
        partial_io_lengths.push_back(iocb->u.c.nbytes);
        partial_io_offsets.push_back(iocb->u.c.offset);
        partial_io_buffers.push_back(reinterpret_cast<uintptr_t>(iocb->u.c.buf));
        if (submit_mode == submitMode::PARTIAL_WRITE && partial_io_lengths.size() <= 2) {
            iocb->u.c.nbytes = partial_write_size;
        }
    }
    if (submit_calls == 1) {
        first_requested = nr;
        if (submit_mode == submitMode::TRANSIENT_ONCE) {
            return -EAGAIN;
        }
        if ((submit_mode == submitMode::PARTIAL_ONCE ||
             submit_mode == submitMode::PARTIAL_THEN_ERROR ||
             submit_mode == submitMode::COMPLETION_ERROR ||
             submit_mode == submitMode::TERMINAL_CONTEXT) &&
            nr > 1) {
            first_submitted = real_submit(ctx, nr / 2, iocbpp);
            return first_submitted;
        }
    } else if (submit_calls == 2) {
        if (submit_mode == submitMode::PARTIAL_THEN_ERROR) {
            return -EINVAL;
        }
        if (submit_mode == submitMode::TERMINAL_CONTEXT) {
            return -EINVAL;
        }
    }

    return real_submit(ctx, nr, iocbpp);
}

extern "C" int
io_cancel(io_context_t ctx, struct iocb *iocb, struct io_event *event) {
    using cancel_fn_t = int (*)(io_context_t, struct iocb *, struct io_event *);
    static auto real_cancel =
        reinterpret_cast<cancel_fn_t>(dlvsym(RTLD_NEXT, "io_cancel", "LIBAIO_0.4"));
    if (!real_cancel) {
        failDlvsym("io_cancel", "LIBAIO_0.4");
    }
    if (submit_mode == submitMode::PARTIAL_THEN_ERROR ||
        submit_mode == submitMode::COMPLETION_ERROR || submit_mode == submitMode::PARTIAL_CANCEL) {
        return -EAGAIN;
    }
    return real_cancel(ctx, iocb, event);
}

extern "C" int
io_getevents(io_context_t ctx,
             long min_nr,
             long nr,
             struct io_event *events,
             struct timespec *timeout) {
    using getevents_fn_t = int (*)(io_context_t, long, long, struct io_event *, struct timespec *);
    static auto real_getevents =
        reinterpret_cast<getevents_fn_t>(dlvsym(RTLD_NEXT, "io_getevents", "LIBAIO_0.4"));
    if (!real_getevents) {
        failDlvsym("io_getevents", "LIBAIO_0.4");
    }
    if (submit_mode == submitMode::TERMINAL_CONTEXT) {
        context_probe_calls++;
        context_probe_requested_events |= min_nr != 0 || nr != 0;
        return context_probe_calls == 1 ? -EINTR : -EINVAL;
    }
    if (min_nr == 0 && nr == 0) {
        context_probe_calls++;
        return real_getevents(ctx, min_nr, nr, events, timeout);
    }
    if (submit_mode == submitMode::PARTIAL_THEN_ERROR && completion_deferrals++ == 0) {
        return 0;
    }
    if (submit_mode == submitMode::COMPLETION_ERROR && !completion_error_injected) {
        struct timespec completion_timeout = {0, 10 * 1000 * 1000}; // 10 ms
        int rc = real_getevents(ctx, 1, nr, events, &completion_timeout);
        if (rc > 0) {
            events[0].res = -EIO;
            completion_error_injected = true;
        }
        return rc;
    }
    const int rc = real_getevents(ctx, min_nr, nr, events, timeout);
    if (rc > 0 && partial_io_lengths.size() <= 2 &&
        (submit_mode == submitMode::PARTIAL_READ || submit_mode == submitMode::PARTIAL_WRITE)) {
        if (submit_mode == submitMode::PARTIAL_READ) {
            events[0].res = partial_read_size;
        }
    }
    if (rc > 0 && submit_mode == submitMode::PARTIAL_CANCEL) {
        events[0].res /= 2;
    }
    return rc;
}

int
main() {
    {
        setSubmitMode(submitMode::PARTIAL_ONCE);
        aioTest test;
        completionState state;
        AIO_CHECK(test.enqueue(state, 0, request_count));
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(first_submitted > 0 && first_submitted < first_requested);
        AIO_CHECK(test.drain() == NIXL_SUCCESS);
        AIO_CHECK(state.completions == request_count && state.errors == 0);
    }
    {
        setSubmitMode(submitMode::TRANSIENT_ONCE);
        aioTest test;
        completionState state;
        AIO_CHECK(test.enqueue(state, 0, request_count));
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(test.drain() == NIXL_SUCCESS);
        AIO_CHECK(submit_calls > 1);
        AIO_CHECK(state.completions == request_count && state.errors == 0);
    }
    for (const bool read : {true, false}) {
        const size_t partial_size = read ? partial_read_size : partial_write_size;
        setSubmitMode(read ? submitMode::PARTIAL_READ : submitMode::PARTIAL_WRITE);
        aioTest test(1);
        std::array<char, partial_io_offset> prefix;
        std::array<char, block_size> expected;
        std::memset(prefix.data(), 0x3c, prefix.size());
        std::memset(expected.data(), read ? 0x5a : 0x1, expected.size());
        AIO_CHECK(pwrite(test.fd, prefix.data(), prefix.size(), 0) ==
                  static_cast<ssize_t>(prefix.size()));
        if (read) {
            AIO_CHECK(pwrite(test.fd, expected.data(), expected.size(), partial_io_offset) ==
                      static_cast<ssize_t>(expected.size()));
            std::memset(test.buffers[0].data(), 0, block_size);
        }
        nixlPosixFileMD file_md(test.fd, "");
        aioRequest transfer(test, file_md, 0, 1, read ? NIXL_READ : NIXL_WRITE, partial_io_offset);
        AIO_CHECK(transfer.request.postXfer() == NIXL_IN_PROG);
        AIO_CHECK(waitFor(transfer) == NIXL_SUCCESS);
        AIO_CHECK(partial_io_lengths.size() == 3);

        const uintptr_t buffer = reinterpret_cast<uintptr_t>(test.buffers[0].data());
        for (size_t i = 0; i < 3; i++) {
            const size_t completed = i * partial_size;
            AIO_CHECK(partial_io_lengths[i] == block_size - completed &&
                      partial_io_offsets[i] == partial_io_offset + completed &&
                      partial_io_buffers[i] == buffer + completed);
        }

        std::array<char, partial_io_offset> actual_prefix{};
        std::array<char, block_size> actual{};
        AIO_CHECK(pread(test.fd, actual_prefix.data(), actual_prefix.size(), 0) ==
                      static_cast<ssize_t>(actual_prefix.size()) &&
                  std::memcmp(actual_prefix.data(), prefix.data(), prefix.size()) == 0);
        if (read) {
            std::memcpy(actual.data(), test.buffers[0].data(), block_size);
        } else {
            AIO_CHECK(pread(test.fd, actual.data(), actual.size(), partial_io_offset) ==
                      static_cast<ssize_t>(actual.size()));
        }
        AIO_CHECK(std::memcmp(actual.data(), expected.data(), block_size) == 0);
    }
    for (const size_t file_size : {size_t{0}, block_size / 2}) {
        setSubmitMode(submitMode::PASS_THROUGH);
        aioTest test(1);
        std::array<char, block_size> expected;
        std::memset(expected.data(), 0x6b, expected.size());
        if (file_size) {
            AIO_CHECK(pwrite(test.fd, expected.data(), expected.size(), 0) ==
                          static_cast<ssize_t>(expected.size()) &&
                      ftruncate(test.fd, file_size) == 0);
        }
        std::memset(test.buffers[0].data(), 0, block_size);

        nixlPosixFileMD file_md(test.fd, "");
        aioRequest read(test, file_md, 0, 1, NIXL_READ);
        AIO_CHECK(read.request.postXfer() == NIXL_IN_PROG);
        AIO_CHECK(waitFor(read) == NIXL_ERR_BACKEND && submit_calls == (file_size ? 2 : 1));
        AIO_CHECK(!file_size ||
                  std::memcmp(test.buffers[0].data(), expected.data(), file_size) == 0);
    }
    {
        setSubmitMode(submitMode::PARTIAL_THEN_ERROR);
        aioTest test(request_count + 1);
        nixlPosixFileMD file_md(test.fd, "");
        aioRequest failed(test, file_md, 0, request_count);
        aioRequest unrelated(test, file_md, request_count, 1);

        nixl_status_t failed_status = failed.request.postXfer();
        AIO_CHECK(failed_status == NIXL_IN_PROG);
        failed_status = failed.request.checkXfer();
        AIO_CHECK(failed_status == NIXL_IN_PROG);
        AIO_CHECK(completion_deferrals == 1);
        AIO_CHECK(context_probe_calls == 1);

        // The rejected IOCB fails its request, whose submitted prefix is then drained.
        // Unrelated requests and later reposts keep using the healthy AIO context.
        setSubmitMode(submitMode::PASS_THROUGH);
        AIO_CHECK(unrelated.request.postXfer() == NIXL_IN_PROG);
        AIO_CHECK(waitFor(unrelated) == NIXL_SUCCESS);
        AIO_CHECK(waitFor(failed, failed_status) == NIXL_ERR_BACKEND);
        AIO_CHECK(failed.request.postXfer() == NIXL_IN_PROG);
        AIO_CHECK(waitFor(failed) == NIXL_SUCCESS);
    }
    {
        constexpr int completion_error_request_count = 65;
        setSubmitMode(submitMode::COMPLETION_ERROR);
        aioTest test(completion_error_request_count);
        nixlPosixFileMD file_md(test.fd, "");
        aioRequest failed(test, file_md, 0, completion_error_request_count);

        AIO_CHECK(failed.request.postXfer() == NIXL_IN_PROG);
        AIO_CHECK(first_submitted > 0 && first_submitted < first_requested);
        AIO_CHECK(waitFor(failed) == NIXL_ERR_BACKEND);
        AIO_CHECK(completion_error_injected);
        AIO_CHECK(submissions_after_completion_error == 0);
    }
    {
        setSubmitMode(submitMode::TERMINAL_CONTEXT);
        aioTest test;
        completionState state;
        AIO_CHECK(test.enqueue(state, 0, request_count));
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(first_submitted > 0 && first_submitted < first_requested);
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(context_probe_calls == 2);
        AIO_CHECK(!context_probe_requested_events);
        AIO_CHECK(state.completions == request_count && state.errors == request_count);
        AIO_CHECK(test.queue->poll() == NIXL_ERR_BACKEND);
        AIO_CHECK(test.queue->enqueue(test.fd,
                                      test.buffers[0].data(),
                                      block_size,
                                      0,
                                      false,
                                      completionCallback,
                                      &state) == NIXL_ERR_BACKEND);
    }
    for (const bool read : {true, false}) {
        setSubmitMode(submitMode::PARTIAL_CANCEL);
        aioTest test(1);
        if (read) {
            AIO_CHECK(pwrite(test.fd, test.buffers[0].data(), block_size, 0) ==
                      static_cast<ssize_t>(block_size));
        }
        completionState cancelled;
        AIO_CHECK(test.queue->enqueue(test.fd,
                                      test.buffers[0].data(),
                                      block_size,
                                      0,
                                      read,
                                      completionCallback,
                                      &cancelled) == NIXL_SUCCESS);
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(submit_calls == 1);

        const unsigned waits = test.queue->cancel(&cancelled, cancelCompletionCallback);
        AIO_CHECK(waits == 0 && test.drain() == NIXL_SUCCESS);
        AIO_CHECK(submit_calls == 1);
        AIO_CHECK(cancelled.completions == 1 && cancelled.errors == 1);
    }
    {
        constexpr int scoped_request_count = 65;
        setSubmitMode(submitMode::PASS_THROUGH);
        aioTest test(scoped_request_count + 1);
        completionState cancelled, unrelated;
        AIO_CHECK(test.enqueue(cancelled, 0, scoped_request_count));
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);

        unsigned waits = test.queue->cancel(&cancelled, cancelCompletionCallback);
        AIO_CHECK(waits == 0);
        AIO_CHECK(test.enqueue(unrelated, scoped_request_count, 1));
        AIO_CHECK(test.queue->post() == NIXL_IN_PROG);
        AIO_CHECK(test.drain() == NIXL_SUCCESS);
        AIO_CHECK(cancelled.completions == scoped_request_count && cancelled.errors > 0);
        AIO_CHECK(cancelled.cancel_completions == static_cast<int>(waits));
        AIO_CHECK(unrelated.completions == 1 && unrelated.errors == 0);
    }

    return 0;
}
