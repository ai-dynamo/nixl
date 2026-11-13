/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Exercise the production batch state machine without a GPU or cuFile driver.
// Linker wrappers supply completion events and copy their reported payloads.
#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gds_batch_engine.h"

namespace {

constexpr ssize_t full = std::numeric_limits<ssize_t>::max();

#define TEST_CHECK(condition)                                                  \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error(std::string(__func__) + ": " #condition); \
        }                                                                      \
    } while (false)

struct Event {
    unsigned int index;
    ssize_t bytes = full;
    decltype(CUfileIOEvents_t::status) status = CUFILE_COMPLETE;
    bool bad_cookie = false;
};

struct Poll {
    std::vector<Event> events;
    bool fail = false;
};

struct Generation {
    std::deque<Poll> polls;
    bool fail_submit = false;
};

struct Submission {
    CUfileIOParams_t *params;
    std::vector<CUfileIOParams_t> snapshot;
    std::vector<bool> complete;
    unsigned int flags;
};

struct MockState {
    unsigned int setups = 0;
    unsigned int cancels = 0;
    std::deque<Generation> generations;
    Generation current;
    std::vector<Submission> submissions;
    std::vector<char> file = std::vector<char>(65536, 'A');
};

MockState mock;

CUfileError_t
error() {
    CUfileError_t result{};
    result.err = static_cast<decltype(result.err)>(5001);
    return result;
}

void
checkLiveParameters() {
    const auto &submission = mock.submissions.back();
    for (size_t i = 0; i < submission.snapshot.size(); ++i) {
        const auto &old = submission.snapshot[i];
        const auto &now = submission.params[i];
        TEST_CHECK(now.fh == old.fh && now.opcode == old.opcode && now.cookie == old.cookie);
        TEST_CHECK(now.u.batch.devPtr_base == old.u.batch.devPtr_base);
        TEST_CHECK(now.u.batch.file_offset == old.u.batch.file_offset);
        TEST_CHECK(now.u.batch.devPtr_offset == old.u.batch.devPtr_offset);
        TEST_CHECK(now.u.batch.size == old.u.batch.size);
    }
}

struct Fixture {
    std::vector<char> buffer = std::vector<char>(16384, 'Q');
    nixlGdsIOBatch batch;

    explicit Fixture(unsigned int count = 1) : batch(count) {}

    void
    add(size_t size = 8192,
        size_t file_offset = 16384,
        size_t buffer_offset = 1024,
        CUfileOpcode_t op = CUFILE_WRITE) {
        TEST_CHECK(batch.addToBatch({}, buffer.data(), size, file_offset, buffer_offset, op) ==
                   NIXL_SUCCESS);
    }

    void
    start() {
        TEST_CHECK(batch.submitBatch(0) == NIXL_SUCCESS);
    }

    void
    tail(size_t submission, size_t size, size_t file_offset, size_t buffer_offset) {
        TEST_CHECK(mock.submissions.at(submission).snapshot.size() == 1);
        const auto &params = mock.submissions[submission].snapshot[0];
        TEST_CHECK(params.u.batch.devPtr_base == buffer.data());
        TEST_CHECK(params.u.batch.size == size);
        TEST_CHECK(params.u.batch.file_offset == static_cast<off_t>(file_offset));
        TEST_CHECK(params.u.batch.devPtr_offset == static_cast<off_t>(buffer_offset));
        TEST_CHECK(params.cookie == mock.submissions[submission].params);
        TEST_CHECK(mock.submissions[submission].flags == 0);
    }
};

void
shortTransfer(CUfileOpcode_t op) {
    mock = {};
    mock.generations = {Generation{{Poll{{{0, 4096}}}}}};
    Fixture f;
    if (op == CUFILE_READ) {
        std::fill(mock.file.begin(), mock.file.end(), 'R');
    }
    f.add(8192, 16384, 1024, op);
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    TEST_CHECK(mock.submissions.size() == 2);
    f.tail(1, 4096, 20480, 5120);
    TEST_CHECK(mock.submissions[1].snapshot[0].opcode == op);
    TEST_CHECK(f.batch.checkStatus() == NIXL_SUCCESS);
    TEST_CHECK(mock.setups == 1 && mock.submissions.size() == 2 && mock.cancels == 0);
    if (op == CUFILE_READ) {
        TEST_CHECK(std::all_of(
            f.buffer.begin() + 1024, f.buffer.begin() + 9216, [](char c) { return c == 'R'; }));
        TEST_CHECK(f.buffer[1023] == 'Q' && f.buffer[9216] == 'Q');
    } else {
        TEST_CHECK(std::all_of(
            mock.file.begin() + 16384, mock.file.begin() + 24576, [](char c) { return c == 'Q'; }));
        TEST_CHECK(mock.file[16383] == 'A' && mock.file[24576] == 'A');
    }
}

void
repeatedShortTransfers() {
    mock = {};
    mock.generations = {Generation{{Poll{{{0, 4096}}}}}, Generation{{Poll{{{0, 2048}}}}}};
    Fixture f;
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    f.tail(2, 2048, 22528, 7168);
    TEST_CHECK(f.batch.checkStatus() == NIXL_SUCCESS);
    TEST_CHECK(mock.setups == 1 && mock.submissions.size() == 3);
    // A reused pool slot starts from the original request, with no stale tail.
    f.batch.reset();
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_SUCCESS);
    TEST_CHECK(mock.submissions.size() == 4);
}

void
outOfOrderCompletions() {
    mock = {};
    mock.generations = {Generation{{Poll{}, Poll{{{2, 2048}}}, Poll{{{0}}}, Poll{{{1, 1024}}}}}};
    Fixture f(3);
    f.add(4096, 0, 0);
    f.add(4096, 8192, 4096);
    f.add(4096, 16384, 8192);
    f.start();
    for (unsigned int i = 0; i < 3; ++i) {
        TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
        TEST_CHECK(mock.submissions.size() == 1);
    }
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    TEST_CHECK(mock.submissions.size() == 2);
    const auto &tails = mock.submissions[1].snapshot;
    TEST_CHECK(tails.size() == 2);
    TEST_CHECK(tails[0].u.batch.file_offset == 18432 && tails[0].u.batch.devPtr_offset == 10240);
    TEST_CHECK(tails[0].u.batch.size == 2048);
    TEST_CHECK(tails[1].u.batch.file_offset == 9216 && tails[1].u.batch.devPtr_offset == 5120);
    TEST_CHECK(tails[1].u.batch.size == 3072);
    TEST_CHECK(f.batch.checkStatus() == NIXL_SUCCESS);
    TEST_CHECK(mock.setups == 1);
}

void
invalidCompletion(Event event) {
    mock = {};
    mock.generations = {Generation{{Poll{{event}}}}};
    Fixture f;
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_ERR_BACKEND);
    TEST_CHECK(f.batch.drain());
    TEST_CHECK(mock.submissions.size() == 1 && mock.cancels == 0);
}

void
failureDiscardsTails() {
    mock = {};
    mock.generations = {
        Generation{{Poll{{{0, 2048}}}, Poll{{{1, -5, CUFILE_FAILED}}}, Poll{{{2}}}}}};
    Fixture f(3);
    for (unsigned int i = 0; i < 3; ++i) {
        f.add(4096, i * 8192, i * 4096);
    }
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    TEST_CHECK(f.batch.checkStatus() == NIXL_ERR_BACKEND);
    TEST_CHECK(f.batch.drain());
    TEST_CHECK(mock.submissions.size() == 1 && mock.cancels == 0);
}

void
tailSubmitFailure() {
    mock = {};
    mock.generations = {Generation{{Poll{{{0, 4096}}}}}, Generation{{}, true}};
    Fixture f;
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_ERR_BACKEND);
    TEST_CHECK(f.batch.drain());
    TEST_CHECK(mock.submissions.size() == 1 && mock.cancels == 0);
    f.batch.reset();
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_SUCCESS);
}

void
noProgressAfterShortTransfer() {
    mock = {};
    mock.generations = {Generation{{Poll{{{0, 4096}}}}}, Generation{{Poll{{{0, 0}}}}}};
    Fixture f;
    f.add();
    f.start();
    TEST_CHECK(f.batch.checkStatus() == NIXL_IN_PROG);
    TEST_CHECK(f.batch.checkStatus() == NIXL_ERR_BACKEND);
    TEST_CHECK(f.batch.drain());
    TEST_CHECK(mock.submissions.size() == 2 && mock.cancels == 0);
}

} // namespace

extern "C" CUfileError_t
__wrap_cuFileBatchIOSetUp(CUfileBatchHandle_t *handle, unsigned int) {
    ++mock.setups;
    *handle = reinterpret_cast<CUfileBatchHandle_t>(&mock);
    return {};
}

extern "C" void
__wrap_cuFileBatchIODestroy(CUfileBatchHandle_t) {}

extern "C" CUfileError_t
__wrap_cuFileBatchIOCancel(CUfileBatchHandle_t) {
    ++mock.cancels;
    return {};
}

extern "C" CUfileError_t
__wrap_cuFileBatchIOSubmit(CUfileBatchHandle_t handle,
                           unsigned int nr,
                           CUfileIOParams_t *params,
                           unsigned int flags) {
    TEST_CHECK(handle == reinterpret_cast<CUfileBatchHandle_t>(&mock));
    if (!mock.submissions.empty()) {
        const auto &complete = mock.submissions.back().complete;
        TEST_CHECK(std::all_of(complete.begin(), complete.end(), [](bool done) { return done; }));
    }
    mock.current = {};
    if (!mock.generations.empty()) {
        mock.current = std::move(mock.generations.front());
        mock.generations.pop_front();
    }
    if (mock.current.fail_submit) {
        return error();
    }
    mock.submissions.push_back({params, {params, params + nr}, std::vector<bool>(nr), flags});
    return {};
}

extern "C" CUfileError_t
__wrap_cuFileBatchIOGetStatus(CUfileBatchHandle_t,
                              unsigned int min_nr,
                              unsigned int *nr,
                              CUfileIOEvents_t *events,
                              struct timespec *timeout) {
    TEST_CHECK(min_nr == 0 && timeout != nullptr && timeout->tv_sec == 0 && timeout->tv_nsec == 0);
    checkLiveParameters();
    auto &submission = mock.submissions.back();
    Poll poll;
    if (mock.current.polls.empty()) {
        for (unsigned int i = 0; i < submission.complete.size(); ++i) {
            if (!submission.complete[i]) {
                poll.events.push_back({i});
            }
        }
    } else {
        poll = std::move(mock.current.polls.front());
        mock.current.polls.pop_front();
    }
    if (poll.fail) {
        return error();
    }
    TEST_CHECK(poll.events.size() <= *nr);
    *nr = poll.events.size();
    for (unsigned int i = 0; i < *nr; ++i) {
        const auto &event = poll.events[i];
        const auto &params = submission.snapshot.at(event.index);
        const size_t done =
            event.bytes == full ? params.u.batch.size : static_cast<size_t>(event.bytes);
        if (event.status == CUFILE_COMPLETE && done > 0 && done <= params.u.batch.size) {
            const auto &io = params.u.batch;
            TEST_CHECK(io.file_offset >= 0 && io.file_offset + done <= mock.file.size());
            auto *buffer = static_cast<char *>(io.devPtr_base) + io.devPtr_offset;
            auto *file = mock.file.data() + io.file_offset;
            if (params.opcode == CUFILE_WRITE) {
                std::memcpy(file, buffer, done);
            } else {
                std::memcpy(buffer, file, done);
            }
        }
        events[i].cookie = event.bad_cookie ? static_cast<void *>(&mock) : params.cookie;
        events[i].status = event.status;
        events[i].ret = done;
        if (event.status != CUFILE_WAITING && event.status != CUFILE_PENDING) {
            TEST_CHECK(!submission.complete[event.index]);
            submission.complete[event.index] = true;
        }
    }
    return {};
}

int
main() {
    try {
        shortTransfer(CUFILE_WRITE);
        shortTransfer(CUFILE_READ);
        repeatedShortTransfers();
        outOfOrderCompletions();
        for (ssize_t bytes : {ssize_t(0), ssize_t(-5), ssize_t(8193)}) {
            invalidCompletion({0, bytes});
        }
        invalidCompletion({0, full, CUFILE_COMPLETE, true});
        for (auto status : {CUFILE_FAILED, CUFILE_CANCELED, CUFILE_TIMEOUT, CUFILE_INVALID}) {
            invalidCompletion({0, -5, status});
        }
        failureDiscardsTails();
        tailSubmitFailure();
        noProgressAfterShortTransfer();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "GDS batch short-completion tests passed\n";
    return 0;
}
