/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/utils.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {
void
check(bool value) {
    if (!value) {
        throw std::runtime_error("MEMOS reporting assertion failed");
    }
}

class FakeRT : public xferBenchRT {
public:
    FakeRT() {
        setSize(1);
        setRank(0);
    }

    int
    sendInt(int *, int) override {
        return 0;
    }

    int
    recvInt(int *, int) override {
        return 0;
    }

    int
    broadcastInt(int *, size_t, int) override {
        return 0;
    }

    int
    sendChar(char *, size_t, int) override {
        return 0;
    }

    int
    recvChar(char *, size_t, int) override {
        return 0;
    }

    int
    barrier(const std::string &, const bool) override {
        return 0;
    }

    int
    reduceSumDouble(double *local, double *global, int) override {
        *global = *local;
        return 0;
    }
};

std::string
render(size_t block, size_t batch, xferBenchStats stats) {
    std::ostringstream out;
    auto *saved = std::cout.rdbuf(out.rdbuf());
    try {
        xferBenchUtils::printStats(false, block, batch, std::move(stats));
    }
    catch (...) {
        std::cout.rdbuf(saved);
        throw;
    }
    std::cout.rdbuf(saved);
    return out.str();
}
} // namespace

int
main() {
    try {
        FakeRT runtime;
        xferBenchUtils::setRT(&runtime);
        xferBenchConfig::mode = XFERBENCH_MODE_SG;
        xferBenchConfig::scheme = XFERBENCH_SCHEME_PAIRWISE;
        xferBenchConfig::num_initiator_dev = 1;
        xferBenchConfig::num_threads = 1;
        xferBenchConfig::duration_seconds = 0;
        xferBenchConfig::num_iter = 1000;
        xferBenchConfig::large_blk_iter_ftr = 1;
        xferBenchStats legacy;
        legacy.total_duration.add(1000000);
        legacy.prepare_duration.add(1);
        std::istringstream old(render(1000000, 1, legacy));
        size_t block, batch;
        double bandwidth, amortized;
        std::string prep;
        old >> block >> batch >> bandwidth >> amortized >> prep;
        check(block == 1000000 && batch == 1 && bandwidth == 1.0 && prep == "1.0");

        xferBenchConfig::num_threads = 8;
        xferBenchConfig::warmup_seconds = 1;
        xferBenchConfig::duration_seconds = 30;
        xferBenchConfig::duration_latency_samples = 2;
        // Ignored legacy parameters must not cause division-by-zero, count
        // substitution, or adjustment of the true completion-based result.
        xferBenchConfig::num_iter = 0;
        xferBenchConfig::large_blk_iter_ftr = 0;
        xferBenchStats timed;
        auto &d = timed.duration;
        d.window_start_us = 1000000;
        d.window_end_us = 31000000;
        d.last_completion_us = 31000007;
        d.workers = d.workers_with_completions = d.workers_with_warmup = 8;
        d.warmup_batches = 8;
        d.measured_batches = 30;
        d.drain_batches = 3;
        d.issued_batches = d.completed_batches = 41;
        d.latency_batches = 28;
        d.latency_samples = 5;
        d.post_samples = 6;
        d.post_eligible_batches = 9;
        d.peak_batches = 16;
        timed.total_duration.add(999999999); // Must not affect steady-state denominator.
        timed.prepare_duration.add(999); // Must not appear as measured preparation.
        xferMetricStats storage;
        storage.preallocate(2);
        check(storage.count() == 0);
        storage.add(7);
        check(storage.count() == 1);
        storage.clear();
        check(storage.count() == 0);
        for (int i = 0; i < 5; ++i) {
            timed.transfer_duration.add(9);
        }
        for (int i = 0; i < 6; ++i) {
            timed.post_duration.add(7);
        }
        const auto output = render(2097152, 16, timed);
        check(output.find("measured_bytes=1006632960") != std::string::npos);
        check(output.find("drain_us=7") != std::string::npos);
        check(output.find("latency_samples_truncated=1") != std::string::npos);
        std::istringstream rows(output);
        std::string line;
        std::getline(rows, line);
        std::getline(rows, line);
        std::istringstream row(line);
        row >> block >> batch >> bandwidth >> amortized >> prep;
        check(block == 2097152 && batch == 16 && prep == "n/a");
        check(std::abs(bandwidth - 0.033554432) < 0.00000051);
        xferBenchConfig::duration_seconds = 31;
        try {
            render(2097152, 16, timed);
            throw std::runtime_error("wrong duration accepted");
        }
        catch (const std::invalid_argument &) {
        }
        xferBenchConfig::duration_seconds = 30;
        auto invalid = timed;
        invalid.duration.valid = false;
        try {
            render(2097152, 16, invalid);
            throw std::runtime_error("invalid stats accepted");
        }
        catch (const std::invalid_argument &) {
        }
        std::cout << "MEMOS native reporting regression tests passed\n";
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
