/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * Telemetry hot-path microbenchmark.
 *
 * Purpose: give a fast, repeatable, in-process proxy for the per-transfer
 * telemetry tax that nixlbench sees on small messages, so a fix can be judged
 * against measured baselines instead of guesses. It measures the pieces the
 * datapath actually pays with telemetry ON:
 *   - a single steady_clock::now() read (the agent takes 3 per transfer),
 *   - nixlTelemetry::addXferStats() in the append regime (buffer has room),
 *   - addXferStats() in the drop regime (buffer saturated -- the regime that
 *     dominates at datapath rates, where the flush cannot keep up),
 *   - the full per-transfer sequence the agent runs (3 now() + 2 duration_casts
 *     + addXferStats), as a proxy for the total datapath tax.
 *
 * These cases are DISABLED_ so they never run in the normal gtest suite (no CI
 * flakiness, no perf gate). Run them on demand via CTest:
 *     ctest --test-dir build -R telemetry_benchmark -V
 * or directly:
 *     build/test/gtest/gtest --gtest_also_run_disabled_tests \
 *         --gtest_filter='*telemetryBenchmark*'
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include "telemetry.h"
#include "telemetry_event.h"
#include "nixl_duration.h"
#include "common.h"

namespace fs = std::filesystem;

namespace {

// Repeatedly time `body` over `iters` iterations, `repeats` times, and return
// the best (minimum) nanoseconds-per-iteration observed. The minimum is the
// most stable estimator for a fixed-cost hot path: it discards samples polluted
// by scheduler preemption, the periodic flush thread, and page faults.
template<typename Body>
[[nodiscard]] double
bestNsPerOp(size_t iters, size_t repeats, Body &&body) {
    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            body(i);
        }
        const auto end = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        best = std::min(best, ns / static_cast<double>(iters));
    }
    return best;
}

void
report(const std::string &label, double ns_per_op) {
    std::cout << "[ telem-bench ] " << label << ": " << ns_per_op << " ns/op" << std::endl;
}

} // namespace

class telemetryBenchmark : public ::testing::Test {
protected:
    void
    SetUp() override {
        dir_ = fs::temp_directory_path() / "nixl_telemetry_bench";
        fs::create_directories(dir_);
        env_.addVar("NIXL_TELEMETRY_ENABLE", "y");
        env_.addVar("NIXL_TELEMETRY_DIR", dir_.string());
    }

    void
    TearDown() override {
        env_.popVar();
        env_.popVar();
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    static constexpr size_t kIters = 200000;
    static constexpr size_t kRepeats = 7;

    fs::path dir_;
    gtest::ScopedEnv env_;
};

// Building block: cost of one steady_clock::now(). The agent takes three of
// these per completed transfer (post start, post-time, xfer-time), so this
// times the largest fixed non-queue cost.
TEST_F(telemetryBenchmark, DISABLED_ClockNowRead) {
    volatile int64_t sink = 0;
    const double ns = bestNsPerOp(kIters, kRepeats, [&](size_t) {
        sink = sink + std::chrono::steady_clock::now().time_since_epoch().count();
    });
    (void)sink;
    report("clock_now_read", ns);
}

// Raw C clock_gettime(CLOCK_MONOTONIC): what steady_clock::now() calls under the
// hood on Linux. Confirms the C++ wrapper adds no meaningful overhead -- the cost
// is the vDSO clock read, not std::chrono.
TEST_F(telemetryBenchmark, DISABLED_ClockGettimeMonotonic) {
    volatile int64_t sink = 0;
    const double ns = bestNsPerOp(kIters, kRepeats, [&](size_t) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        sink = sink + ts.tv_nsec;
    });
    (void)sink;
    report("clock_gettime_monotonic", ns);
}

// Building block: cost of a raw hardware-counter read, the candidate cheap
// replacement for steady_clock::now() on the datapath -- rdtsc on x86_64, the
// cntvct_el0 virtual counter on aarch64. Reports -1 on other architectures.
TEST_F(telemetryBenchmark, DISABLED_HwCounterRead) {
#if defined(__x86_64__)
    volatile uint64_t sink = 0;
    const double ns = bestNsPerOp(kIters, kRepeats, [&](size_t) {
        unsigned lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        sink = sink + (((static_cast<uint64_t>(hi) << 32) | lo));
    });
    (void)sink;
    report("hw_counter_read", ns);
#elif defined(__aarch64__)
    volatile uint64_t sink = 0;
    const double ns = bestNsPerOp(kIters, kRepeats, [&](size_t) {
        uint64_t cnt;
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(cnt));
        sink = sink + cnt;
    });
    (void)sink;
    report("hw_counter_read", ns);
#else
    report("hw_counter_read", -1.0);
#endif
}

// Cost of a nixlDuration measurement -- the CPU-counter-backed stopwatch the
// datapath uses for per-transfer durations. Compare against clock_now_read.
TEST_F(telemetryBenchmark, DISABLED_NixlDurationElapsed) {
    nixlTime::nixlDuration timer;
    volatile int64_t sink = 0;
    const double ns =
        bestNsPerOp(kIters, kRepeats, [&](size_t) { sink = sink + timer.elapsed().count(); });
    (void)sink;
    report(nixlTime::fastClockUsesHwCounter() ? "nixl_duration_elapsed (hw)" :
                                                "nixl_duration_elapsed (fallback)",
           ns);
}

// Per-transfer tax with the new split: a steady_clock::now() for the exposed
// absolute startTime plus a nixlDuration stopwatch for the durations. The
// before/after for the fix. Compare against per_transfer_tax_saturated.
TEST_F(telemetryBenchmark, DISABLED_PerTransferTaxNixlDuration) {
    constexpr size_t buf_cap = 64;
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(buf_cap));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "3600000");

    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < kRepeats; ++r) {
        nixlTelemetry telemetry("bench_tax_fast", "NOP");
        for (size_t i = 0; i < buf_cap; ++i) {
            telemetry.updateTxBytes(i);
        }
        volatile int64_t sink = 0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIters; ++i) {
            const auto startTime = std::chrono::steady_clock::now();
            nixlTime::nixlDuration timer;
            const auto post = timer.elapsed();
            const auto xfer = timer.elapsed();
            telemetry.addXferStats(xfer, true, 4096, post);
            sink = sink + startTime.time_since_epoch().count();
        }
        const auto end = std::chrono::steady_clock::now();
        (void)sink;
        const double per =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<double>(kIters);
        best = std::min(best, per);
    }
    report("per_transfer_tax_nixl_duration", best);

    env_.popVar();
    env_.popVar();
}

// addXferStats append regime: buffer sized to hold every batch and the flush
// disabled, so all four events are appended each call (no drop). Isolates the
// mutex + 4x emplace_back cost.
TEST_F(telemetryBenchmark, DISABLED_AddXferStatsAppend) {
    // One event per emplace; 4 per call. Size the buffer to hold them all so no
    // call ever drops, and push the flush interval out of the way.
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(kIters * 4 + 16));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "3600000");

    // A fresh instance per pass (the buffer only holds one pass of appends).
    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < kRepeats; ++r) {
        nixlTelemetry telemetry("bench_append", "NOP");
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIters; ++i) {
            telemetry.addXferStats(
                std::chrono::microseconds(100), true, 4096, std::chrono::microseconds(10));
        }
        const auto end = std::chrono::steady_clock::now();
        const double per =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<double>(kIters);
        best = std::min(best, per);
    }
    report("add_xfer_stats_append", best);

    env_.popVar();
    env_.popVar();
}

// addXferStats drop regime: tiny buffer, pre-filled to capacity, flush disabled,
// so every measured call rejects its whole batch. This is the regime that
// dominates at datapath rates (the flush cannot keep up), and the number that
// most directly matches the reported small-message regression.
TEST_F(telemetryBenchmark, DISABLED_AddXferStatsDrop) {
    constexpr size_t buf_cap = 64;
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(buf_cap));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "3600000");

    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < kRepeats; ++r) {
        nixlTelemetry telemetry("bench_drop", "NOP");
        // Fill to capacity so every addXferStats below drops its 4-event batch.
        for (size_t i = 0; i < buf_cap; ++i) {
            telemetry.updateTxBytes(i);
        }
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIters; ++i) {
            telemetry.addXferStats(
                std::chrono::microseconds(100), true, 4096, std::chrono::microseconds(10));
        }
        const auto end = std::chrono::steady_clock::now();
        const double per =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<double>(kIters);
        best = std::min(best, per);
    }
    report("add_xfer_stats_drop", best);

    env_.popVar();
    env_.popVar();
}

// addXferStats with the four per-transfer metrics deactivated via
// NIXL_TELEMETRY_METRICS: every call is rejected by the producer gate before the
// lock (batch_size == 0). Isolates the cost of source-side activation checks --
// the "metric is off" fast path. Buffer sized to never drop so the only thing
// measured is the gate.
TEST_F(telemetryBenchmark, DISABLED_AddXferStatsDeactivated) {
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(kIters * 4 + 16));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "3600000");
    // Activate only an unrelated metric, so the four xfer events are all off.
    env_.addVar(TELEMETRY_METRICS_VAR, "agent_memory_registered");

    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < kRepeats; ++r) {
        nixlTelemetry telemetry("bench_deactivated", "NOP");
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIters; ++i) {
            telemetry.addXferStats(
                std::chrono::microseconds(100), true, 4096, std::chrono::microseconds(10));
        }
        const auto end = std::chrono::steady_clock::now();
        const double per =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<double>(kIters);
        best = std::min(best, per);
    }
    report("add_xfer_stats_deactivated", best);

    env_.popVar();
    env_.popVar();
    env_.popVar();
}

// Full per-transfer datapath tax proxy: what nixlAgent runs per completed async
// transfer with telemetry ON -- three steady_clock::now() reads, two
// duration_casts, and one addXferStats. Measured in the drop regime (the
// realistic saturated case).
TEST_F(telemetryBenchmark, DISABLED_PerTransferTaxSaturated) {
    constexpr size_t buf_cap = 64;
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(buf_cap));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "3600000");

    double best = std::numeric_limits<double>::max();
    for (size_t r = 0; r < kRepeats; ++r) {
        nixlTelemetry telemetry("bench_tax", "NOP");
        for (size_t i = 0; i < buf_cap; ++i) {
            telemetry.updateTxBytes(i);
        }
        volatile int64_t sink = 0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIters; ++i) {
            // Reproduce the agent's three clock reads per transfer -- their
            // count is what we are measuring. In the real datapath they bracket
            // real work (t0 before the backend submit, t1 after it returns, t2
            // on completion); here there is no transfer between them, so t0/t1
            // are back-to-back. Both are kept so the read count is not undercounted.
            const auto t0 = std::chrono::steady_clock::now(); // postXferReq: startTime
            const auto t1 = std::chrono::steady_clock::now(); // updateRequestStats(POST)
            const auto post = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
            const auto t2 = std::chrono::steady_clock::now(); // updateRequestStats(FINISH)
            const auto xfer = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0);
            telemetry.addXferStats(xfer, true, 4096, post);
            sink = sink + t2.time_since_epoch().count();
        }
        const auto end = std::chrono::steady_clock::now();
        (void)sink;
        const double per =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<double>(kIters);
        best = std::min(best, per);
    }
    report("per_transfer_tax_saturated", best);

    env_.popVar();
    env_.popVar();
}
