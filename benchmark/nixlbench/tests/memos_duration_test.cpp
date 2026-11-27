/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/memos_duration.h"
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
void
checkAt(bool value, int line) {
    if (!value) {
        throw std::runtime_error("MEMOS duration assertion failed at line " + std::to_string(line));
    }
}

#define check(value) checkAt((value), __LINE__)

template<typename Function>
void
reject(Function fn) {
    try {
        fn();
    }
    catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid duration configuration accepted");
}

struct Simulation {
    uint64_t clock = 0;
    int depth;
    bool reuse;
    std::vector<int> remaining;
    std::vector<std::pair<uint64_t, uint64_t>> events;
    std::vector<uint64_t> posting;
    std::vector<int> visits;
    int issued = 0, completed = 0, latency_samples = 0;
    int fail_post = -1, fail_poll = -1, cancel_after = -1;
    nixlbench::MemosDurationStats stats;

    Simulation(size_t banks, int d, bool r)
        : depth(d),
          reuse(r),
          remaining(banks),
          posting(banks),
          visits(banks) {}

    bool
    run(uint64_t warmup = 40,
        uint64_t duration = 300,
        nixlbench::MemosDurationWorkspace *workspace = nullptr) {
        return nixlbench::runMemosDurationPipeline(
            remaining.size(),
            depth,
            reuse,
            0,
            warmup,
            duration,
            [&](size_t bank) {
                check(remaining[bank] == 0 && clock - 1 < warmup + duration);
                if (reuse) {
                    for (size_t other = 0; other < remaining.size(); ++other) {
                        check(remaining[other] == 0 || other % depth != bank % depth);
                    }
                }
                if (issued == fail_post) {
                    return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::error, 0};
                }
                posting[bank] = clock - 1; // Clock call immediately before post.
                // Slow ownership class zero; other lanes may make more progress.
                remaining[bank] = bank % depth == 0 ? 7 : 1;
                ++visits[bank];
                ++issued;
                return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::posted,
                                                  posting[bank]};
            },
            [&](size_t bank) {
                check(remaining[bank] > 0);
                if (completed == fail_poll) {
                    return nixlbench::MemosBatchStatus::error;
                }
                return --remaining[bank] == 0 ? nixlbench::MemosBatchStatus::complete :
                                                nixlbench::MemosBatchStatus::pending;
            },
            [&](size_t bank, bool latency) {
                // The clock increments after returning its observation timestamp.
                const uint64_t t = clock - 1;
                events.emplace_back(posting[bank], t);
                if (latency) {
                    check(posting[bank] >= warmup && t < warmup + duration);
                    ++latency_samples;
                }
                ++completed;
            },
            [&] { return clock++; },
            [&] { return cancel_after >= 0 && completed >= cancel_after; },
            stats,
            workspace);
    }
};
} // namespace

int
main() {
    try {
        check(nixlbench::completedBytes(7, 16, 1024 * 1024) == 7ULL * 16 * 1024 * 1024);
        try {
            nixlbench::completedBytes(std::numeric_limits<uint64_t>::max(), 32, 1024);
            throw std::runtime_error("byte count overflow accepted");
        }
        catch (const std::overflow_error &) {
        }
        nixlbench::validateDurationSeconds(0, 0);
        nixlbench::validateDurationSeconds(1, 30);
        nixlbench::validateDurationSeconds(0, 30);
        for (auto values :
             {std::pair<int, int>{-1, 30}, {1, -1}, {1, 0}, {86401, 30}, {1, 86401}}) {
            reject([&] { nixlbench::validateDurationSeconds(values.first, values.second); });
        }
        for (bool reuse : {false, true}) {
            for (int depth : {1, 2, 4}) {
                for (uint64_t warmup : {uint64_t{0}, uint64_t{40}}) {
                    Simulation s(16, depth, reuse);
                    check(s.run(warmup));
                    check(s.stats.window_start_us == warmup &&
                          s.stats.window_end_us == warmup + 300);
                    check(s.issued == s.completed);
                    check(s.stats.issued_batches == static_cast<uint64_t>(s.issued));
                    check(s.stats.completed_batches == static_cast<uint64_t>(s.completed));
                    check(s.stats.peak_batches == static_cast<uint64_t>(depth));
                    uint64_t warm = 0, measured = 0, drain = 0;
                    for (const auto &e : s.events) {
                        if (e.second < warmup) {
                            ++warm;
                        } else if (e.second < warmup + 300) {
                            ++measured;
                        } else {
                            ++drain;
                        }
                    }
                    check(s.stats.warmup_batches == warm && s.stats.measured_batches == measured &&
                          s.stats.drain_batches == drain);
                    check(warm + measured + drain == s.stats.completed_batches && measured > 0);
                    check(s.stats.latency_batches == static_cast<uint64_t>(s.latency_samples));
                    check(s.stats.workers == 1 && s.stats.workers_with_completions == 1);
                    check(s.stats.workers_with_warmup == (warmup == 0 ? 0 : 1));
                    for (int n : s.remaining) {
                        check(n == 0);
                    }
                    if (reuse && depth > 1) {
                        // Time mode has no equal iteration quota: fast lanes
                        // actually complete more batches than the slow lane.
                        size_t slow = 0, fast = 0;
                        for (size_t b = 0; b < s.visits.size(); ++b) {
                            (b % depth == 0 ? slow : fast) += s.visits[b];
                        }
                        check(fast > slow);
                    }
                }
            }
            for (int fail : {0, 1, 5}) {
                Simulation a(8, 2, reuse);
                a.fail_post = fail;
                check(!a.run());
                check(a.issued == fail);
                Simulation b(8, 2, reuse);
                b.fail_poll = fail;
                check(!b.run());
                check(b.completed == fail);
                Simulation c(8, 2, reuse);
                c.cancel_after = fail;
                check(!c.run());
                check(c.completed == fail);
            }
        }
        // Boundary timestamps: exactly warmup-end is measured; exactly
        // measurement-end is drained. Warmup requests may count as steady
        // completions, but cannot become latency samples.
        for (uint64_t t : {uint64_t{9}, uint64_t{10}, uint64_t{39}, uint64_t{40}}) {
            nixlbench::MemosDurationStats stats;
            int posts = 0;
            uint64_t next = 0;
            check(nixlbench::runMemosDurationPipeline(
                1,
                1,
                true,
                0,
                10,
                30,
                [&](size_t) {
                    ++posts;
                    return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::posted, 0};
                },
                [&](size_t) {
                    next = t;
                    return nixlbench::MemosBatchStatus::complete;
                },
                [&](size_t, bool latency) {
                    if (posts == 1) {
                        check(!latency);
                    }
                },
                [&] {
                    const auto v = next;
                    next = 40;
                    return v;
                },
                [] { return false; },
                stats));
            check(posts == 1);
            check(stats.warmup_batches == (t < 10 ? 1 : 0));
            check(stats.measured_batches == (t >= 10 && t < 40 ? 1 : 0));
            check(stats.drain_batches == (t >= 40 ? 1 : 0));
        }
        // Dispatch crosses the deadline after the scheduler checked it. The
        // second post is skipped, while the first accepted request still drains.
        {
            uint64_t clock = 0;
            int dispatches = 0, completions = 0;
            nixlbench::MemosDurationStats stats;
            check(nixlbench::runMemosDurationPipeline(
                2,
                2,
                true,
                0,
                0,
                10,
                [&](size_t) {
                    if (++dispatches == 2) {
                        clock = 10;
                        return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::skipped,
                                                          clock};
                    }
                    return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::posted, 0};
                },
                [](size_t) { return nixlbench::MemosBatchStatus::complete; },
                [&](size_t, bool) { ++completions; },
                [&] { return clock; },
                [] { return false; },
                stats));
            check(dispatches == 2 && completions == 1);
            check(stats.issued_batches == 1 && stats.completed_batches == 1);
            check(stats.drain_batches == 1 && stats.peak_batches == 1);
        }
        // Scheduler observation and actual dispatch can straddle warmup-end.
        // Throughput follows completion; latency eligibility follows dispatch.
        for (uint64_t dispatch : {uint64_t{9}, uint64_t{10}, uint64_t{11}}) {
            uint64_t clock = 9;
            bool sampled = false;
            nixlbench::MemosDurationStats stats;
            check(nixlbench::runMemosDurationPipeline(
                1,
                1,
                false,
                0,
                10,
                10,
                [&](size_t) {
                    clock = dispatch;
                    return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::posted, dispatch};
                },
                [&](size_t) {
                    clock = 12;
                    return nixlbench::MemosBatchStatus::complete;
                },
                [&](size_t, bool latency) {
                    sampled = latency;
                    clock = 20;
                },
                [&] { return clock; },
                [] { return false; },
                stats));
            check(stats.measured_batches == 1 && stats.completed_batches == 1);
            check(sampled == (dispatch >= 10));
            check(stats.latency_batches == (dispatch >= 10 ? 1 : 0));
        }
        nixlbench::MemosDurationStats aggregate;
        Simulation a(8, 2, true), b(8, 2, true);
        check(a.run() && b.run());
        aggregate.add(a.stats);
        aggregate.add(b.stats);
        check(aggregate.window_end_us - aggregate.window_start_us == 300);
        check(aggregate.measured_batches == a.stats.measured_batches + b.stats.measured_batches);
        check(aggregate.workers == 2 && aggregate.workers_with_completions == 2);
        check(nixlbench::validDurationStats(aggregate, 2));
        auto mismatch = b.stats;
        ++mismatch.window_start_us;
        auto rejected = a.stats;
        rejected.add(mismatch);
        check(!rejected.valid && !nixlbench::validDurationStats(rejected, 1));
        auto overflow = a.stats;
        overflow.issued_batches = std::numeric_limits<uint64_t>::max();
        overflow.add(b.stats);
        check(!overflow.valid);
        auto wrong_sum = aggregate;
        ++wrong_sum.completed_batches;
        check(!nixlbench::validDurationStats(wrong_sum, 2));
        auto wrong_workers = aggregate;
        check(!nixlbench::validDurationStats(wrong_workers, 3));
        for (int limit : {0, -1, 1000001}) {
            reject([&] { nixlbench::validateDurationSampleLimit(limit); });
        }
        nixlbench::validateDurationSampleLimit(1);
        nixlbench::validateDurationSampleLimit(65536);
        nixlbench::MemosDurationWorkspace prepared(16, 2);
        Simulation ready(16, 2, true);
        auto *active_storage = prepared.active.data();
        auto *timestamp_storage = prepared.posted_at.data();
        check(ready.run(40, 300, &prepared));
        check(prepared.active.data() == active_storage &&
              prepared.posted_at.data() == timestamp_storage);
        const int already_issued = ready.issued;
        check(!ready.run(40, 300, &prepared) && ready.issued == already_issued);
        nixlbench::MemosDurationWorkspace bad_shape(16, 4);
        Simulation invalid_workspace(16, 2, true);
        check(!invalid_workspace.run(40, 300, &bad_shape));
        for (int depth : {-1, 0, 9}) {
            Simulation s(8, depth, true);
            check(!s.run());
            check(s.issued == 0);
        }
        Simulation nondivisible(6, 4, true);
        check(!nondivisible.run());
        Simulation zero(8, 2, true);
        check(!zero.run(0, 0));
        int callbacks = 0;
        nixlbench::MemosDurationStats impossible;
        check(!nixlbench::runMemosDurationPipeline(
            1,
            1,
            true,
            std::numeric_limits<uint64_t>::max() - 5,
            10,
            30,
            [&](size_t) {
                ++callbacks;
                return nixlbench::MemosPostResult{nixlbench::MemosPostStatus::posted, 0};
            },
            [&](size_t) {
                ++callbacks;
                return nixlbench::MemosBatchStatus::complete;
            },
            [&](size_t, bool) { ++callbacks; },
            [] { return uint64_t{0}; },
            [] { return false; },
            impossible));
        check(callbacks == 0);

        std::cout << "MEMOS duration tests passed: boundaries, unequal lane rates, drain, errors, "
                     "cancellation, aggregation\n";
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
