/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/memos_pipeline.h"

#include <iostream>
#include <stdexcept>

namespace {

void
check(bool condition) {
    if (!condition) {
        throw std::runtime_error("MEMOS pipeline assertion failed");
    }
}

struct MockTransfers {
    std::vector<int> remaining;
    std::vector<int> visits;
    size_t in_flight = 0;
    size_t peak = 0;
    int issued = 0;
    int completed = 0;
    int fail_post = -1;
    int fail_poll = -1;
    int reuse_depth = 0;
    bool refilled_before_first_completion = false;

    explicit MockTransfers(size_t banks) : remaining(banks), visits(banks) {}

    bool
    post(size_t bank) {
        check(bank < remaining.size() && remaining[bank] == 0);
        if (reuse_depth != 0) {
            // Different key banks can alias the same physical local buffers.
            for (size_t other = 0; other < remaining.size(); ++other) {
                check(remaining[other] == 0 || other % reuse_depth != bank % reuse_depth);
            }
        }
        if (issued == fail_post) {
            return false;
        }
        if (issued >= 2 && remaining[0] != 0 && visits[0] == 1) {
            refilled_before_first_completion = true;
        }
        // Bank zero stalls while other lanes complete and refill.
        remaining[bank] = bank == 0 ? 100 : 2;
        ++visits[bank];
        ++issued;
        ++in_flight;
        peak = std::max(peak, in_flight);
        return true;
    }

    nixlbench::MemosBatchStatus
    poll(size_t bank) {
        check(remaining[bank] > 0);
        if (completed == fail_poll) {
            return nixlbench::MemosBatchStatus::error;
        }
        if (--remaining[bank] != 0) {
            return nixlbench::MemosBatchStatus::pending;
        }
        --in_flight;
        ++completed;
        return nixlbench::MemosBatchStatus::complete;
    }

    bool
    run(int depth, int iterations, int cancel_after = -1, bool reuse_local = false) {
        reuse_depth = reuse_local ? depth : 0;
        return nixlbench::runMemosPipeline(
            remaining.size(),
            depth,
            iterations,
            [&](size_t bank) { return post(bank); },
            [&](size_t bank) { return poll(bank); },
            [&] { return cancel_after >= 0 && completed >= cancel_after; },
            reuse_local);
    }
};

} // namespace

int
main() {
    try {
        // Exact issue budget and final drain, including zero and fewer
        // iterations than lanes, one bank per lane, and many pool rotations.
        for (size_t banks : {size_t{1}, size_t{2}, size_t{32}, size_t{64}}) {
            for (int depth : {1, 2, 32}) {
                if (static_cast<size_t>(depth) > banks) {
                    continue;
                }
                for (int iterations : {0, 1, 3, 31, 32, 33, 257}) {
                    MockTransfers transfers(banks);
                    check(transfers.run(depth, iterations));
                    check(transfers.issued == iterations);
                    check(transfers.completed == iterations);
                    check(transfers.in_flight == 0);
                    check(transfers.peak == static_cast<size_t>(std::min(depth, iterations)));
                    if (iterations >= static_cast<int>(banks)) {
                        for (int visits : transfers.visits) {
                            check(visits > 0);
                        }
                    }
                }
            }
        }
        // Reused-buffer lanes must never post two banks sharing buffers.
        // A slow bank zero also exercises out-of-order completions and fair
        // traversal during one whole-pool prepopulation/verification pass.
        for (size_t banks : {size_t{4}, size_t{32}, size_t{256}}) {
            for (int depth : {1, 2, 4}) {
                for (int iterations : {0, 1, 3, 4, 5, 33, 259, static_cast<int>(banks)}) {
                    MockTransfers reused(banks);
                    check(reused.run(depth, iterations, -1, true));
                    check(reused.issued == iterations && reused.completed == iterations);
                    check(reused.in_flight == 0);
                    check(reused.peak == static_cast<size_t>(std::min(depth, iterations)));
                    if (iterations == static_cast<int>(banks)) {
                        for (int visits : reused.visits) {
                            check(visits == 1);
                        }
                    }
                }
            }
        }
        MockTransfers nondivisible(6);
        check(!nondivisible.run(4, 20, -1, true));
        check(nondivisible.issued == 0);
        for (int fail_at : {0, 1, 5}) {
            MockTransfers post_error(8);
            post_error.fail_post = fail_at;
            check(!post_error.run(2, 20, -1, true));
            check(post_error.issued == fail_at);
            MockTransfers poll_error(8);
            poll_error.fail_poll = fail_at;
            check(!poll_error.run(2, 20, -1, true));
            check(poll_error.completed == fail_at);
            MockTransfers cancelled(8);
            check(!cancelled.run(2, 20, fail_at, true));
            check(cancelled.completed == fail_at);
        }
        MockTransfers independent(8);
        check(independent.run(2, 80));
        check(independent.refilled_before_first_completion);

        for (int fail_at : {0, 1, 5}) {
            MockTransfers post_error(8);
            post_error.fail_post = fail_at;
            check(!post_error.run(2, 20));
            check(post_error.issued == fail_at);
            MockTransfers poll_error(8);
            poll_error.fail_poll = fail_at;
            check(!poll_error.run(2, 20));
            check(poll_error.completed == fail_at);
            MockTransfers cancelled(8);
            check(!cancelled.run(2, 20, fail_at));
            check(cancelled.completed == fail_at);
        }
        for (int depth : {-1, 0, 9}) {
            MockTransfers invalid(8);
            check(!invalid.run(depth, 20));
            check(invalid.issued == 0);
        }
        MockTransfers no_banks(0);
        check(!no_banks.run(1, 20));
        MockTransfers negative_iterations(8);
        check(!negative_iterations.run(2, -1));
        std::cout << "MEMOS pipeline tests passed\n";
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
