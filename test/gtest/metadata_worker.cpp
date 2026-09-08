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
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "nixl_metadata_worker.h"

// The worker's contract under contention: a submitted task always runs, tasks
// never run two at a time, and stop() finishes even while other threads keep
// submitting. The backends depend on all three -- they hold no locks of their
// own and are torn down as soon as stop() returns.
namespace gtest::metadata_worker {

namespace {

    // Long enough to interleave, short enough to keep the suite quick.
    constexpr auto step = std::chrono::microseconds(50);

    class overlapDetector {
    public:
        void
        run() {
            if (inFlight_.fetch_add(1) != 0) {
                ++overlaps_;
            }
            std::this_thread::sleep_for(step);
            inFlight_.fetch_sub(1);
            ++ran_;
        }

        [[nodiscard]] int
        ran() const {
            return ran_.load();
        }

        [[nodiscard]] int
        overlaps() const {
            return overlaps_.load();
        }

    private:
        std::atomic<int> inFlight_{0};
        std::atomic<int> overlaps_{0};
        std::atomic<int> ran_{0};
    };

} // namespace

TEST(MetadataWorker, RunsOnTheCallerWhenNotStarted) {
    nixlMetadataWorker worker;
    std::thread::id ran_on;

    worker.submit([&ran_on] { ran_on = std::this_thread::get_id(); });

    EXPECT_EQ(ran_on, std::this_thread::get_id());
}

TEST(MetadataWorker, RunsOnTheWorkerOnceStarted) {
    nixlMetadataWorker worker;
    std::atomic<std::thread::id> ran_on{};

    worker.start([] {}, std::chrono::microseconds(100));
    worker.submit([&ran_on] { ran_on.store(std::this_thread::get_id()); });
    worker.stop();

    EXPECT_NE(ran_on.load(), std::thread::id{});
    EXPECT_NE(ran_on.load(), std::this_thread::get_id());
}

TEST(MetadataWorker, StopIsSafeFromSeveralThreads) {
    nixlMetadataWorker worker;
    worker.start([] {}, std::chrono::microseconds(100));

    std::vector<std::thread> stoppers;
    for (int i = 0; i < 4; ++i) {
        stoppers.emplace_back([&worker] { worker.stop(); });
    }
    for (auto &stopper : stoppers) {
        stopper.join();
    }

    // Every caller returned, and the worker takes work again afterwards.
    bool ran = false;
    worker.submit([&ran] { ran = true; });
    EXPECT_TRUE(ran);
}

TEST(MetadataWorker, ShutdownKeepsUpWithSubmitters) {
    overlapDetector work;
    std::atomic<int> submitted{0};
    std::atomic<bool> done{false};
    nixlMetadataWorker worker;

    std::vector<std::thread> submitters;
    for (int i = 0; i < 4; ++i) {
        submitters.emplace_back([&] {
            while (!done.load()) {
                worker.submit([&work] { work.run(); });
                ++submitted;
                std::this_thread::sleep_for(step);
            }
        });
    }

    for (int round = 0; round < 20; ++round) {
        worker.start([] {}, std::chrono::microseconds(100));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // Draining the queue to empty here would never finish while the
        // submitters run, so this is what catches a shutdown that waits for one.
        worker.stop();
    }
    done.store(true);
    for (auto &submitter : submitters) {
        submitter.join();
    }
    worker.stop();

    EXPECT_EQ(work.ran(), submitted.load()) << "a submitted task was dropped";
    EXPECT_EQ(work.overlaps(), 0) << "two tasks ran at once";
}

TEST(MetadataWorker, ShutdownRunsWhatThePollSubmitted) {
    overlapDetector work;
    std::atomic<int> submitted{0};
    nixlMetadataWorker worker;

    // A poll that queues work is how a backend services its store, and it is the
    // one submitter that cannot fall back to running the task itself.
    worker.start(
        [&] {
            worker.submit([&work] { work.run(); });
            ++submitted;
        },
        std::chrono::microseconds(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    worker.stop();

    EXPECT_GT(submitted.load(), 0);
    EXPECT_EQ(work.ran(), submitted.load());
    EXPECT_EQ(work.overlaps(), 0);
}

} // namespace gtest::metadata_worker
