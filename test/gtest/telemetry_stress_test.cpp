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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "common/cyclic_buffer.h"
#include "nixl_types.h"
#include "telemetry.h"
#include "telemetry_event.h"

// Full-pipeline telemetry stress coverage. These tests exercise the complete
// observable telemetry path -- concurrent producers -> metric mapping and
// activation -> nixlTelemetryStagingQueue -> periodic flush -> exporter ->
// observable output -- rather than the queue mechanics the staging-queue unit
// tests already cover directly. They must pass unchanged on both the mutex-backed
// staging queue and a future low-lock implementation.

namespace fs = std::filesystem;

namespace {

// Sanitizer builds run the same correctness checks with fewer repetitions: it
// keeps TSan/ASan within a CI-friendly runtime and limits the repeated
// thread-pool create/destroy that stresses the sanitizer thread registry, while
// still surfacing intermittent timer/queue/exporter defects. Detection is
// compile-time and test-only.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool kSanitizerBuild = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
constexpr bool kSanitizerBuild = true;
#else
constexpr bool kSanitizerBuild = false;
#endif
#else
constexpr bool kSanitizerBuild = false;
#endif

constexpr std::size_t
eventIndex(nixl_telemetry_event_type_t type) noexcept {
    return static_cast<std::size_t>(type);
}

using event_counts_t = std::array<uint64_t, nixl_telemetry_event_type_count>;
// Per event type, a histogram of received event values -> occurrence count. Used
// to prove value parity: the multiset of values drained from the ring must equal
// the multiset produced, so no value is corrupted, duplicated, or lost.
using value_histograms_t =
    std::array<std::unordered_map<uint64_t, uint64_t>, nixl_telemetry_event_type_count>;

struct RingTally {
    event_counts_t counts{}; // occurrences per event type in the drained ring
    value_histograms_t valueHistogram{}; // received value -> count, per non-drop event type
    uint64_t nonDropEvents = 0; // total events excluding the synthetic drop event
    uint64_t droppedTotal = 0; // summed value of synthetic drop events
    uint64_t dropEventsSeen = 0; // number of synthetic drop events observed
    bool allInRange = true; // every drained enum was a defined event type
};

// Smallest power of two strictly greater than n, so a BUFFER ring of that size
// (usable capacity size-1) can hold every one of n exported events without the
// downstream cyclic ring ever filling (the drop counter tracks only staging-queue
// drops, so ring loss must be structurally impossible in the under-capacity tests).
uint64_t
ringSizeAbove(uint64_t n) {
    uint64_t size = 1;
    while (size <= n) {
        size <<= 1;
    }
    return size;
}

// A start latch so every producer thread is released together, maximizing the
// concurrency the staging queue actually sees (per the plan: coordinate with
// atomics/barriers, do not rely on sleeps).
class StartGate {
public:
    void
    wait() {
        while (!go_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void
    release() {
        go_.store(true, std::memory_order_release);
    }

private:
    std::atomic<bool> go_{false};
};

// Non-destructively poll the writer's published event count (the reader never
// pops during polling, so read_pos stays 0 and size() == events pushed) until it
// reaches expected or the deadline passes. Returns the last observed size.
uint64_t
waitForRingSize(sharedRingBuffer<nixlTelemetryEvent> &reader,
                uint64_t expected,
                std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    uint64_t observed = reader.size();
    while (observed < expected && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        observed = reader.size();
    }
    return observed;
}

RingTally
drainRing(sharedRingBuffer<nixlTelemetryEvent> &reader) {
    RingTally tally;
    nixlTelemetryEvent event;
    while (reader.pop(event)) {
        const auto idx = eventIndex(event.eventType_);
        if (idx >= nixl_telemetry_event_type_count) {
            tally.allInRange = false;
            continue;
        }
        ++tally.counts[idx];
        if (event.eventType_ == nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED) {
            tally.droppedTotal += event.value_;
            ++tally.dropEventsSeen;
        } else {
            ++tally.valueHistogram[idx][event.value_];
            ++tally.nonDropEvents;
        }
    }
    return tally;
}

} // namespace

class telemetryStressTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        testDir_ = fs::path("/tmp") /
            ("telemetry_stress_" + std::to_string(::getpid()) + "_" +
             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(testDir_);

        env_.addVar("NIXL_TELEMETRY_ENABLE", "y");
        env_.addVar("NIXL_TELEMETRY_DIR", testDir_.string());
    }

    void
    TearDown() override {
        env_.popVar();
        env_.popVar();
        std::error_code ec;
        fs::remove_all(testDir_, ec);
    }

    std::string
    ringPath(const std::string &file) const {
        return (testDir_ / file).string();
    }

    fs::path testDir_;
    gtest::ScopedEnv env_;
};

// Mixed concurrent full-pipeline value parity under capacity.
//
// Several producers drive every metric-producing path concurrently into one
// BUFFER instance whose ring is sized above the complete event total, so nothing
// can be staging-dropped or ring-dropped. Each producer emits a distinct,
// disjoint value stream, so after the queue drains the multiset of values
// received per event type must equal the multiset produced -- proving every
// value sent was received exactly once, intact (no corruption, duplication, or
// loss), on top of exact per-type counts and no synthetic drop event.
TEST_F(telemetryStressTest, MixedConcurrentValueParityUnderCapacity) {
    const uint64_t iters = kSanitizerBuild ? 128 : 512; // divisible by the error set size (4)
    const std::string file = "mixed_value_parity";

    constexpr std::array<nixl_status_t, 4> error_statuses{
        NIXL_ERR_BACKEND, NIXL_ERR_INVALID_PARAM, NIXL_ERR_NOT_FOUND, NIXL_ERR_MISMATCH};

    // Disjoint per-producer value bases (spaced far beyond iters) so every value
    // is uniquely traceable and the expected multiset is unambiguous.
    constexpr uint64_t kTxBytesBase = 1'000'000;
    constexpr uint64_t kRxBytesBase = 2'000'000;
    constexpr uint64_t kTxReqBase = 3'000'000;
    constexpr uint64_t kRxReqBase = 4'000'000;
    constexpr uint64_t kMemRegBase = 5'000'000;
    constexpr uint64_t kMemDeregBase = 6'000'000;
    constexpr uint64_t kWPostBase = 7'000'000;
    constexpr uint64_t kWXferBase = 8'000'000;
    constexpr uint64_t kWBytesBase = 9'000'000;
    constexpr uint64_t kRPostBase = 10'000'000;
    constexpr uint64_t kRXferBase = 11'000'000;
    constexpr uint64_t kRBytesBase = 12'000'000;

    using event_type_t = nixl_telemetry_event_type_t;

    // Expected value multiset per event type, mirroring the producer workloads.
    value_histograms_t expected_values{};
    const auto add_range = [&](event_type_t type, uint64_t base) {
        for (uint64_t i = 0; i < iters; ++i) {
            ++expected_values[eventIndex(type)][base + i];
        }
    };
    const auto add_const = [&](event_type_t type, uint64_t value, uint64_t n) {
        expected_values[eventIndex(type)][value] += n;
    };

    add_range(event_type_t::AGENT_TX_BYTES, kTxBytesBase);
    add_range(event_type_t::AGENT_RX_BYTES, kRxBytesBase);
    add_range(event_type_t::AGENT_TX_REQUESTS_NUM, kTxReqBase);
    add_range(event_type_t::AGENT_RX_REQUESTS_NUM, kRxReqBase);
    add_range(event_type_t::AGENT_MEMORY_REGISTERED, kMemRegBase);
    add_range(event_type_t::AGENT_MEMORY_DEREGISTERED, kMemDeregBase);
    for (const auto status : error_statuses) {
        add_const(nixlTelemetryEventTypeForStatus(status), 1, iters / error_statuses.size());
    }
    // addXferStats submits one all-or-none batch; requests carry the fixed value 1.
    add_range(event_type_t::AGENT_XFER_POST_TIME, kWPostBase);
    add_range(event_type_t::AGENT_XFER_TIME, kWXferBase);
    add_range(event_type_t::AGENT_TX_BYTES, kWBytesBase);
    add_const(event_type_t::AGENT_TX_REQUESTS_NUM, 1, iters);
    add_range(event_type_t::AGENT_XFER_POST_TIME, kRPostBase);
    add_range(event_type_t::AGENT_XFER_TIME, kRXferBase);
    add_range(event_type_t::AGENT_RX_BYTES, kRBytesBase);
    add_const(event_type_t::AGENT_RX_REQUESTS_NUM, 1, iters);

    // Derive per-type counts and the grand total from the value multiset -- one
    // source of truth for both the count and the value-parity assertions.
    event_counts_t expected{};
    uint64_t expected_total = 0;
    for (std::size_t t = 0; t < nixl_telemetry_event_type_count; ++t) {
        for (const auto &[value, n] : expected_values[t]) {
            (void)value;
            expected[t] += n;
        }
        expected_total += expected[t];
    }

    const uint64_t ring_size = ringSizeAbove(expected_total);
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(ring_size));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "5");

    StartGate gate;
    nixlTelemetry telemetry(file, "BUFFER");

    std::vector<std::thread> producers;
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateTxBytes(kTxBytesBase + i);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateRxBytes(kRxBytesBase + i);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateTxRequestsNum(static_cast<uint32_t>(kTxReqBase + i));
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateRxRequestsNum(static_cast<uint32_t>(kRxReqBase + i));
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateMemoryRegistered(kMemRegBase + i);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateMemoryDeregistered(kMemDeregBase + i);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateErrorCount(error_statuses[i % error_statuses.size()]);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.addXferStats(std::chrono::microseconds(kWXferBase + i),
                                   true,
                                   kWBytesBase + i,
                                   std::chrono::microseconds(kWPostBase + i));
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.addXferStats(std::chrono::microseconds(kRXferBase + i),
                                   false,
                                   kRBytesBase + i,
                                   std::chrono::microseconds(kRPostBase + i));
        }
    });

    gate.release();
    for (auto &producer : producers) {
        producer.join();
    }

    auto reader = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        ringPath(file), false, TELEMETRY_VERSION);
    const auto timeout = std::chrono::seconds(kSanitizerBuild ? 60 : 20);
    const uint64_t observed = waitForRingSize(*reader, expected_total, timeout);
    ASSERT_EQ(observed, expected_total)
        << "all produced events must reach the BUFFER ring under capacity";

    const RingTally tally = drainRing(*reader);
    EXPECT_TRUE(tally.allInRange) << "every drained event must be a defined event type";
    EXPECT_EQ(tally.dropEventsSeen, 0u) << "no staging drop may occur under capacity";
    EXPECT_EQ(tally.droppedTotal, 0u);
    EXPECT_EQ(tally.nonDropEvents, expected_total);
    for (std::size_t i = 0; i < nixl_telemetry_event_type_count; ++i) {
        EXPECT_EQ(tally.counts[i], expected[i])
            << "event type " << i << " count mismatch under concurrent production";
        EXPECT_TRUE(tally.valueHistogram[i] == expected_values[i])
            << "event type " << i
            << " value multiset mismatch: a value sent was corrupted, duplicated, or lost";
    }
}

// Pipeline-level addXferStats() batch atomicity.
//
// Concurrent write and read addXferStats calls interleave with single-event
// producers. Each call carries a unique per-call id in its post-time, transfer-
// time, and bytes values, so the drained ring can be scanned to prove every
// call's four events are published as one indivisible, correctly ordered group
// ([post, transfer, bytes, requests]) with a consistent id -- no other event
// (another batch or a single-event producer) ever falls between them. Equal
// aggregate counts alone would not prove this (an implementation appending the
// four events independently would still hit the same totals under capacity);
// contiguity + identity is what actually rejects a torn or interleaved batch.
TEST_F(telemetryStressTest, PipelineBatchAtomicity) {
    const uint64_t iters = kSanitizerBuild ? 128 : 512;
    const std::string file = "batch_atomicity";

    constexpr int kWriteThreads = 2;
    constexpr int kReadThreads = 2;
    const uint64_t write_calls = kWriteThreads * iters;
    const uint64_t read_calls = kReadThreads * iters;
    const uint64_t mem_events = iters;
    const uint64_t err_events = iters;

    using event_type_t = nixl_telemetry_event_type_t;

    // Disjoint per-call id ranges so a batch's id also pins its direction, and
    // never collides with the single-event memory values (4096 + i).
    constexpr uint64_t kWriteIdBase = 1'000'000'000;
    constexpr uint64_t kReadIdBase = 2'000'000'000;

    event_counts_t expected{};
    expected[eventIndex(event_type_t::AGENT_XFER_POST_TIME)] = write_calls + read_calls;
    expected[eventIndex(event_type_t::AGENT_XFER_TIME)] = write_calls + read_calls;
    expected[eventIndex(event_type_t::AGENT_TX_BYTES)] = write_calls;
    expected[eventIndex(event_type_t::AGENT_TX_REQUESTS_NUM)] = write_calls;
    expected[eventIndex(event_type_t::AGENT_RX_BYTES)] = read_calls;
    expected[eventIndex(event_type_t::AGENT_RX_REQUESTS_NUM)] = read_calls;
    expected[eventIndex(event_type_t::AGENT_MEMORY_REGISTERED)] = mem_events;
    expected[eventIndex(event_type_t::AGENT_ERR_BACKEND)] = err_events;

    uint64_t expected_total = 0;
    for (const auto c : expected) {
        expected_total += c;
    }

    const uint64_t ring_size = ringSizeAbove(expected_total);
    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(ring_size));
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "5");

    StartGate gate;
    nixlTelemetry telemetry(file, "BUFFER");

    std::vector<std::thread> producers;
    for (int w = 0; w < kWriteThreads; ++w) {
        producers.emplace_back([&, w] {
            gate.wait();
            for (uint64_t i = 0; i < iters; ++i) {
                const uint64_t id = kWriteIdBase + static_cast<uint64_t>(w) * iters + i;
                telemetry.addXferStats(
                    std::chrono::microseconds(id), true, id, std::chrono::microseconds(id));
            }
        });
    }
    for (int r = 0; r < kReadThreads; ++r) {
        producers.emplace_back([&, r] {
            gate.wait();
            for (uint64_t i = 0; i < iters; ++i) {
                const uint64_t id = kReadIdBase + static_cast<uint64_t>(r) * iters + i;
                telemetry.addXferStats(
                    std::chrono::microseconds(id), false, id, std::chrono::microseconds(id));
            }
        });
    }
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateMemoryRegistered(4096 + i);
        }
    });
    producers.emplace_back([&] {
        gate.wait();
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateErrorCount(NIXL_ERR_BACKEND);
        }
    });

    gate.release();
    for (auto &producer : producers) {
        producer.join();
    }

    auto reader = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        ringPath(file), false, TELEMETRY_VERSION);
    const auto timeout = std::chrono::seconds(kSanitizerBuild ? 60 : 20);
    const uint64_t observed = waitForRingSize(*reader, expected_total, timeout);
    ASSERT_EQ(observed, expected_total) << "all batched and single events must reach the ring";

    // Drain in insertion order (the BUFFER ring preserves it) and walk it: single
    // events stand alone; every transfer batch must be an intact ordered quad.
    std::vector<nixlTelemetryEvent> events;
    events.reserve(expected_total);
    for (nixlTelemetryEvent e; reader->pop(e);) {
        events.push_back(e);
    }
    ASSERT_EQ(events.size(), expected_total);

    event_counts_t counts{};
    // Range-guarded tally so corrupted ring content fails cleanly instead of
    // indexing counts[] out of bounds (mirrors drainRing()'s guard).
    const auto bump = [&counts](nixl_telemetry_event_type_t t) {
        const std::size_t idx = eventIndex(t);
        if (idx >= nixl_telemetry_event_type_count) {
            ADD_FAILURE() << "drained event enum out of range: " << idx;
            return;
        }
        ++counts[idx];
    };
    uint64_t observed_batches = 0;
    std::size_t i = 0;
    while (i < events.size()) {
        const auto type = events[i].eventType_;
        bump(type);
        if (type == event_type_t::AGENT_MEMORY_REGISTERED ||
            type == event_type_t::AGENT_ERR_BACKEND) {
            ++i; // a single-event producer's record stands on its own
            continue;
        }

        // Anything else must be the first member of an intact addXferStats quad.
        ASSERT_EQ(type, event_type_t::AGENT_XFER_POST_TIME)
            << "torn batch: a transfer event appeared outside an ordered batch at index " << i;
        ASSERT_LE(i + 4, events.size()) << "torn batch: truncated quad at end of ring";

        const nixlTelemetryEvent post = events[i];
        const nixlTelemetryEvent xfer = events[i + 1];
        const nixlTelemetryEvent bytes = events[i + 2];
        const nixlTelemetryEvent reqs = events[i + 3];

        EXPECT_EQ(xfer.eventType_, event_type_t::AGENT_XFER_TIME)
            << "batch member 2 must be xfer time";
        const bool is_tx = bytes.eventType_ == event_type_t::AGENT_TX_BYTES;
        const bool is_rx = bytes.eventType_ == event_type_t::AGENT_RX_BYTES;
        EXPECT_TRUE(is_tx || is_rx) << "batch member 3 must be a bytes event";
        EXPECT_EQ(reqs.eventType_,
                  is_tx ? event_type_t::AGENT_TX_REQUESTS_NUM : event_type_t::AGENT_RX_REQUESTS_NUM)
            << "batch member 4 must be the matching requests event";

        // Identity: the three id-carrying members share one call's id, that id is
        // in the direction's range, and the request count is the fixed 1.
        EXPECT_EQ(post.value_, xfer.value_);
        EXPECT_EQ(post.value_, bytes.value_);
        EXPECT_EQ(reqs.value_, 1u);
        if (is_tx) {
            EXPECT_GE(post.value_, kWriteIdBase);
            EXPECT_LT(post.value_, kReadIdBase);
        } else {
            EXPECT_GE(post.value_, kReadIdBase);
        }

        bump(xfer.eventType_);
        bump(bytes.eventType_);
        bump(reqs.eventType_);
        ++observed_batches;
        i += 4;
    }

    EXPECT_EQ(observed_batches, write_calls + read_calls) << "every accepted call must appear once";
    for (std::size_t t = 0; t < nixl_telemetry_event_type_count; ++t) {
        EXPECT_EQ(counts[t], expected[t]) << "event type " << t << " count mismatch";
    }
    EXPECT_EQ(counts[eventIndex(event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED)], 0u)
        << "no staging drop under configured capacity";
}

// Repeated queue/periodic-task lifecycle shake-out.
//
// Many short iterations each build a fresh telemetry instance with a unique
// output identity, run producers, observe at least one periodic drain, then
// destroy normally. No completeness assertion is made (there is no final-flush
// contract); the value is letting sanitizers exercise timer/queue/exporter
// construction and teardown repeatedly.
TEST_F(telemetryStressTest, RepeatedLifecycleShakeOut) {
    const int iterations = kSanitizerBuild ? 8 : 30;
    const uint64_t iters = kSanitizerBuild ? 64 : 256;

    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "1024");
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "2");

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string file = "lifecycle_" + std::to_string(iteration);
        StartGate gate;
        {
            nixlTelemetry telemetry(file, "BUFFER");

            std::vector<std::thread> producers;
            for (int t = 0; t < 3; ++t) {
                producers.emplace_back([&, t] {
                    gate.wait();
                    for (uint64_t i = 0; i < iters; ++i) {
                        if (t == 0) {
                            telemetry.updateTxBytes(1024 + i);
                        } else if (t == 1) {
                            telemetry.updateRxBytes(2048 + i);
                        } else {
                            telemetry.addXferStats(std::chrono::microseconds(30),
                                                   true,
                                                   512,
                                                   std::chrono::microseconds(3));
                        }
                    }
                });
            }

            gate.release();
            for (auto &producer : producers) {
                producer.join();
            }

            // Observe at least one periodic drain by waiting until the ring has
            // received some events (bounded), so destruction races a real,
            // recently-active flush cycle rather than an idle one.
            auto reader = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
                ringPath(file), false, TELEMETRY_VERSION);
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(kSanitizerBuild ? 30 : 10);
            while (reader->size() == 0 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            EXPECT_GT(reader->size(), 0u) << "at least one periodic drain must be observed";
        }
    }
    SUCCEED() << "repeated construction/drain/destruction completed without hang or crash";
}

// Safe shutdown with an in-flight consumer.
//
// Producers are fully joined before destruction (no concurrent API calls during
// the destructor -- that would violate object lifetime). With a very short flush
// interval and a still-populated staging queue, the periodic consumer is likely
// mid-flush when the destructor runs; teardown must stop and join the pool
// cleanly and return within a bounded time, with no crash, deadlock, or
// sanitizer report. No export-completeness is asserted after destruction.
//
// The destructor runs on a separate thread guarded by an external timeout, so a
// hang (e.g. a teardown deadlock on the flush timer) is reported as a hard
// failure rather than blocking the whole test binary until the CI/meson timeout.
// (Deterministically forcing overlap with an in-flight flush would need a
// flush-entry hook in production code, which this test-only change avoids; the
// short interval plus a large pending backlog makes overlap highly likely, and
// sanitizer builds catch any teardown race.)
TEST_F(telemetryStressTest, SafeShutdownWithInFlightConsumer) {
    const int iterations = kSanitizerBuild ? 4 : 12;
    const uint64_t iters = kSanitizerBuild ? 256 : 2000;

    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "1024");
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    const auto shutdown_budget = std::chrono::seconds(kSanitizerBuild ? 30 : 10);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string file = "shutdown_" + std::to_string(iteration);
        StartGate gate;

        auto telemetry = std::make_unique<nixlTelemetry>(file, "BUFFER");

        std::vector<std::thread> producers;
        for (int t = 0; t < 4; ++t) {
            producers.emplace_back([&] {
                gate.wait();
                for (uint64_t i = 0; i < iters; ++i) {
                    telemetry->addXferStats(
                        std::chrono::microseconds(20), true, 256, std::chrono::microseconds(2));
                }
            });
        }

        gate.release();
        for (auto &producer : producers) {
            producer.join();
        }

        // Destroy on a worker thread bounded by an external timeout. Ownership is
        // moved into the task, so the object lives inside the worker's closure --
        // not on this stack. On a hang we detach and return without touching the
        // moved-from local, so the abandoned deleter thread references only its
        // own closure (no dangling stack reference or cross-thread race).
        std::packaged_task<void()> shutdown_task(
            [tel = std::move(telemetry)]() mutable { tel.reset(); });
        std::future<void> shutdown_done = shutdown_task.get_future();
        std::thread runner(std::move(shutdown_task));

        if (shutdown_done.wait_for(shutdown_budget) == std::future_status::ready) {
            runner.join();
            shutdown_done.get(); // surface any exception thrown during teardown
        } else {
            runner.detach();
            ADD_FAILURE() << "telemetry destruction did not complete within "
                          << shutdown_budget.count()
                          << "s -- likely a teardown deadlock on the flush timer";
            return;
        }
    }
    SUCCEED() << "shutdown with a possibly in-flight periodic flush stayed bounded and crash-free";
}
