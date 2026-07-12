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

using EventCounts = std::array<uint64_t, nixl_telemetry_event_type_count>;
// Per event type, a histogram of received event values -> occurrence count. Used
// to prove value parity: the multiset of values drained from the ring must equal
// the multiset produced, so no value is corrupted, duplicated, or lost.
using ValueHistograms =
    std::array<std::unordered_map<uint64_t, uint64_t>, nixl_telemetry_event_type_count>;

struct RingTally {
    EventCounts counts{}; // occurrences per event type in the drained ring
    ValueHistograms valueHistogram{}; // received value -> count, per non-drop event type
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

    using et = nixl_telemetry_event_type_t;

    // Expected value multiset per event type, mirroring the producer workloads.
    ValueHistograms expected_values{};
    const auto add_range = [&](et type, uint64_t base) {
        for (uint64_t i = 0; i < iters; ++i) {
            ++expected_values[eventIndex(type)][base + i];
        }
    };
    const auto add_const = [&](et type, uint64_t value, uint64_t n) {
        expected_values[eventIndex(type)][value] += n;
    };

    add_range(et::AGENT_TX_BYTES, kTxBytesBase);
    add_range(et::AGENT_RX_BYTES, kRxBytesBase);
    add_range(et::AGENT_TX_REQUESTS_NUM, kTxReqBase);
    add_range(et::AGENT_RX_REQUESTS_NUM, kRxReqBase);
    add_range(et::AGENT_MEMORY_REGISTERED, kMemRegBase);
    add_range(et::AGENT_MEMORY_DEREGISTERED, kMemDeregBase);
    for (const auto status : error_statuses) {
        add_const(nixlTelemetryEventTypeForStatus(status), 1, iters / error_statuses.size());
    }
    // addXferStats submits one all-or-none batch; requests carry the fixed value 1.
    add_range(et::AGENT_XFER_POST_TIME, kWPostBase);
    add_range(et::AGENT_XFER_TIME, kWXferBase);
    add_range(et::AGENT_TX_BYTES, kWBytesBase);
    add_const(et::AGENT_TX_REQUESTS_NUM, 1, iters);
    add_range(et::AGENT_XFER_POST_TIME, kRPostBase);
    add_range(et::AGENT_XFER_TIME, kRXferBase);
    add_range(et::AGENT_RX_BYTES, kRBytesBase);
    add_const(et::AGENT_RX_REQUESTS_NUM, 1, iters);

    // Derive per-type counts and the grand total from the value multiset -- one
    // source of truth for both the count and the value-parity assertions.
    EventCounts expected{};
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
// producers (kept to non-transfer event types so the transfer counts are
// attributable only to the batch path). With the default allowlist every call
// stages all four events all-or-none; sized above capacity there are no drops,
// so each direction's four series must equal that direction's call count exactly
// -- no batch may be torn.
TEST_F(telemetryStressTest, PipelineBatchAtomicity) {
    const uint64_t iters = kSanitizerBuild ? 128 : 512;
    const std::string file = "batch_atomicity";

    const uint64_t write_calls = 2 * iters;
    const uint64_t read_calls = 2 * iters;
    const uint64_t mem_events = iters;
    const uint64_t err_events = iters;

    using et = nixl_telemetry_event_type_t;
    EventCounts expected{};
    expected[eventIndex(et::AGENT_XFER_POST_TIME)] = write_calls + read_calls;
    expected[eventIndex(et::AGENT_XFER_TIME)] = write_calls + read_calls;
    expected[eventIndex(et::AGENT_TX_BYTES)] = write_calls;
    expected[eventIndex(et::AGENT_TX_REQUESTS_NUM)] = write_calls;
    expected[eventIndex(et::AGENT_RX_BYTES)] = read_calls;
    expected[eventIndex(et::AGENT_RX_REQUESTS_NUM)] = read_calls;
    expected[eventIndex(et::AGENT_MEMORY_REGISTERED)] = mem_events;
    expected[eventIndex(et::AGENT_ERR_BACKEND)] = err_events;

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
    for (int w = 0; w < 2; ++w) {
        producers.emplace_back([&] {
            gate.wait();
            for (uint64_t i = 0; i < iters; ++i) {
                telemetry.addXferStats(
                    std::chrono::microseconds(100), true, 3000, std::chrono::microseconds(10));
            }
        });
    }
    for (int r = 0; r < 2; ++r) {
        producers.emplace_back([&] {
            gate.wait();
            for (uint64_t i = 0; i < iters; ++i) {
                telemetry.addXferStats(
                    std::chrono::microseconds(70), false, 1500, std::chrono::microseconds(7));
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

    const RingTally tally = drainRing(*reader);
    EXPECT_TRUE(tally.allInRange);
    EXPECT_EQ(tally.dropEventsSeen, 0u) << "no staging drop under configured capacity";
    EXPECT_EQ(tally.droppedTotal, 0u);

    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_XFER_POST_TIME)], write_calls + read_calls)
        << "post-time count must equal successful addXferStats call count";
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_XFER_TIME)], write_calls + read_calls)
        << "transfer-time count must equal successful addXferStats call count";
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_TX_BYTES)], write_calls);
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_TX_REQUESTS_NUM)], write_calls);
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_RX_BYTES)], read_calls);
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_RX_REQUESTS_NUM)], read_calls);
    // Atomicity: with no drops the four members of each direction move in exact
    // lockstep, so no accepted batch was published only partially.
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_TX_BYTES)],
              tally.counts[eventIndex(et::AGENT_TX_REQUESTS_NUM)]);
    EXPECT_EQ(tally.counts[eventIndex(et::AGENT_RX_BYTES)],
              tally.counts[eventIndex(et::AGENT_RX_REQUESTS_NUM)]);
    EXPECT_EQ(tally.nonDropEvents, expected_total);
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

        const auto start = std::chrono::steady_clock::now();
        telemetry.reset(); // destructor races an active periodic flush
        const auto elapsed = std::chrono::steady_clock::now() - start;
        EXPECT_LT(elapsed, shutdown_budget)
            << "telemetry destruction must complete promptly, not deadlock on the flush timer";
    }
    SUCCEED() << "shutdown with a possibly in-flight periodic flush stayed bounded and crash-free";
}
