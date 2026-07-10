/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <climits>
#include <atomic>

#include "telemetry.h"
#include "telemetry_event.h"
#include "nixl_types.h"
#include "common/cyclic_buffer.h"
#include "common.h"
#include "backend/backend_engine.h"
#include "mocks/gmock_engine.h"

namespace fs = std::filesystem;
constexpr char TELEMETRY_ENABLED_VAR[] = "NIXL_TELEMETRY_ENABLE";
constexpr char TELEMETRY_DIR_VAR[] = "NIXL_TELEMETRY_DIR";
constexpr char TELEMETRY_EXPORTER_VAR[] = "NIXL_TELEMETRY_EXPORTER";

class telemetryTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        testDir_ = "/tmp/telemetry_test_files";
        testFile_ = "test_telemetry";
        try {
            if (!fs::exists(testDir_)) {
                fs::create_directory(testDir_);
            }
        }
        catch (const fs::filesystem_error &e) {
            throw std::runtime_error("Could not create the directory for telemetry test.");
        }

        envHelper_.addVar(TELEMETRY_ENABLED_VAR, "y");
        envHelper_.addVar(TELEMETRY_DIR_VAR, testDir_.string());
    }

    void
    TearDown() override {
        envHelper_.popVar();
        envHelper_.popVar();
        if (fs::exists(testDir_)) {
            try {
                fs::remove_all(testDir_);
            }
            catch (const fs::filesystem_error &e) {
                // ignore can fail due to nsf
            }
        }
    }

    void
    validateState() {
        auto path = std::string(testDir_.string()) + "/" + testFile_;
        auto buffer =
            std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
        EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
        EXPECT_EQ(buffer->capacity(), capacity_);
        EXPECT_EQ(buffer->size(), size_);
        EXPECT_EQ(buffer->empty(), size_ == 0);
        EXPECT_EQ(buffer->full(), size_ == capacity_);
    }

    fs::path testDir_;
    std::string testFile_;
    gtest::ScopedEnv envHelper_;
    size_t capacity_ = 4096;
    size_t size_ = 0;
    size_t readPos_ = 0;
    size_t writePos_ = 0;
    size_t mask_ = 4096 - 1;
    // backend_map_t backendMap_;
};

TEST_F(telemetryTest, BasicInitialization) {
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(testFile_, "BUFFER");
        validateState();
    });
}

TEST_F(telemetryTest, InitializationWithEmptyFileName) {
    EXPECT_THROW({ nixlTelemetry telemetry("", "BUFFER"); }, std::invalid_argument);
}

TEST_F(telemetryTest, CustomBufferSize) {
    auto tmp_capacity = capacity_;
    capacity_ = 32;
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "32");

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(testFile_, "BUFFER");
        validateState();
    });
    capacity_ = tmp_capacity;
    envHelper_.popVar();
}

TEST_F(telemetryTest, InvalidBufferSize) {
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "0");

    EXPECT_THROW({ nixlTelemetry telemetry(testFile_, "BUFFER"); }, std::invalid_argument);
    envHelper_.popVar();
}

TEST_F(telemetryTest, NonexistentExporterThrows) {
    // An explicitly requested exporter that cannot be loaded must surface the
    // failure rather than silently disabling telemetry.

    // The plugin manager logs an expected warning when the .so is not found.
    gtest::LogIgnoreGuard ignore_missing_plugin("Plugin file does not exist");

    EXPECT_THROW(
        { nixlTelemetry telemetry(testFile_, "nixl_nonexistent_exporter"); }, std::runtime_error);
}

TEST_F(telemetryTest, CreateFallsBackToNopWithoutSink) {
    envHelper_.popVar(); // drop NIXL_TELEMETRY_DIR set by SetUp -> no sink
    // Neutralize any inherited NIXL_TELEMETRY_EXPORTER (empty is ignored) so the
    // sinkless NOP path is exercised deterministically.
    envHelper_.addVar(TELEMETRY_EXPORTER_VAR, "");

    auto telemetry = nixlTelemetry::create(testFile_);

    envHelper_.popVar(); // drop empty NIXL_TELEMETRY_EXPORTER
    envHelper_.addVar(TELEMETRY_DIR_VAR, testDir_.string()); // restore for TearDown symmetry

    ASSERT_NE(telemetry, nullptr)
        << "telemetry requested without a sink must collect in-process via NOP";
    EXPECT_NO_THROW(telemetry->updateTxBytes(1024));

    // Collect-only NOP fallback writes nothing to disk.
    EXPECT_FALSE(fs::exists(testDir_.string() + "/" + testFile_));
}

TEST_F(telemetryTest, NopExporterIsActiveAndWritesNothing) {
    // NIXL_TELEMETRY_EXPORTER=NOP keeps telemetry active (event collection and
    // in-band getXferTelemetry) without writing anywhere, so the overhead of the
    // collection path can be measured in isolation. It takes precedence over
    // NIXL_TELEMETRY_DIR and produces no output file.
    envHelper_.addVar(TELEMETRY_EXPORTER_VAR, "NOP");
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    auto telemetry = nixlTelemetry::create(testFile_);
    ASSERT_NE(telemetry, nullptr) << "NOP exporter must keep telemetry active";

    EXPECT_NO_THROW(telemetry->updateTxBytes(1024));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // The NOP exporter discards events, so no buffer file is created.
    EXPECT_FALSE(fs::exists(testDir_.string() + "/" + testFile_));

    envHelper_.popVar(); // TELEMETRY_RUN_INTERVAL_VAR
    envHelper_.popVar(); // TELEMETRY_EXPORTER_VAR
}

// Test transfer bytes tracking
TEST_F(telemetryTest, TransferBytesTracking) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    nixlTelemetry telemetry(testFile_, "BUFFER");

    EXPECT_NO_THROW(telemetry.updateTxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateRxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateTxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateRxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND));
    EXPECT_NO_THROW(telemetry.updateMemoryRegistered(1024));
    EXPECT_NO_THROW(telemetry.updateMemoryDeregistered(1024));
    EXPECT_NO_THROW(telemetry.addXferStats(
        std::chrono::microseconds(100), true, 2000, std::chrono::microseconds(10)));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 11);
    EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
    EXPECT_EQ(buffer->capacity(), capacity_);
    EXPECT_EQ(buffer->empty(), false);
    EXPECT_EQ(buffer->full(), false);
    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_RX_BYTES);
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM);
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM);
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_ERR_BACKEND);
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED);
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_MEMORY_DEREGISTERED);
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME);
    EXPECT_EQ(event.value_, 10);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_XFER_TIME);
    EXPECT_EQ(event.value_, 100);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);
    EXPECT_EQ(event.value_, 2000);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_REQUESTS_NUM);
    EXPECT_EQ(event.value_, 1);
    envHelper_.popVar();
}

// addXferStats RX branch (is_write == false): covers the AGENT_RX_BYTES /
// AGENT_RX_REQUESTS_NUM mapping that TransferBytesTracking (write path) does not.
TEST_F(telemetryTest, AddXferStatsRxBranch) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    nixlTelemetry telemetry(testFile_, "BUFFER");

    EXPECT_NO_THROW(telemetry.addXferStats(
        std::chrono::microseconds(50), false, 3000, std::chrono::microseconds(7)));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 4);
    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME);
    EXPECT_EQ(event.value_, 7);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_XFER_TIME);
    EXPECT_EQ(event.value_, 50);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_RX_BYTES);
    EXPECT_EQ(event.value_, 3000);
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_RX_REQUESTS_NUM);
    EXPECT_EQ(event.value_, 1);
    envHelper_.popVar();
}

TEST_F(telemetryTest, TelemetryEventStructure) {
    nixlTelemetryEvent event1(nixl_telemetry_event_type_t::AGENT_TX_BYTES, 42);

    EXPECT_EQ(TELEMETRY_VERSION, 4);
    EXPECT_EQ(sizeof(nixlTelemetryEvent), 16);
    EXPECT_EQ(event1.value_, 42);
    EXPECT_EQ(event1.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);
}

TEST(telemetryMetricContract, DescriptorIsUnifiedExporterSeriesContract) {
    using et = nixl_telemetry_event_type_t;

    struct expectedSeries {
        et type;
        const char *counter;
        const char *gauge;
    };

    const std::vector<expectedSeries> contract = {
        {et::AGENT_TX_BYTES, "agent_tx_bytes_total", "agent_tx_last_bytes"},
        {et::AGENT_RX_BYTES, "agent_rx_bytes_total", "agent_rx_last_bytes"},
        {et::AGENT_TX_REQUESTS_NUM, "agent_tx_requests_num_total", nullptr},
        {et::AGENT_RX_REQUESTS_NUM, "agent_rx_requests_num_total", nullptr},
        {et::AGENT_MEMORY_REGISTERED,
         "agent_memory_registered_total",
         "agent_memory_registered_last_bytes"},
        {et::AGENT_MEMORY_DEREGISTERED,
         "agent_memory_deregistered_total",
         "agent_memory_deregistered_last_bytes"},
        {et::AGENT_XFER_TIME, "agent_xfer_time_total", "agent_xfer_time"},
        {et::AGENT_XFER_POST_TIME, "agent_xfer_post_time_total", "agent_xfer_post_time"},
        {et::AGENT_TELEMETRY_EVENTS_DROPPED, "agent_telemetry_events_dropped_total", nullptr},
    };

    ASSERT_EQ(contract.size(), telemetry_metric_event_types.size());
    for (const auto &expected : contract) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(expected.type);
        ASSERT_NE(descriptor.counterName, nullptr);
        EXPECT_EQ(std::string(descriptor.counterName), expected.counter);
        if (expected.gauge == nullptr) {
            EXPECT_EQ(descriptor.gaugeName, nullptr);
        } else {
            ASSERT_NE(descriptor.gaugeName, nullptr);
            EXPECT_EQ(std::string(descriptor.gaugeName), expected.gauge);
        }
    }

    for (const auto type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        ASSERT_NE(descriptor.counterName, nullptr);
        const std::string counter(descriptor.counterName);
        EXPECT_EQ(counter.substr(counter.size() - std::string("_total").size()), "_total");
    }

    for (const auto type : telemetry_error_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        EXPECT_EQ(descriptor.counterName, nullptr);
        EXPECT_EQ(descriptor.gaugeName, nullptr);
    }
}

TEST_F(telemetryTest, ShortRunInterval) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    EXPECT_NO_THROW({ nixlTelemetry telemetry(testFile_, "BUFFER"); });
    envHelper_.popVar();
}

TEST_F(telemetryTest, LargeRunInterval) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "10000");

    EXPECT_NO_THROW({ nixlTelemetry telemetry(testFile_, "BUFFER"); });
    envHelper_.popVar();
}

TEST_F(telemetryTest, BufferOverflowHandling) {
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "4");

    nixlTelemetry telemetry(testFile_, "BUFFER");

    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(telemetry.updateTxBytes(i * 100));
    }

    envHelper_.popVar();
}

TEST_F(telemetryTest, CustomTelemetryDirectory) {
    fs::path custom_dir = testDir_ / "custom_telemetry";
    fs::create_directory(custom_dir);
    envHelper_.addVar(TELEMETRY_DIR_VAR, custom_dir.string());

    EXPECT_NO_THROW({
        std::string telemetry_file = "test_telemetry";
        nixlTelemetry telemetry(telemetry_file, "BUFFER");

        std::string file_path = custom_dir.string() + "/" + telemetry_file;

        EXPECT_TRUE(fs::exists(file_path));
    });
    envHelper_.popVar();
}

// Test concurrent access (basic thread safety)
TEST_F(telemetryTest, ConcurrentAccess) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    testFile_ = "test_concurrent_access";
    nixlTelemetry telemetry(testFile_, "BUFFER");

    const int num_threads = 4;
    const int operations_per_thread = 100;

    std::vector<std::thread> threads;

    // Create threads that perform different telemetry operations
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&telemetry, i]() {
            for (int j = 0; j < operations_per_thread; ++j) {
                switch (i % 4) {
                case 0:
                    telemetry.updateTxBytes(j * 100);
                    break;
                case 1:
                    telemetry.updateRxBytes(j * 50);
                    break;
                case 2:
                    telemetry.updateTxRequestsNum(j);
                    break;
                case 3:
                    telemetry.updateRxRequestsNum(j);
                    break;
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
        thread.join();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_ = operations_per_thread * num_threads;
    readPos_ = 0;
    writePos_ = size_;
    validateState();
    envHelper_.popVar();
}

TEST_F(telemetryTest, TelemetryAgentEventsOne) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    nixlTelemetry telemetry(testFile_, "BUFFER");

    // Add some agent events
    telemetry.updateTxBytes(1024);
    telemetry.updateRxBytes(2048);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify that only agent events are written (no backend events)
    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 2); // Only agent events

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);

    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_RX_BYTES);

    envHelper_.popVar();
}

TEST_F(telemetryTest, TelemetryAgentEventsTwo) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    nixlTelemetry telemetry(testFile_, "BUFFER");

    // Add agent events
    telemetry.updateTxBytes(1024);
    telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Verify that both agent and backend events are written
    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 2); // 2 agent events

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);

    buffer->pop(event);
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_ERR_BACKEND);

    envHelper_.popVar();
}

// The synthetic AGENT_TELEMETRY_EVENTS_DROPPED event must reach the BUFFER raw stream (it
// is what the NIX-1205 stress reader consumes) carrying the exact number of
// staging-queue drops.
//
// Deterministic and single-threaded. The BUFFER ring's usable capacity is
// size-1, so a *full* staging queue can never be flushed into it in one shot;
// the test therefore never fills the queue. It stages exactly size-2 accepted
// events via updateData, then makes each addXferStats call reject its whole
// 4-event batch (size+4 > capacity) WITHOUT growing the queue. The staged state
// is built with a tight CPU loop (microseconds) while the periodic flush
// interval is hundreds of milliseconds away, so the first (and only meaningful)
// flush observes exactly size-2 data events + one synthetic drop event = size-1
// = the ring's exact capacity. Nothing is dropped by the ring and the counts are
// exact regardless of hardware/scheduling (the timing margin is ~5 orders of
// magnitude). Later flushes see an empty queue and a zero delta, so they add
// nothing.
TEST_F(telemetryTest, DroppedEventsAppearInBufferStream) {
    constexpr uint64_t kCapacity = 1024;
    constexpr uint64_t kAccepted = kCapacity - 2; // leaves room for 4>free -> reject
    constexpr uint64_t kDroppedBatches = 10;
    constexpr uint64_t kEventsPerBatch = 4;
    constexpr uint64_t kExpectedDropped = kDroppedBatches * kEventsPerBatch;

    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, std::to_string(kCapacity));
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "200");
    testFile_ = "test_dropped_buffer_stream";
    const auto path = testDir_.string() + "/" + testFile_;

    {
        nixlTelemetry telemetry(testFile_, "BUFFER");
        for (uint64_t i = 0; i < kAccepted; ++i) {
            telemetry.updateTxBytes(i);
        }
        // Queue is at capacity-2; every 4-event batch is rejected as a whole and
        // leaves the queue untouched, so each call drops exactly 4.
        for (uint64_t i = 0; i < kDroppedBatches; ++i) {
            telemetry.addXferStats(
                std::chrono::microseconds(10), true, 2000, std::chrono::microseconds(1));
        }
        // Wait out several flush intervals: the first flush drains the staged
        // events and emits the drop delta; the instance stays alive so the
        // exporter keeps the ring file open.
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }

    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);

    uint64_t exported = 0;
    uint64_t dropped = 0;
    uint64_t dropped_events_seen = 0;
    nixlTelemetryEvent event;
    while (buffer->pop(event)) {
        if (event.eventType_ == nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED) {
            dropped += event.value_;
            ++dropped_events_seen;
        } else {
            ++exported;
        }
    }

    EXPECT_EQ(dropped_events_seen, 1u)
        << "exactly one synthetic drop event (one flush with a positive delta)";
    EXPECT_EQ(dropped, kExpectedDropped) << "synthetic must carry the exact staging-drop count";
    EXPECT_EQ(exported, kAccepted) << "all accepted events must reach the BUFFER stream";
    EXPECT_EQ(exported + dropped, kAccepted + kExpectedDropped)
        << "produced must equal exported + dropped (no silent loss)";

    envHelper_.popVar();
    envHelper_.popVar();
}
