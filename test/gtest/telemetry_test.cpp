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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <climits>
#include <atomic>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

#include "nixl.h"
#include "telemetry.h"
#include "telemetry_event.h"
#include "tracing/trace.h"
#include "tracing/trace_sink.h"
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
        testDir_ = fs::temp_directory_path() / "telemetry_test_files";
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

TEST_F(telemetryTest, ReaderReattachSeedsCachedPos) {
    // Cached write and read pos need to be seeded on attach
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    sharedRingBuffer<nixlTelemetryEvent> producer(path, true, TELEMETRY_VERSION, 8);

    for (size_t i = 0; i < 3; i++) {
        EXPECT_TRUE(producer.push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    // First consumer reads part of the stream, then gets destroyed
    {
        sharedRingBuffer<nixlTelemetryEvent> consumer1(path, false, TELEMETRY_VERSION);
        nixlTelemetryEvent out;
        for (size_t i = 0; i < 2; i++) {
            EXPECT_TRUE(consumer1.pop(out));
            EXPECT_EQ(out.value_, i);
        }
    }
    // A second consumer attaches mid-stream, still SPSC
    {
        // Must see exactly the one unread event, then empty
        sharedRingBuffer<nixlTelemetryEvent> consumer2(path, false, TELEMETRY_VERSION);
        nixlTelemetryEvent out;
        EXPECT_EQ(consumer2.size(), 1u);
        EXPECT_TRUE(consumer2.pop(out));
        EXPECT_EQ(out.value_, 2u);
        EXPECT_FALSE(consumer2.pop(out));
    }
    EXPECT_FALSE(producer.full());
}

TEST_F(telemetryTest, PushReloadsCachedReadPosWhenFull) {
    // push() only re-reads read_pos when the ring looks full: fill the ring,
    // fail one push, free one slot and push again to cover both reload outcomes
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    sharedRingBuffer<nixlTelemetryEvent> producer(path, true, TELEMETRY_VERSION, 8);
    sharedRingBuffer<nixlTelemetryEvent> consumer(path, false, TELEMETRY_VERSION);
    nixlTelemetryEvent out;

    for (size_t i = 0; i < 7; i++) {
        EXPECT_TRUE(producer.push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    // full: push reloads the cached read_pos and fails
    EXPECT_FALSE(producer.push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    EXPECT_TRUE(consumer.pop(out));
    EXPECT_EQ(out.value_, 0u);
    // one slot freed: the reload lets the push through
    EXPECT_TRUE(producer.push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    for (size_t i = 1; i <= 7; i++) {
        EXPECT_TRUE(consumer.pop(out));
        EXPECT_EQ(out.value_, i);
    }
    // empty: pop reloads the cached write_pos and fails
    EXPECT_FALSE(consumer.pop(out));

    // test the same for cached write_pos
    EXPECT_TRUE(producer.push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 8}));
    EXPECT_TRUE(consumer.pop(out));
    EXPECT_EQ(out.value_, 8u);
    EXPECT_FALSE(consumer.pop(out));
}

TEST_F(telemetryTest, NoAttachedConsumerBufferIsReset) {
    // A restarted producer resets the buffer in place if no consumer is attached, whether none
    // ever was or the one that was got destroyed
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    auto producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    nixlTelemetryEvent out;
    struct stat old_st{}, new_st{};

    for (size_t i = 0; i < 7; i++) {
        EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    EXPECT_FALSE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    // Producer exits with the ring full and restarts with nobody attached
    EXPECT_EQ(stat(path.c_str(), &old_st), 0);
    producer.reset();
    producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    EXPECT_EQ(stat(path.c_str(), &new_st), 0);
    EXPECT_EQ(old_st.st_ino, new_st.st_ino);
    EXPECT_EQ(producer->size(), 0u);
    EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));
    {
        sharedRingBuffer<nixlTelemetryEvent> consumer(path, false, TELEMETRY_VERSION);
        EXPECT_EQ(consumer.size(), 1u);
        EXPECT_TRUE(consumer.pop(out));
        EXPECT_EQ(out.value_, 7u);
        EXPECT_FALSE(consumer.pop(out));
    }

    // Same after a consumer was attached and destroyed: its lock went with it
    EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 8}));
    producer.reset();
    producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    EXPECT_EQ(stat(path.c_str(), &new_st), 0);
    EXPECT_EQ(old_st.st_ino, new_st.st_ino);
    EXPECT_EQ(producer->size(), 0u);
}

TEST_F(telemetryTest, NoAttachedConsumerIncompatibleBufferIsReplaced) {
    // With no consumer attached a capacity change replaces the file on a new inode, like with
    // one attached. The pin keeps the old inode alive so the filesystem cannot hand its number
    // to the new file
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    auto producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    struct stat old_st{}, new_st{};
    EXPECT_EQ(stat(path.c_str(), &old_st), 0);
    nixl::scopedFd pin(open(path.c_str(), O_RDONLY));

    producer.reset();
    producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 16);
    EXPECT_EQ(stat(path.c_str(), &new_st), 0);
    EXPECT_NE(old_st.st_ino, new_st.st_ino);
    EXPECT_EQ(producer->capacity(), 16u);

    sharedRingBuffer<nixlTelemetryEvent> consumer(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(consumer.capacity(), 16u);
    EXPECT_EQ(consumer.size(), 0u);
}

TEST_F(telemetryTest, ConsumerSurvivesProducerRestart) {
    // The ring being reused depends on the consumer's lock existing and the ring's
    // compatibility: an attached consumer must see the events left unread by the old
    // producer, and a producer restarted on a full ring must seed its cached read_pos
    // from the kept header: its first push fails until the consumer frees a slot
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    auto producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    sharedRingBuffer<nixlTelemetryEvent> consumer(path, false, TELEMETRY_VERSION);
    nixlTelemetryEvent out;

    for (size_t i = 0; i < 7; i++) {
        EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    EXPECT_FALSE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    // Producer exits with the ring full and restarts with the same path and capacity
    producer.reset();
    producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    EXPECT_FALSE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    EXPECT_TRUE(consumer.pop(out));
    EXPECT_EQ(out.value_, 0u);
    EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    EXPECT_EQ(consumer.size(), 7u);
    for (size_t i = 1; i <= 7; i++) {
        EXPECT_TRUE(consumer.pop(out));
        EXPECT_EQ(out.value_, i);
    }
    EXPECT_FALSE(consumer.pop(out));
}

TEST_F(telemetryTest, IncompatibleBufferIsReplaced) {
    // A restarted producer with an incompatible header and a consumer attached replaces the file
    // on a new inode: the old consumer keeps its mapping and only drains what it has left, a new
    // consumer attaches to the new producer and sees only its events. The old ring is full, the
    // new one starts empty
    auto path = std::string(testDir_.string()) + "/" + testFile_;
    auto producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 8);
    sharedRingBuffer<nixlTelemetryEvent> consumer(path, false, TELEMETRY_VERSION);
    nixlTelemetryEvent out;
    struct stat old_st{}, new_st{};

    for (size_t i = 0; i < 7; i++) {
        EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    EXPECT_FALSE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 7}));

    // Producer exits with the ring full and restarts with a bigger capacity
    EXPECT_EQ(stat(path.c_str(), &old_st), 0);
    producer.reset();
    producer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, true, TELEMETRY_VERSION, 16);
    EXPECT_EQ(stat(path.c_str(), &new_st), 0);
    EXPECT_NE(old_st.st_ino, new_st.st_ino);
    sharedRingBuffer<nixlTelemetryEvent> consumer2(path, false, TELEMETRY_VERSION);
    // The new ring starts empty: fill it until a push fails
    for (size_t i = 7; i < 22; i++) {
        EXPECT_TRUE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, i}));
    }
    EXPECT_FALSE(producer->push({nixl_telemetry_event_type_t::AGENT_TX_BYTES, 22}));

    EXPECT_EQ(consumer.size(), 7u);
    for (size_t i = 0; i < 7; i++) {
        EXPECT_TRUE(consumer.pop(out));
        EXPECT_EQ(out.value_, i);
    }
    EXPECT_FALSE(consumer.pop(out));

    EXPECT_EQ(consumer2.size(), 15u);
    for (size_t i = 7; i < 22; i++) {
        EXPECT_TRUE(consumer2.pop(out));
        EXPECT_EQ(out.value_, i);
    }
    EXPECT_FALSE(consumer2.pop(out));
}

TEST_F(telemetryTest, NonexistentExporterThrows) {
    // An explicitly requested exporter that cannot be loaded must surface the
    // failure rather than silently disabling telemetry.

    // The plugin manager logs an expected warning when the .so is not found.
    gtest::LogIgnoreGuard ignore_missing_plugin("Plugin file does not exist");

    EXPECT_THROW(
        { nixlTelemetry telemetry(testFile_, "nixl_nonexistent_exporter"); }, std::runtime_error);
}

TEST_F(telemetryTest, ExporterBindFailureRunsWithoutSink) {
    envHelper_.addVar(TELEMETRY_EXPORTER_VAR, "mock_bind_failed");
    const gtest::LogIgnoreGuard ignore_missing_plugin("Plugin file does not exist");
    const gtest::LogIgnoreGuard ignore_sinkless("will run without a telemetry sink");

    std::unique_ptr<nixlTelemetry> telemetry;
    EXPECT_NO_THROW(telemetry = nixlTelemetry::create(testFile_));
    EXPECT_EQ(telemetry, nullptr);
    EXPECT_EQ(ignore_sinkless.getIgnoredCount(), 1u);

    envHelper_.popVar();
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

    EXPECT_EQ(TELEMETRY_VERSION, 6);
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
        {et::AGENT_TRACE_PHASES_DROPPED, "agent_trace_phases_dropped_total", nullptr},
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

// A subset allowlist exports only the listed event; deactivated metrics are
// skipped at the source and never enter the staging queue (BUFFER here).
TEST_F(telemetryTest, MetricAllowlistSubset) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    envHelper_.addVar(TELEMETRY_ENABLED_METRICS_VAR, "agent_tx_bytes");
    testFile_ = "test_allowlist_subset";

    {
        nixlTelemetry telemetry(testFile_, "BUFFER");
        telemetry.updateTxBytes(1024); // allowed
        telemetry.updateRxBytes(2048); // filtered
        telemetry.updateMemoryRegistered(4096); // filtered
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 1);
    nixlTelemetryEvent event;
    ASSERT_TRUE(buffer->pop(event));
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_TX_BYTES);
    EXPECT_EQ(event.value_, 1024);

    envHelper_.popVar(); // TELEMETRY_ENABLED_METRICS_VAR
    envHelper_.popVar(); // TELEMETRY_RUN_INTERVAL_VAR
}

// A family glob (agent_err_*) enables every matching event and nothing else.
TEST_F(telemetryTest, MetricAllowlistFamilyGlob) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    envHelper_.addVar(TELEMETRY_ENABLED_METRICS_VAR, "agent_err_*");
    testFile_ = "test_allowlist_family";

    {
        nixlTelemetry telemetry(testFile_, "BUFFER");
        telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND); // allowed
        telemetry.updateTxBytes(1024); // filtered
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 1);
    nixlTelemetryEvent event;
    ASSERT_TRUE(buffer->pop(event));
    EXPECT_EQ(event.eventType_, nixl_telemetry_event_type_t::AGENT_ERR_BACKEND);

    envHelper_.popVar(); // TELEMETRY_ENABLED_METRICS_VAR
    envHelper_.popVar(); // TELEMETRY_RUN_INTERVAL_VAR
}

// An allowlist whose tokens match nothing warns and exports nothing.
TEST_F(telemetryTest, MetricAllowlistUnknownTokenExportsNothing) {
    gtest::LogIgnoreGuard ignore_unknown("no telemetry metric matches");
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    envHelper_.addVar(TELEMETRY_ENABLED_METRICS_VAR, "nope_*, also_missing");
    testFile_ = "test_allowlist_unknown";

    {
        nixlTelemetry telemetry(testFile_, "BUFFER");
        telemetry.updateTxBytes(1024);
        telemetry.updateRxBytes(2048);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 0);

    envHelper_.popVar(); // TELEMETRY_ENABLED_METRICS_VAR
    envHelper_.popVar(); // TELEMETRY_RUN_INTERVAL_VAR
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

namespace {

class throwingTraceBackend final : public nixl::trace::TraceBackend {
public:
    [[nodiscard]] std::unique_ptr<nixl::trace::SpanBackend>
    beginSpan(std::string_view, nixl::trace::Kind) override {
        throw std::runtime_error("beginSpan failed");
    }

    void
    mark(std::string_view, nixl::trace::Kind) override {}

    void
    emitSpan(std::string_view,
             nixl::trace::Kind,
             const nixl::trace::TraceContext &,
             std::uint64_t,
             nixlTime::us_t,
             nixlTime::us_t,
             std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitSpan failed");
    }

    void
    emitEvent(std::string_view,
              nixl::trace::Kind,
              const nixl::trace::TraceContext &,
              std::uint64_t,
              nixlTime::us_t,
              std::span<const nixl::trace::Attribute>) override {
        throw std::runtime_error("emitEvent failed");
    }

    void
    pushContext(const nixl::trace::TraceContext &, std::uint64_t) override {}

    void
    popContext() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "throwing";
    }
};

} // namespace

TEST_F(telemetryTest, TracePhaseDropsAreCounted) {
    constexpr uint64_t kDrops = 3;
    const gtest::LogIgnoreGuard lig("Dropping a trace phase");
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    testFile_ = "test_trace_phase_drops";

    {
        nixlTelemetry telemetry(testFile_, "BUFFER");
        std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
        backends.push_back(std::make_unique<throwingTraceBackend>());
        nixl::trace::Tracer tracer(std::move(backends), 0.0);
        nixl::trace::TracerPhaseSink sink{tracer, "UCX", &telemetry};
        for (uint64_t i = 0; i < kDrops; ++i) {
            sink.recordPhase(nixl_trace_phase_t::SUBMIT, {}, i, {}, nullptr, 0, 0);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto path = testDir_.string() + "/" + testFile_;
    auto buffer =
        std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path, false, TELEMETRY_VERSION);
    uint64_t dropped = 0;
    nixlTelemetryEvent event;
    while (buffer->pop(event)) {
        if (event.eventType_ == nixl_telemetry_event_type_t::AGENT_TRACE_PHASES_DROPPED) {
            dropped += event.value_;
        }
    }
    EXPECT_EQ(dropped, kDrops);

    envHelper_.popVar();
}

class telemetryXferTest : public telemetryTest {
protected:
    using et = nixl_telemetry_event_type_t;

    void
    SetUp() override {
        telemetryTest::SetUp();
        envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
        agentName_ = testing::UnitTest::GetInstance()->current_test_info()->name();
        agent_ = std::make_unique<nixlAgent>(agentName_, nixlAgentConfig());

        nixl_b_params_t params;
        engine_.SetToParams(params);
        nixlBackendH *backend = nullptr;
        ASSERT_EQ(agent_->createBackend(gtest::GetMockBackendName(), params, backend),
                  NIXL_SUCCESS);
        extraParams_.backends.push_back(backend);
        ASSERT_EQ(agent_->registerMem(bufferRegDlist(), &extraParams_), NIXL_SUCCESS);
    }

    void
    TearDown() override {
        agent_.reset();
        envHelper_.popVar();
        telemetryTest::TearDown();
    }

    [[nodiscard]] nixlBasicDesc
    bufferDesc() const {
        return nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer_.data()), buffer_.size(), 0);
    }

    [[nodiscard]] nixl_reg_dlist_t
    bufferRegDlist() const {
        nixl_reg_dlist_t reg_dlist(DRAM_SEG);
        reg_dlist.addDesc(nixlBlobDesc(bufferDesc(), ""));
        return reg_dlist;
    }

    [[nodiscard]] nixlXferReqH *
    createLoopbackReq() const {
        nixl_xfer_dlist_t xfer_dlist(DRAM_SEG);
        xfer_dlist.addDesc(bufferDesc());
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(agent_->createXferReq(
                      NIXL_WRITE, xfer_dlist, xfer_dlist, agentName_, req, &extraParams_),
                  NIXL_SUCCESS);
        return req;
    }

    [[nodiscard]] std::vector<nixlTelemetryEvent>
    flushedEvents() {
        EXPECT_EQ(agent_->deregisterMem(bufferRegDlist(), &extraParams_), NIXL_SUCCESS);
        sharedRingBuffer<nixlTelemetryEvent> ring(
            (testDir_ / agentName_).string(), false, TELEMETRY_VERSION);
        std::vector<nixlTelemetryEvent> events;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (countOf(events, et::AGENT_MEMORY_DEREGISTERED) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            for (nixlTelemetryEvent event; ring.pop(event);) {
                events.push_back(event);
            }
        }
        EXPECT_EQ(countOf(events, et::AGENT_MEMORY_DEREGISTERED), 1);
        return events;
    }

    [[nodiscard]] static std::ptrdiff_t
    countOf(const std::vector<nixlTelemetryEvent> &events, et type) {
        return std::ranges::count(events, type, &nixlTelemetryEvent::eventType_);
    }

    [[nodiscard]] static std::ptrdiff_t
    finishes(const std::vector<nixlTelemetryEvent> &events) {
        const auto requests = countOf(events, et::AGENT_TX_REQUESTS_NUM);
        EXPECT_EQ(countOf(events, et::AGENT_TX_BYTES), requests);
        EXPECT_EQ(countOf(events, et::AGENT_XFER_TIME), requests);
        EXPECT_EQ(countOf(events, et::AGENT_XFER_POST_TIME), requests);
        return requests;
    }

    [[nodiscard]] static uint64_t
    sumOf(const std::vector<nixlTelemetryEvent> &events, et type) {
        uint64_t sum = 0;
        for (const auto &event : events) {
            if (event.eventType_ == type) {
                sum += event.value_;
            }
        }
        return sum;
    }

    testing::NiceMock<mocks::GMockBackendEngine> engine_;
    std::vector<char> buffer_ = std::vector<char>(1024);
    std::string agentName_;
    std::unique_ptr<nixlAgent> agent_;
    nixl_opt_args_t extraParams_;
};

TEST_F(telemetryXferTest, PostCompletedAtOnceIsCountedOnce) {
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->getXferStatus(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(finishes(events), 1);
    EXPECT_EQ(sumOf(events, et::AGENT_TX_BYTES), buffer_.size());
}

TEST_F(telemetryXferTest, CompletionSeenByTheRepostCheckIsCounted) {
    EXPECT_CALL(engine_, postXfer)
        .WillOnce(testing::Return(NIXL_IN_PROG))
        .WillOnce(testing::Return(NIXL_SUCCESS));
    EXPECT_CALL(engine_, checkXfer).WillOnce(testing::Return(NIXL_SUCCESS));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(finishes(events), 2);
    EXPECT_EQ(sumOf(events, et::AGENT_TX_BYTES), 2 * buffer_.size());
}

TEST_F(telemetryXferTest, CompletionSeenByReleaseIsCountedButCancelledIsNot) {
    EXPECT_CALL(engine_, postXfer).Times(2).WillRepeatedly(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer)
        .WillOnce(testing::Return(NIXL_SUCCESS))
        .WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, releaseReqH).WillOnce(testing::Return(NIXL_SUCCESS));
    nixlXferReqH *completed = createLoopbackReq();
    nixlXferReqH *cancelled = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(completed), NIXL_IN_PROG);
    EXPECT_EQ(agent_->postXferReq(cancelled), NIXL_IN_PROG);
    EXPECT_EQ(agent_->releaseXferReq(completed), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(cancelled), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(finishes(events), 1);
    EXPECT_EQ(sumOf(events, et::AGENT_TX_BYTES), buffer_.size());
}

TEST_F(telemetryXferTest, RejectedRepostKeepsTheInFlightTiming) {
    constexpr auto in_flight = std::chrono::milliseconds(50);
    const gtest::LogIgnoreGuard lig("cannot be reposted");
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer)
        .WillOnce(testing::Return(NIXL_IN_PROG))
        .WillOnce(testing::Return(NIXL_SUCCESS));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    std::this_thread::sleep_for(in_flight);
    const auto rejected_at = std::chrono::steady_clock::now();
    EXPECT_EQ(agent_->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
    EXPECT_EQ(agent_->getXferStatus(req), NIXL_SUCCESS);

    nixl_xfer_telem_t telemetry{};
    EXPECT_EQ(agent_->getXferTelemetry(req, telemetry), NIXL_SUCCESS);
    EXPECT_LT(telemetry.startTime, rejected_at);
    EXPECT_GE(telemetry.xferDuration, in_flight);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
}

TEST_F(telemetryXferTest, ErrorSeenByTheRepostCheckIsCountedOnce) {
    EXPECT_CALL(engine_, postXfer)
        .WillOnce(testing::Return(NIXL_IN_PROG))
        .WillOnce(testing::Return(NIXL_SUCCESS));
    EXPECT_CALL(engine_, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(countOf(events, et::AGENT_ERR_BACKEND), 1);
    EXPECT_EQ(finishes(events), 1);
}

TEST_F(telemetryXferTest, ErrorSeenByReleaseIsCountedOnce) {
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(countOf(events, et::AGENT_ERR_BACKEND), 1);
    EXPECT_EQ(finishes(events), 0);
}

TEST_F(telemetryXferTest, RejectedRepostIsCountedAsRepostActive) {
    const gtest::LogIgnoreGuard lig("cannot be reposted");
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer)
        .WillOnce(testing::Return(NIXL_IN_PROG))
        .WillOnce(testing::Return(NIXL_SUCCESS));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
    EXPECT_EQ(agent_->getXferStatus(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(countOf(events, et::AGENT_ERR_REPOST_ACTIVE), 1);
    EXPECT_EQ(finishes(events), 1);
}

TEST_F(telemetryXferTest, DisconnectSeenByThePostIsCounted) {
    const gtest::LogIgnoreGuard lig("was disconnected");
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_ERR_REMOTE_DISCONNECT));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    const auto events = flushedEvents();
    EXPECT_EQ(countOf(events, et::AGENT_ERR_REMOTE_DISCONNECT), 1);
    EXPECT_EQ(finishes(events), 0);
}

TEST_F(telemetryXferTest, DisconnectSeenByTheRepostCheckIsCounted) {
    const gtest::LogIgnoreGuard lig("was disconnected");
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer).WillOnce(testing::Return(NIXL_ERR_REMOTE_DISCONNECT));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->postXferReq(req), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    EXPECT_EQ(countOf(flushedEvents(), et::AGENT_ERR_REMOTE_DISCONNECT), 1);
}

TEST_F(telemetryXferTest, DisconnectSeenByTheStatusCheckIsCounted) {
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer).WillOnce(testing::Return(NIXL_ERR_REMOTE_DISCONNECT));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->getXferStatus(req), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    EXPECT_EQ(countOf(flushedEvents(), et::AGENT_ERR_REMOTE_DISCONNECT), 1);
}

TEST_F(telemetryXferTest, FailedReleaseIsCountedAsRepostActive) {
    const gtest::LogIgnoreGuard lig("could not release transfer request");
    EXPECT_CALL(engine_, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
    EXPECT_CALL(engine_, checkXfer)
        .WillOnce(testing::Return(NIXL_IN_PROG))
        .WillOnce(testing::Return(NIXL_SUCCESS));
    EXPECT_CALL(engine_, releaseReqH).WillOnce(testing::Return(NIXL_ERR_BACKEND));
    nixlXferReqH *req = createLoopbackReq();

    EXPECT_EQ(agent_->postXferReq(req), NIXL_IN_PROG);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_ERR_REPOST_ACTIVE);
    EXPECT_EQ(agent_->getXferStatus(req), NIXL_SUCCESS);
    EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);

    EXPECT_EQ(countOf(flushedEvents(), et::AGENT_ERR_REPOST_ACTIVE), 1);
}
