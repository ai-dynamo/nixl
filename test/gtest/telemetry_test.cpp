/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common.h"
#include "backend/backend_engine.h"
#include "mocks/gmock_engine.h"

namespace fs = std::filesystem;
constexpr char TELEMETRY_ENABLED_VAR[] = "NIXL_TELEMETRY_ENABLE";

// Custom mock backend class for testing backend telemetry events
class telemetryTestBackend : public mocks::GMockBackendEngine {
public:
    telemetryTestBackend(const nixlBackendInitParams *init_params)
        : mocks::GMockBackendEngine(init_params) {}

    void
    addTestTelemetryEvent(const std::string &event_name, uint64_t value) {
        addTelemetryEvent(event_name, value);
    }
};

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
        auto path = fs::path(testDir_) / testFile_;
        auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
            path.string(), false, TELEMETRY_VERSION);
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
    backend_map_t backendMap_;
};

TEST_F(telemetryTest, BasicInitialization) {
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(testFile_, backendMap_);
        validateState();
    });
}

TEST_F(telemetryTest, InitializationWithEmptyFileName) {
    EXPECT_THROW({ nixlTelemetry telemetry("", backendMap_); }, std::invalid_argument);
}

TEST_F(telemetryTest, CustomBufferSize) {
    auto tmp_capacity = capacity_;
    capacity_ = 32;
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "32");

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(testFile_, backendMap_);
        validateState();
    });
    capacity_ = tmp_capacity;
    envHelper_.popVar();
}

TEST_F(telemetryTest, InvalidBufferSize) {
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "0");

    EXPECT_THROW({ nixlTelemetry telemetry(testFile_, backendMap_); }, std::invalid_argument);
    envHelper_.popVar();
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "1023");
    EXPECT_THROW({ nixlTelemetry telemetry(testFile_, backendMap_); }, std::invalid_argument);
    envHelper_.popVar();
}

// Test transfer bytes tracking
TEST_F(telemetryTest, TransferBytesTracking) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    nixlTelemetry telemetry(testFile_, backendMap_);

    EXPECT_NO_THROW(telemetry.updateTxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateRxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateTxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateRxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND));
    EXPECT_NO_THROW(telemetry.updateMemoryRegistered(1024));
    EXPECT_NO_THROW(telemetry.updateMemoryDeregistered(1024));
    EXPECT_NO_THROW(telemetry.addXferTime(std::chrono::microseconds(100), true, 2000));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 10);
    EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
    EXPECT_EQ(buffer->capacity(), capacity_);
    EXPECT_EQ(buffer->empty(), false);
    EXPECT_EQ(buffer->full(), false);
    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_bytes");
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_rx_bytes");
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_requests_num");
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_rx_requests_num");
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_,
                 nixlEnumStrings::statusStr(nixl_status_t::NIXL_ERR_BACKEND).c_str());
    EXPECT_EQ(event.value_, 1);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_memory_registered");
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_memory_deregistered");
    EXPECT_EQ(event.value_, 1024);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_xfer_time");
    EXPECT_EQ(event.value_, 100);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_bytes");
    EXPECT_EQ(event.value_, 2000);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_requests_num");
    EXPECT_EQ(event.value_, 1);
    envHelper_.popVar();
}

TEST_F(telemetryTest, TelemetryEventStructure) {
    nixlTelemetryEvent event1(
        1234567890, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, "test_event", 42);

    EXPECT_EQ(event1.timestampUs_, 1234567890);
    EXPECT_EQ(event1.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);
    EXPECT_EQ(event1.value_, 42);
    EXPECT_STREQ(event1.eventName_, "test_event");
}

TEST_F(telemetryTest, ShortRunInterval) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    EXPECT_NO_THROW({ nixlTelemetry telemetry(testFile_, backendMap_); });
    envHelper_.popVar();
}

TEST_F(telemetryTest, LargeRunInterval) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "10000");

    EXPECT_NO_THROW({ nixlTelemetry telemetry(testFile_, backendMap_); });
    envHelper_.popVar();
}

TEST_F(telemetryTest, BufferOverflowHandling) {
    envHelper_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "4");

    nixlTelemetry telemetry(testFile_, backendMap_);

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
        nixlTelemetry telemetry(testFile_, backendMap_);

        fs::path telemetry_file = custom_dir / testFile_;
        EXPECT_TRUE(fs::exists(telemetry_file));
    });
    envHelper_.popVar();
}

TEST_F(telemetryTest, TelemetryCategoryStringConversion) {
    for (int i = 0; i < static_cast<int>(nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM) + 1;
         ++i) {
        auto category = static_cast<nixl_telemetry_category_t>(i);
        std::string category_str = nixlEnumStrings::telemetryCategoryStr(category);
        EXPECT_FALSE(category_str.empty());
        EXPECT_NE(category_str, "BAD_CATEGORY");
    }

    auto invalid_category = static_cast<nixl_telemetry_category_t>(999);
    std::string invalid_str = nixlEnumStrings::telemetryCategoryStr(invalid_category);
    EXPECT_EQ(invalid_str, "BAD_CATEGORY");
}

// Test concurrent access (basic thread safety)
TEST_F(telemetryTest, ConcurrentAccess) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    testFile_ = "test_concurrent_access";
    nixlTelemetry telemetry(testFile_, backendMap_);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    size_ = operations_per_thread * num_threads;
    readPos_ = 0;
    writePos_ = size_;
    validateState();
    envHelper_.popVar();
}

TEST_F(telemetryTest, BackendTelemetryEventsCollection) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    nixlBackendInitParams init_params;
    init_params.enableTelemetry_ = true;
    nixl_b_params_t custom_params;
    init_params.customParams = &custom_params;
    // Create mock backends and add them to the backend map
    auto mock_backend1 = std::make_unique<telemetryTestBackend>(&init_params);
    auto mock_backend2 = std::make_unique<telemetryTestBackend>(&init_params);

    backendMap_["CUSTOM"] = mock_backend1.get();
    backendMap_["GPUNETIO"] = mock_backend2.get();

    nixlTelemetry telemetry(testFile_, backendMap_);

    // Add some telemetry events to the backends
    mock_backend1->addTestTelemetryEvent("backend1_event1", 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mock_backend1->addTestTelemetryEvent("backend1_event2", 200);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mock_backend2->addTestTelemetryEvent("backend2_event1", 300);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    // Verify that backend events are collected and written to buffer
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 3); // Should have 3 backend events

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend1_event1");
    EXPECT_EQ(event.value_, 100);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend1_event2");
    EXPECT_EQ(event.value_, 200);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend2_event1");
    EXPECT_EQ(event.value_, 300);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    envHelper_.popVar();
}

TEST_F(telemetryTest, BackendTelemetryEventsEmptyBackendMap) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    // Create telemetry with empty backend map
    backend_map_t empty_backend_map;
    nixlTelemetry telemetry(testFile_, empty_backend_map);

    // Add some agent events
    telemetry.updateTxBytes(1024);
    telemetry.updateRxBytes(2048);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Verify that only agent events are written (no backend events)
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 2); // Only agent events

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_bytes");
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_rx_bytes");
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);

    envHelper_.popVar();
}

TEST_F(telemetryTest, BackendTelemetryEventsMixedWithAgentEvents) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    nixlBackendInitParams init_params;
    nixl_b_params_t custom_params;
    init_params.customParams = &custom_params;
    init_params.enableTelemetry_ = true;
    auto mock_backend = std::make_unique<telemetryTestBackend>(&init_params);
    backendMap_["CUSTOM"] = mock_backend.get();

    nixlTelemetry telemetry(testFile_, backendMap_);

    // Add agent events
    telemetry.updateTxBytes(1024);
    telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND);

    // Add backend events
    mock_backend->addTestTelemetryEvent("backend_event1", 100);
    mock_backend->addTestTelemetryEvent("backend_event2", 200);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Verify that both agent and backend events are written
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 4); // 2 agent events + 2 backend events

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "agent_tx_bytes");
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_,
                 nixlEnumStrings::statusStr(nixl_status_t::NIXL_ERR_BACKEND).c_str());
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend_event1");
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend_event2");
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    envHelper_.popVar();
}

TEST_F(telemetryTest, BackendTelemetryEventsDisabledTelemetry) {
    // Disable telemetry
    unsetenv(TELEMETRY_ENABLED_VAR);

    nixlBackendInitParams init_params;
    nixl_b_params_t custom_params;
    init_params.customParams = &custom_params;
    init_params.enableTelemetry_ = false;
    auto mock_backend = std::make_unique<telemetryTestBackend>(&init_params);
    backendMap_["CUSTOM"] = mock_backend.get();

    nixlTelemetry telemetry(testFile_, backendMap_);
    // Add backend events (should be ignored)
    mock_backend->addTestTelemetryEvent("backend_event_disabled", 100);

    // Wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Verify that no events are written
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 0);

    // Restore environment variable
    setenv(TELEMETRY_ENABLED_VAR, "y", 1);
}

TEST_F(telemetryTest, BackendTelemetryEventsMultipleBackends) {
    envHelper_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    // Create multiple mock backends
    nixlBackendInitParams init_params;
    nixl_b_params_t custom_params;
    init_params.customParams = &custom_params;
    init_params.enableTelemetry_ = true;
    auto mock_backend1 = std::make_unique<telemetryTestBackend>(&init_params);
    auto mock_backend2 = std::make_unique<telemetryTestBackend>(&init_params);
    auto mock_backend3 = std::make_unique<telemetryTestBackend>(&init_params);

    backendMap_["CUSTOM"] = mock_backend1.get();
    backendMap_["GPUNETIO"] = mock_backend2.get();
    backendMap_["GDS_MT"] = mock_backend3.get();

    nixlTelemetry telemetry(testFile_, backendMap_);

    // Add events to each backend
    mock_backend1->addTestTelemetryEvent("backend1_event", 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mock_backend2->addTestTelemetryEvent("backend2_event", 200);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mock_backend3->addTestTelemetryEvent("backend3_event", 300);

    // Wait for the telemetry to be written
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Verify that events from all backends are collected
    auto path = fs::path(testDir_) / testFile_;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);

    EXPECT_EQ(buffer->size(), 3);

    nixlTelemetryEvent event;
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend1_event");
    EXPECT_EQ(event.value_, 100);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend2_event");
    EXPECT_EQ(event.value_, 200);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "backend3_event");
    EXPECT_EQ(event.value_, 300);
    EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);

    envHelper_.popVar();
}
