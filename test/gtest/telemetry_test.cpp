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
#include "nixl_types.h"
#include "common.h"

namespace fs = std::filesystem;

class telemetryTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        test_dir = "./telemetry_test_files";
        test_file = "test_telemetry";
        if (!fs::exists(test_dir)) {
            fs::create_directory(test_dir);
        }

        env_helper.addVar(TELEMETRY_ENABLED_VAR, "1");
        env_helper.addVar(TELEMETRY_DIR_VAR, test_dir.string());
    }

    void
    TearDown() override {
        env_helper.popVar();
        env_helper.popVar();
        if (fs::exists(test_dir)) {
            try {
                fs::remove_all(test_dir);
            }
            catch (const fs::filesystem_error &e) {
                // ignore can fail due to nsf
            }
        }
    }

    void
    validateState() {
        auto path = fs::path(test_dir) / test_file;
        auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
            path.string(), false, TELEMETRY_VERSION);
        EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
        EXPECT_EQ(buffer->capacity(), capacity_);
        EXPECT_EQ(buffer->size(), size_);
        EXPECT_EQ(buffer->empty(), size_ == 0);
        EXPECT_EQ(buffer->full(), size_ == capacity_);
    }

    fs::path test_dir;
    std::string test_file;
    gtest::ScopedEnv env_helper;
    size_t capacity_ = 4096;
    size_t size_ = 0;
    size_t read_pos_ = 0;
    size_t write_pos_ = 0;
    size_t mask_ = 4096 - 1;
};

TEST_F(telemetryTest, BasicInitialization) {
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());
        validateState();
    });
}

TEST_F(telemetryTest, InitializationWithEmptyFileName) {
    auto tmp_file = test_file;
    test_file = TELEMETRY_PREFIX + std::string(".") + std::to_string(getpid());
    EXPECT_NO_THROW({
        nixlTelemetry telemetry("");
        validateState();
        EXPECT_TRUE(telemetry.isEnabled());
    });

    test_file = tmp_file;
}

TEST_F(telemetryTest, TelemetryDisabled) {
    std::string test_file = "test_telemetry_disabled";

    unsetenv(TELEMETRY_ENABLED_VAR);
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_FALSE(telemetry.isEnabled());
        EXPECT_TRUE(fs::is_empty(test_dir));
    });

    // Restore environment variable
    setenv(TELEMETRY_ENABLED_VAR, "1", 1);
}

TEST_F(telemetryTest, CustomBufferSize) {
    auto tmp_capacity = capacity_;
    capacity_ = 32;
    env_helper.addVar(TELEMETRY_BUFFER_SIZE_VAR, "32");

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        validateState();
        EXPECT_TRUE(telemetry.isEnabled());
    });
    capacity_ = tmp_capacity;
    env_helper.popVar();
}

TEST_F(telemetryTest, InvalidBufferSize) {
    env_helper.addVar(TELEMETRY_BUFFER_SIZE_VAR, "0");

    EXPECT_THROW({ nixlTelemetry telemetry(test_file); }, std::invalid_argument);
    env_helper.popVar();
    env_helper.addVar(TELEMETRY_BUFFER_SIZE_VAR, "1023");
    EXPECT_THROW({ nixlTelemetry telemetry(test_file); }, std::invalid_argument);
    env_helper.popVar();
}

// Test transfer bytes tracking
TEST_F(telemetryTest, TransferBytesTracking) {
    env_helper.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    nixlTelemetry telemetry(test_file);

    EXPECT_NO_THROW(telemetry.updateTxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateRxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateTxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateRxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND));
    EXPECT_NO_THROW(telemetry.updateMemoryRegistered(1024));
    EXPECT_NO_THROW(telemetry.updateMemoryDeregistered(1024));
    EXPECT_NO_THROW(telemetry.addTransactionTime(std::chrono::microseconds(100)));
    EXPECT_NO_THROW(telemetry.addBackendTelemetry("test_event", 100));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto path = fs::path(test_dir) / test_file;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 9);
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
    EXPECT_STREQ(event.eventName_, "agent_transaction_time");
    EXPECT_EQ(event.value_, 100);
    buffer->pop(event);
    EXPECT_STREQ(event.eventName_, "test_event");
    EXPECT_EQ(event.value_, 100);
    env_helper.popVar();
}

TEST_F(telemetryTest, TelemetryEventStructure) {
    nixlTelemetryEvent event1(
        1234567890, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER, "test_event", 42);

    EXPECT_EQ(event1.timestampUs_, 1234567890);
    EXPECT_EQ(event1.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);
    EXPECT_EQ(event1.value_, 42);
    EXPECT_STREQ(event1.eventName_, "test_event");
}

TEST_F(telemetryTest, EmptyEventNames) {
    std::string test_file = "test_empty_eventName_s";
    nixlTelemetry telemetry(test_file);

    EXPECT_NO_THROW(telemetry.addBackendTelemetry("", 100));
}

TEST_F(telemetryTest, ShortRunInterval) {
    env_helper.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    std::string test_file = "test_short_interval";

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());
    });
    env_helper.popVar();
}

TEST_F(telemetryTest, LargeRunInterval) {
    env_helper.addVar(TELEMETRY_RUN_INTERVAL_VAR, "10000");

    std::string test_file = "test_large_interval";

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());
    });
    env_helper.popVar();
}

TEST_F(telemetryTest, BufferOverflowHandling) {
    env_helper.addVar(TELEMETRY_BUFFER_SIZE_VAR, "4");

    std::string test_file = "test_buffer_overflow";
    nixlTelemetry telemetry(test_file);

    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(telemetry.updateTxBytes(i * 100));
    }

    env_helper.popVar();
}

TEST_F(telemetryTest, CustomTelemetryDirectory) {
    fs::path custom_dir = test_dir / "custom_telemetry";
    fs::create_directory(custom_dir);
    env_helper.addVar(TELEMETRY_DIR_VAR, custom_dir.string());

    std::string test_file = "test_custom_dir";
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());

        fs::path telemetry_file = custom_dir / test_file;
        EXPECT_TRUE(fs::exists(telemetry_file));
    });
    env_helper.popVar();
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
    env_helper.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    test_file = "test_concurrent_access";
    nixlTelemetry telemetry(test_file);

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
    read_pos_ = 0;
    write_pos_ = size_;
    validateState();
    env_helper.popVar();
}

TEST_F(telemetryTest, RuntimeEnableDisable) {
    env_helper.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");
    // start with telemetry disabled
    unsetenv(TELEMETRY_ENABLED_VAR);

    nixlTelemetry telemetry(test_file);
    EXPECT_FALSE(telemetry.isEnabled());

    // enable telemetry
    setenv(TELEMETRY_ENABLED_VAR, "1", 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    EXPECT_TRUE(telemetry.isEnabled());

    telemetry.updateTxBytes(1024);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    auto path = fs::path(test_dir) / test_file;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);
    EXPECT_EQ(buffer->size(), 1);

    // disable telemetry
    unsetenv(TELEMETRY_ENABLED_VAR);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    EXPECT_FALSE(telemetry.isEnabled());

    size_t initial_size = buffer->size();
    // will be ignored because telemetry is disabled
    telemetry.updateRxBytes(1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(buffer->size(), initial_size);

    // re-enable telemetry
    setenv(TELEMETRY_ENABLED_VAR, "1", 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    EXPECT_TRUE(telemetry.isEnabled());
    telemetry.updateRxBytes(2048);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
        path.string(), false, TELEMETRY_VERSION);
    // buffer is reset so it will be 1
    EXPECT_EQ(buffer->size(), initial_size);
    env_helper.popVar();
}

TEST_F(telemetryTest, CleanupOnDisable) {
    nixlTelemetry telemetry(test_file);
    EXPECT_TRUE(telemetry.isEnabled());

    // Add some events
    telemetry.updateTxBytes(1024);
    telemetry.updateRxBytes(2048);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Disable telemetry
    unsetenv(TELEMETRY_ENABLED_VAR);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));

    EXPECT_FALSE(telemetry.isEnabled());

    // Verify that adding events when disabled doesn't cause issues
    EXPECT_NO_THROW(telemetry.updateTxBytes(4096));
    EXPECT_NO_THROW(telemetry.updateRxBytes(8192));
    EXPECT_NO_THROW(telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND));
    EXPECT_NO_THROW(telemetry.updateMemoryRegistered(1024));
    EXPECT_NO_THROW(telemetry.updateMemoryDeregistered(512));
    EXPECT_NO_THROW(telemetry.addTransactionTime(std::chrono::microseconds(100)));
    EXPECT_NO_THROW(telemetry.addBackendTelemetry("test_event", 100));
}

TEST_F(telemetryTest, ConcurrentAccessWithRuntimeChanges) {
    nixlTelemetry telemetry(test_file);
    EXPECT_TRUE(telemetry.isEnabled());

    const int num_threads = 4;
    const int operations_per_thread = 50;
    std::atomic<bool> stop_threads(false);
    std::atomic<int> size(0);
    std::vector<std::thread> threads;

    // Create threads that perform telemetry operations
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&telemetry, i, &stop_threads, &size]() {
            for (int j = 0; j < operations_per_thread && !stop_threads; ++j) {
                switch (i % 4) {
                case 0:
                    telemetry.updateTxBytes(j * 100);
                    size += 1;
                    break;
                case 1:
                    telemetry.updateRxBytes(j * 50);
                    size += 1;
                    break;
                case 2:
                    telemetry.updateTxRequestsNum(j);
                    size += 1;
                    break;
                case 3:
                    telemetry.updateRxRequestsNum(j);
                    size += 1;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    // Change telemetry state while threads are running
    std::thread state_change_thread([&stop_threads]() {
        for (int i = 0; i < 3 && !stop_threads; ++i) {
            unsetenv(TELEMETRY_ENABLED_VAR);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            setenv(TELEMETRY_ENABLED_VAR, "1", 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    // Wait for all threads to complete
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_EQ(size, operations_per_thread * num_threads);
    size_ = size;
    read_pos_ = 0;
    write_pos_ = size;
    validateState();
    stop_threads = true;
    state_change_thread.join();

    // Verify no crashes occurred and telemetry is in a consistent state
    EXPECT_NO_THROW({
        telemetry.updateTxBytes(1000);
        telemetry.updateRxBytes(2000);
    });
}
