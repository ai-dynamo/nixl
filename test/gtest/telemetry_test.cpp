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

        env_helper.addVar("NIXL_ENABLE_TELEMETRY", "1");
        env_helper.addVar("NIXL_TELEMETRY_DIR", test_dir.string());
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
        auto buffer =
            std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path.string(), false, 0);
        EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
        EXPECT_EQ(buffer->capacity(), capacity_);
        EXPECT_EQ(buffer->size(), size_);
        EXPECT_EQ(buffer->empty(), size_ == 0);
        EXPECT_EQ(buffer->full(), size_ == capacity_);
    }

    fs::path test_dir;
    std::string test_file;
    gtest::ScopedEnv env_helper;
    size_t capacity_ = DEFAULT_TELEMETRY_BUFFER_SIZE;
    size_t size_ = 0;
    size_t read_pos_ = 0;
    size_t write_pos_ = 0;
    size_t mask_ = DEFAULT_TELEMETRY_BUFFER_SIZE - 1;
};

#ifdef NIXL_ENABLE_TELEMETRY

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

    unsetenv("NIXL_ENABLE_TELEMETRY");
    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_FALSE(telemetry.isEnabled());
        EXPECT_TRUE(fs::is_empty(test_dir));
    });

    // Restore environment variable
    setenv("NIXL_ENABLE_TELEMETRY", "1", 1);
}

TEST_F(telemetryTest, CustomBufferSize) {
    auto tmp_capacity = capacity_;
    capacity_ = 32;
    env_helper.addVar("NIXL_TELEMETRY_BUFFER_SIZE", "32");

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        validateState();
        EXPECT_TRUE(telemetry.isEnabled());
    });
    capacity_ = tmp_capacity;
    env_helper.popVar();
}

TEST_F(telemetryTest, InvalidBufferSize) {
    env_helper.addVar("NIXL_TELEMETRY_BUFFER_SIZE", "0");

    EXPECT_THROW({ nixlTelemetry telemetry(test_file); }, std::invalid_argument);
    env_helper.popVar();
    env_helper.addVar("NIXL_TELEMETRY_BUFFER_SIZE", "1023");
    EXPECT_THROW({ nixlTelemetry telemetry(test_file); }, std::invalid_argument);
    env_helper.popVar();
}

// Test transfer bytes tracking
TEST_F(telemetryTest, TransferBytesTracking) {
    env_helper.addVar("NIXL_TELEMETRY_RUN_INTERVAL", "1");
    nixlTelemetry telemetry(test_file);

    EXPECT_NO_THROW(telemetry.updateTxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateRxBytes(1024));
    EXPECT_NO_THROW(telemetry.updateTxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateRxRequestsNum(1));
    EXPECT_NO_THROW(telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND));
    EXPECT_NO_THROW(telemetry.updateMemoryRegistered(1024));
    EXPECT_NO_THROW(telemetry.updateMemoryDeregistered(1024));
    EXPECT_NO_THROW(telemetry.addTransactionTime(std::chrono::microseconds(100)));
    EXPECT_NO_THROW(telemetry.addGeneralTelemetry("test_event", 100));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto path = fs::path(test_dir) / test_file;
    auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(path.string(), false);
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

    EXPECT_NO_THROW(telemetry.addGeneralTelemetry("", 100));
}

TEST_F(telemetryTest, ShortRunInterval) {
    env_helper.addVar("NIXL_TELEMETRY_RUN_INTERVAL", "1");

    std::string test_file = "test_short_interval";

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());
    });
    env_helper.popVar();
}

TEST_F(telemetryTest, LargeRunInterval) {
    env_helper.addVar("NIXL_TELEMETRY_RUN_INTERVAL", "10000");

    std::string test_file = "test_large_interval";

    EXPECT_NO_THROW({
        nixlTelemetry telemetry(test_file);
        EXPECT_TRUE(telemetry.isEnabled());
    });
    env_helper.popVar();
}

TEST_F(telemetryTest, BufferOverflowHandling) {
    env_helper.addVar("NIXL_TELEMETRY_BUFFER_SIZE", "4");

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
    env_helper.addVar("NIXL_TELEMETRY_DIR", custom_dir.string());

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
    for (int i = 0; i < static_cast<int>(nixl_telemetry_category_t::NIXL_TELEMETRY_MAX); ++i) {
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
    env_helper.addVar("NIXL_TELEMETRY_RUN_INTERVAL", "1");
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

#else
TEST_F(telemetryTest, TelemetryDisabledAtCompileTime) {
    std::string test_file = "test_compile_time_disabled";
    nixlTelemetry telemetry(test_file);
    EXPECT_FALSE(telemetry.isEnabled());
    EXPECT_TRUE(fs::is_empty(test_dir));
}
#endif
