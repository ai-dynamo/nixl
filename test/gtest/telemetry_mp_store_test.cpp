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
#include "mp_store.h"

#include "common.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using nixl::telemetry::mp::mpStoreWriter;
using nixl::telemetry::mp::readProcessStartTime;
using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;
constexpr auto ERR_BACKEND = nixl_telemetry_event_type_t::AGENT_ERR_BACKEND;

[[nodiscard]] std::size_t
idx(nixl_telemetry_event_type_t t) {
    return static_cast<std::size_t>(t);
}

class MpStoreTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_store_" + std::to_string(::getpid()) + "_" + info->name());
        std::filesystem::create_directories(dir_);
    }

    void
    TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] std::filesystem::path
    storePath(const std::string &name) const {
        return dir_ / name;
    }

    std::filesystem::path dir_;
};

TEST_F(MpStoreTest, WriteReadRoundTrip) {
    const auto path = storePath("agent-a");
    mpStoreWriter writer(path, "agent-a", "host-1", "3", 7);
    writer.addCounter(TX_BYTES, 1000);
    writer.setGauge(TX_BYTES, 1000);
    writer.addCounter(ERR_BACKEND, 1);

    const auto snap = readStoreSnapshot(path);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-a");
    EXPECT_EQ(snap->hostname, "host-1");
    EXPECT_EQ(snap->localRank, "3");
    EXPECT_EQ(snap->instance, 7u);
    EXPECT_EQ(snap->pid, static_cast<int64_t>(::getpid()));
    EXPECT_GT(snap->startTime, 0u);
    EXPECT_GT(snap->lastUpdateNs, 0u);
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->counters[idx(ERR_BACKEND)], 1u);
    // Untouched slots stay zero.
    EXPECT_EQ(snap->counters[idx(RX_BYTES)], 0u);
    EXPECT_EQ(snap->gauges[idx(RX_BYTES)], 0u);
}

TEST_F(MpStoreTest, CounterAccumulatesGaugeReplaces) {
    const auto path = storePath("agent-b");
    mpStoreWriter writer(path, "agent-b", "host-1", "", 0);
    writer.addCounter(TX_BYTES, 100);
    writer.addCounter(TX_BYTES, 250);
    writer.addCounter(TX_BYTES, 650);
    writer.setGauge(TX_BYTES, 100);
    writer.setGauge(TX_BYTES, 650); // last write wins

    const auto snap = readStoreSnapshot(path);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 650u);
}

TEST_F(MpStoreTest, EmptyRankIsEmpty) {
    const auto path = storePath("agent-c");
    mpStoreWriter writer(path, "agent-c", "host-1", "", 0);

    const auto snap = readStoreSnapshot(path);
    ASSERT_TRUE(snap.has_value());
    EXPECT_TRUE(snap->localRank.empty());
}

TEST_F(MpStoreTest, DestructorRemovesStoreFile) {
    const auto path = storePath("agent-cleanup");
    {
        mpStoreWriter writer(path, "agent-cleanup", "host-1", "", 0);
        EXPECT_TRUE(std::filesystem::exists(path));
    }
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(MpStoreTest, LongAgentNameTruncated) {
    const auto path = storePath("agent-long");
    const std::string long_name(1000, 'x');
    const gtest::LogIgnoreGuard lig("exceeds 255 chars");
    mpStoreWriter writer(path, long_name, "host-1", "", 0);

    const auto snap = readStoreSnapshot(path);
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName.size(), 255u);
    EXPECT_EQ(snap->agentName, long_name.substr(0, 255));
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpStoreTest, MissingFileReturnsNullopt) {
    EXPECT_FALSE(readStoreSnapshot(storePath("does-not-exist")).has_value());
}

TEST_F(MpStoreTest, TooSmallFileReturnsNullopt) {
    const auto path = storePath("tiny");
    {
        std::ofstream f(path, std::ios::binary);
        const char junk[16] = {0};
        f.write(junk, sizeof(junk));
    }
    EXPECT_FALSE(readStoreSnapshot(path).has_value());
}

TEST_F(MpStoreTest, ZeroMagicReturnsNulloptQuietly) {
    const auto path = storePath("zero-magic");
    {
        // Large enough to pass the size check, but all-zero: a store mid-creation
        // or an orphan. Must be skipped WITHOUT a warning (no LogIgnoreGuard).
        std::ofstream f(path, std::ios::binary);
        const std::string zeros(64 * 1024, '\0');
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    EXPECT_FALSE(readStoreSnapshot(path).has_value());
}

TEST_F(MpStoreTest, BadMagicWarnsAndReturnsNullopt) {
    const auto path = storePath("bad-magic");
    {
        std::ofstream f(path, std::ios::binary);
        const uint64_t bad_magic = 0xDEADBEEFULL; // non-zero, not our magic
        f.write(reinterpret_cast<const char *>(&bad_magic), sizeof(bad_magic));
        const std::string zeros(64 * 1024, '\0');
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    const gtest::LogIgnoreGuard lig("bad magic");
    EXPECT_FALSE(readStoreSnapshot(path).has_value());
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpStoreTest, ProcessStartTimeSelfNonZeroBogusZero) {
    EXPECT_GT(readProcessStartTime(::getpid()), 0u);
    // A pid that will not exist -> 0.
    EXPECT_EQ(readProcessStartTime(0x7fffffff), 0u);
}

} // namespace
