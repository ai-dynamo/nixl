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
#include "mp_store.h"
#include "mp_telemetry_fixture.h"

#include "common.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using nixl::telemetry::mp::storeWriter;
using nixl::telemetry::mp::storeWriterAlive;
using nixl::telemetry::mp::processRunMarker;
using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;
constexpr auto ERR_BACKEND = nixl_telemetry_event_type_t::AGENT_ERR_BACKEND;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

const std::vector<double> kBuckets = {10, 100, 1000};

class MpStoreTest : public mpTempDirTest {
protected:
    [[nodiscard]] std::filesystem::path
    storePath(const std::string &name) const {
        return dir_ / name;
    }
};

TEST_F(MpStoreTest, WriteReadRoundTrip) {
    const auto path = storePath("agent-a");
    storeWriter writer(path, "agent-a", "host-1", "3", 7, kBuckets);
    writer.addCounter(TX_BYTES, 1000);
    writer.setGauge(TX_BYTES, 1000);
    writer.addCounter(ERR_BACKEND, 1);

    const auto res = readStoreSnapshot(path);
    const auto &snap = res.snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_FALSE(res.contentInvalid);
    EXPECT_EQ(snap->agentName, "agent-a");
    EXPECT_EQ(snap->hostname, "host-1");
    EXPECT_EQ(snap->localRank, "3");
    EXPECT_EQ(snap->instance, 7u);
    EXPECT_EQ(snap->pid, static_cast<int64_t>(::getpid()));
    EXPECT_GT(snap->lastUpdateNs, 0u);
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->counters[idx(ERR_BACKEND)], 1u);
    // Untouched slots stay zero.
    EXPECT_EQ(snap->counters[idx(RX_BYTES)], 0u);
    EXPECT_EQ(snap->gauges[idx(RX_BYTES)], 0u);
}

TEST_F(MpStoreTest, HeartbeatAdvancesOnlyWhenRefreshed) {
    const auto path = storePath("agent-heartbeat");
    storeWriter writer(path, "agent-heartbeat", "host-1", "", 0, kBuckets);

    const auto created = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(created.has_value());
    ASSERT_GT(created->lastUpdateNs, 0u);

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    writer.addCounter(TX_BYTES, 1);
    writer.setGauge(TX_BYTES, 1);
    writer.observeHistogram(XFER_TIME, 1);

    const auto after_updates = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(after_updates.has_value());
    EXPECT_EQ(after_updates->lastUpdateNs, created->lastUpdateNs);

    const uint64_t refreshed = writer.refreshHeartbeat();

    const auto after_refresh = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(after_refresh.has_value());
    EXPECT_GT(after_refresh->lastUpdateNs, created->lastUpdateNs);
    EXPECT_EQ(after_refresh->lastUpdateNs, refreshed);
}

TEST_F(MpStoreTest, CounterAccumulatesGaugeReplaces) {
    const auto path = storePath("agent-b");
    storeWriter writer(path, "agent-b", "host-1", "", 0, kBuckets);
    writer.addCounter(TX_BYTES, 100);
    writer.addCounter(TX_BYTES, 250);
    writer.addCounter(TX_BYTES, 650);
    writer.setGauge(TX_BYTES, 100);
    writer.setGauge(TX_BYTES, 650); // last write wins

    const auto snap = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 650u);
}

TEST_F(MpStoreTest, EmptyRankIsEmpty) {
    const auto path = storePath("agent-c");
    storeWriter writer(path, "agent-c", "host-1", "", 0, kBuckets);

    const auto snap = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_TRUE(snap->localRank.empty());
}

TEST_F(MpStoreTest, HistogramBucketBoundsAreInclusive) {
    const auto path = storePath("agent-hist");
    storeWriter writer(path, "agent-hist", "host-1", "", 0, kBuckets);
    // Bounds {10, 100, 1000}: a value equal to a bound belongs to that bucket,
    // and anything above the last bound lands in the trailing overflow slot.
    writer.observeHistogram(XFER_TIME, 1);
    writer.observeHistogram(XFER_TIME, 10);
    writer.observeHistogram(XFER_TIME, 11);
    writer.observeHistogram(XFER_TIME, 1000);
    writer.observeHistogram(XFER_TIME, 1001);

    const auto snap = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->bucketCount, kBuckets.size());
    EXPECT_DOUBLE_EQ(snap->bucketBounds[0], 10.0);
    EXPECT_DOUBLE_EQ(snap->bucketBounds[2], 1000.0);

    const auto &counts = snap->histBuckets[idx(XFER_TIME)];
    EXPECT_EQ(counts[0], 2u); // 1, 10
    EXPECT_EQ(counts[1], 1u); // 11
    EXPECT_EQ(counts[2], 1u); // 1000
    EXPECT_EQ(counts[3], 1u); // 1001 -> overflow
    EXPECT_EQ(snap->histSums[idx(XFER_TIME)], 1u + 10u + 11u + 1000u + 1001u);
}

TEST_F(MpStoreTest, HistogramUntouchedTypesStayEmpty) {
    const auto path = storePath("agent-hist-empty");
    storeWriter writer(path, "agent-hist-empty", "host-1", "", 0, kBuckets);
    writer.observeHistogram(XFER_TIME, 5);

    const auto snap = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->histSums[idx(TX_BYTES)], 0u);
    for (const auto count : snap->histBuckets[idx(TX_BYTES)]) {
        EXPECT_EQ(count, 0u);
    }
}

TEST_F(MpStoreTest, TooManyBucketsRejected) {
    using nixl::telemetry::mp::MP_STORE_MAX_BUCKETS;
    std::vector<double> too_many(MP_STORE_MAX_BUCKETS + 1);
    for (std::size_t i = 0; i < too_many.size(); ++i) {
        too_many[i] = static_cast<double>(i + 1);
    }
    const auto path = storePath("agent-too-many");
    EXPECT_THROW({ storeWriter writer(path, "a", "host-1", "", 0, too_many); }, std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(MpStoreTest, DestructorKeepsStoreFileWithFinalValues) {
    const auto path = storePath("agent-cleanup");
    {
        storeWriter writer(path, "agent-cleanup", "host-1", "", 0, kBuckets);
        writer.addCounter(TX_BYTES, 4096);
        (void)writer.refreshHeartbeat();
        EXPECT_TRUE(std::filesystem::exists(path));
    }
    ASSERT_TRUE(std::filesystem::exists(path)) << "a clean exit must not drop unscraped values";

    const auto result = readStoreSnapshot(path);
    ASSERT_TRUE(result.snapshot.has_value());
    EXPECT_EQ(result.snapshot->counters[idx(TX_BYTES)], 4096u);
}

TEST_F(MpStoreTest, LiveWriterHoldsItsStoreAndReleasesItOnDestruction) {
    const auto path = storePath("agent-lock");
    {
        const storeWriter writer(path, "agent-lock", "host-1", "", 0, kBuckets);
        // The probe runs in the writer's own process, which is the exporter's case:
        // it owns a store and collects every store in the directory, its own
        // included, so the lock has to be visible to it too.
        EXPECT_TRUE(storeWriterAlive(path));
    }
    EXPECT_FALSE(storeWriterAlive(path));
}

TEST_F(MpStoreTest, StoreNobodyEverWroteReadsAsAbandoned) {
    const auto path = storePath("planted");
    { std::ofstream(path) << "not a store"; }
    EXPECT_FALSE(storeWriterAlive(path));
}

TEST_F(MpStoreTest, UnprobableStoreReadsAsHeld) {
    // Nothing to lock is not evidence that a writer is gone: a probe that could
    // not run must never license a reap.
    EXPECT_TRUE(storeWriterAlive(storePath("does-not-exist")));
}

TEST_F(MpStoreTest, CreationPublishesTheStoreAndLeavesNothingBehind) {
    const auto path = storePath("agent-atomic");
    const storeWriter writer(path, "agent-atomic", "host-1", "", 0, kBuckets);

    // A store becomes visible only once it is initialized and locked, so the
    // directory holds exactly the published file -- never a staging one, and never
    // a store with an unwritten header.
    std::vector<std::string> names;
    for (const auto &entry : std::filesystem::directory_iterator(dir_)) {
        names.push_back(entry.path().filename().string());
    }
    ASSERT_EQ(names, std::vector<std::string>{path.filename().string()});
    EXPECT_TRUE(readStoreSnapshot(path).snapshot.has_value());
}

TEST_F(MpStoreTest, LongAgentNameTruncated) {
    const auto path = storePath("agent-long");
    const std::string long_name(1000, 'x');
    const gtest::LogIgnoreGuard lig("exceeds 255 chars");
    storeWriter writer(path, long_name, "host-1", "", 0, kBuckets);

    const auto snap = readStoreSnapshot(path).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName.size(), 255u);
    EXPECT_EQ(snap->agentName, long_name.substr(0, 255));
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpStoreTest, ForeignOwnedStoreIsIgnoredAndNotReapable) {
    if (::geteuid() != 0) {
        GTEST_SKIP() << "needs privileges to give a store file another owner";
    }
    // A name no run of this process has used: the warning is emitted once per
    // path, so a repeated run would otherwise see none.
    static int run = 0;
    const auto path = storePath("agent-foreign-" + std::to_string(++run));
    storeWriter writer(path, "agent-foreign", "host-1", "", 0, kBuckets);
    writer.addCounter(TX_BYTES, 7);
    constexpr uid_t nobody = 65534;
    if (::chown(path.c_str(), nobody, static_cast<gid_t>(-1)) != 0) {
        GTEST_SKIP() << "cannot give a store file another owner: " << strerror(errno);
    }

    const gtest::LogIgnoreGuard lig("owned by uid");
    const auto res = readStoreSnapshot(path);
    EXPECT_FALSE(res.snapshot.has_value());
    EXPECT_FALSE(res.contentInvalid);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpStoreTest, MissingFileReturnsNulloptNotContentInvalid) {
    // A file we cannot open (missing here, or a transient error) must NOT be
    // reported as invalid content -- otherwise the collector could reap a live
    // peer it merely failed to read.
    const auto res = readStoreSnapshot(storePath("does-not-exist"));
    EXPECT_FALSE(res.snapshot.has_value());
    EXPECT_FALSE(res.contentInvalid);
}

TEST_F(MpStoreTest, TooSmallFileReturnsNullopt) {
    const auto path = storePath("tiny");
    {
        std::ofstream f(path, std::ios::binary);
        const char junk[16] = {0};
        f.write(junk, sizeof(junk));
    }
    const auto res = readStoreSnapshot(path);
    EXPECT_FALSE(res.snapshot.has_value());
    EXPECT_TRUE(res.contentInvalid);
}

TEST_F(MpStoreTest, ZeroMagicReturnsNulloptQuietly) {
    const auto path = storePath("zero-magic");
    {
        // Large enough to pass the size check, but all-zero: a file somebody left
        // in the directory. Must be skipped WITHOUT a warning (no LogIgnoreGuard).
        std::ofstream f(path, std::ios::binary);
        const std::string zeros(64 * 1024, '\0');
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    const auto res = readStoreSnapshot(path);
    EXPECT_FALSE(res.snapshot.has_value());
    EXPECT_TRUE(res.contentInvalid);
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
    const auto res = readStoreSnapshot(path);
    EXPECT_FALSE(res.snapshot.has_value());
    EXPECT_TRUE(res.contentInvalid);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpStoreTest, RunMarkerIsNonZeroAndFixedForThisProcess) {
    const uint64_t marker = processRunMarker();
    EXPECT_GT(marker, 0u);
    EXPECT_EQ(processRunMarker(), marker);
}

} // namespace
