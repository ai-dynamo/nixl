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
#include "mp_collector.h"
#include "mp_store.h"

#include <gtest/gtest.h>

#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <prometheus/metric_type.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace {

using nixl::telemetry::mp::buildMetricFamilies;
using nixl::telemetry::mp::isProcessAlive;
using nixl::telemetry::mp::isSnapshotLive;
using nixl::telemetry::mp::makeStoreFileName;
using nixl::telemetry::mp::mpStoreSnapshot;
using nixl::telemetry::mp::mpStoreWriter;
using nixl::telemetry::mp::nixlMultiprocessCollector;
using nixl::telemetry::mp::readProcessStartTime;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto ERR_BACKEND = nixl_telemetry_event_type_t::AGENT_ERR_BACKEND;

[[nodiscard]] std::size_t
idx(nixl_telemetry_event_type_t t) {
    return static_cast<std::size_t>(t);
}

[[nodiscard]] uint64_t
nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

[[nodiscard]] mpStoreSnapshot
makeSnap(const std::string &agent, const std::string &rank) {
    mpStoreSnapshot s;
    s.pid = ::getpid();
    s.startTime = 1;
    s.lastUpdateNs = nowNs();
    s.agentName = agent;
    s.hostname = "host";
    s.dpRank = rank;
    return s;
}

[[nodiscard]] const prometheus::MetricFamily *
findFamily(const std::vector<prometheus::MetricFamily> &fams, const std::string &name) {
    for (const auto &f : fams) {
        if (f.name == name) {
            return &f;
        }
    }
    return nullptr;
}

[[nodiscard]] const prometheus::ClientMetric *
findByLabel(const prometheus::MetricFamily &fam, const std::string &key, const std::string &value) {
    for (const auto &m : fam.metric) {
        for (const auto &l : m.label) {
            if (l.name == key && l.value == value) {
                return &m;
            }
        }
    }
    return nullptr;
}

[[nodiscard]] bool
hasLabel(const prometheus::ClientMetric &m, const std::string &key) {
    for (const auto &l : m.label) {
        if (l.name == key) {
            return true;
        }
    }
    return false;
}

TEST(MpCollectorTest, EmptySnapshotsYieldNoFamilies) {
    EXPECT_TRUE(buildMetricFamilies({}).empty());
}

TEST(MpCollectorTest, PerProcessCountersAndGauges) {
    auto a = makeSnap("agent-a", "0");
    a.counters[idx(TX_BYTES)] = 1000;
    a.gauges[idx(TX_BYTES)] = 200;
    auto b = makeSnap("agent-b", "1");
    b.counters[idx(TX_BYTES)] = 50;
    b.gauges[idx(TX_BYTES)] = 50;

    const auto fams = buildMetricFamilies({a, b});

    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    EXPECT_EQ(tx->type, prometheus::MetricType::Counter);
    ASSERT_EQ(tx->metric.size(), 2u);
    const auto *tx_a = findByLabel(*tx, "agent_name", "agent-a");
    const auto *tx_b = findByLabel(*tx, "agent_name", "agent-b");
    ASSERT_NE(tx_a, nullptr);
    ASSERT_NE(tx_b, nullptr);
    EXPECT_DOUBLE_EQ(tx_a->counter.value, 1000.0);
    EXPECT_DOUBLE_EQ(tx_b->counter.value, 50.0);

    const auto *gauge = findFamily(fams, "agent_tx_last_bytes");
    ASSERT_NE(gauge, nullptr);
    EXPECT_EQ(gauge->type, prometheus::MetricType::Gauge);
    const auto *g_a = findByLabel(*gauge, "agent_name", "agent-a");
    ASSERT_NE(g_a, nullptr);
    EXPECT_DOUBLE_EQ(g_a->gauge.value, 200.0);
}

TEST(MpCollectorTest, PidLabelDisambiguatesSameAgentName) {
    // Two processes that (mis)use the same agent name and no dp_rank must still
    // produce distinct series, keyed by pid, rather than a duplicate series.
    auto a = makeSnap("dup", "");
    a.pid = 1001;
    a.counters[idx(TX_BYTES)] = 10;
    auto b = makeSnap("dup", "");
    b.pid = 1002;
    b.counters[idx(TX_BYTES)] = 20;

    const auto fams = buildMetricFamilies({a, b});
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 2u);
    const auto *m_a = findByLabel(*tx, "pid", "1001");
    const auto *m_b = findByLabel(*tx, "pid", "1002");
    ASSERT_NE(m_a, nullptr);
    ASSERT_NE(m_b, nullptr);
    EXPECT_DOUBLE_EQ(m_a->counter.value, 10.0);
    EXPECT_DOUBLE_EQ(m_b->counter.value, 20.0);
    EXPECT_TRUE(hasLabel(*m_a, "pid"));
}

TEST(MpCollectorTest, DpRankLabelOnlyWhenPresent) {
    const auto with_rank = makeSnap("agent-a", "3");
    const auto without_rank = makeSnap("agent-b", "");

    const auto fams = buildMetricFamilies({with_rank, without_rank});
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);

    const auto *m_with = findByLabel(*tx, "agent_name", "agent-a");
    const auto *m_without = findByLabel(*tx, "agent_name", "agent-b");
    ASSERT_NE(m_with, nullptr);
    ASSERT_NE(m_without, nullptr);
    EXPECT_TRUE(hasLabel(*m_with, "dp_rank"));
    EXPECT_FALSE(hasLabel(*m_without, "dp_rank"));
}

TEST(MpCollectorTest, ErrorFamilyCarriesStatusLabel) {
    auto a = makeSnap("agent-a", "0");
    a.counters[idx(ERR_BACKEND)] = 5;

    const auto fams = buildMetricFamilies({a});
    const auto *errors = findFamily(fams, "agent_errors_total");
    ASSERT_NE(errors, nullptr);
    EXPECT_EQ(errors->type, prometheus::MetricType::Counter);

    const auto *backend = findByLabel(*errors, "status", "backend");
    ASSERT_NE(backend, nullptr);
    EXPECT_DOUBLE_EQ(backend->counter.value, 5.0);
    const auto *canceled = findByLabel(*errors, "status", "canceled");
    ASSERT_NE(canceled, nullptr);
    EXPECT_DOUBLE_EQ(canceled->counter.value, 0.0);
}

TEST(MpCollectorTest, IsProcessAliveGuardsPidReuse) {
    EXPECT_TRUE(isProcessAlive(::getpid(), readProcessStartTime(::getpid())));
    EXPECT_TRUE(isProcessAlive(::getpid(), 0)); // unknown start time -> existence only
    EXPECT_FALSE(isProcessAlive(0x7fffffff, 0)); // no such process
    // Existing pid but wrong start time -> treated as reused -> not our process.
    EXPECT_FALSE(isProcessAlive(::getpid(), 0xffffffffffffffffULL));
}

TEST(MpCollectorTest, SnapshotLivenessByProcessThenTtl) {
    const auto ttl = std::chrono::seconds(30);

    auto alive = makeSnap("a", "");
    alive.startTime = readProcessStartTime(::getpid());
    alive.lastUpdateNs = 0; // even with an old heartbeat, a live process stays live
    EXPECT_TRUE(isSnapshotLive(alive, ttl));

    auto dead_fresh = makeSnap("b", "");
    dead_fresh.pid = 0x7fffffff;
    dead_fresh.lastUpdateNs = nowNs();
    EXPECT_TRUE(isSnapshotLive(dead_fresh, ttl));

    auto dead_stale = makeSnap("c", "");
    dead_stale.pid = 0x7fffffff;
    dead_stale.lastUpdateNs = nowNs() - std::chrono::nanoseconds(std::chrono::seconds(60)).count();
    EXPECT_FALSE(isSnapshotLive(dead_stale, ttl));
}

class MpCollectorFileTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_collector_" + std::to_string(::getpid()) + "_" + info->name());
        std::filesystem::create_directories(dir_);
    }

    void
    TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    std::filesystem::path dir_;
};

TEST_F(MpCollectorFileTest, CollectReadsLiveStoresAndIgnoresOthers) {
    // Two distinct store files; both headers stamp this (live) process.
    mpStoreWriter w1(dir_ / makeStoreFileName(111, 1, 0), "agent-1", "host", "0");
    w1.addCounter(TX_BYTES, 500);
    w1.setGauge(TX_BYTES, 500);
    mpStoreWriter w2(dir_ / makeStoreFileName(222, 2, 0), "agent-2", "host", "1");
    w2.addCounter(TX_BYTES, 700);

    // A non-store file must be ignored.
    { std::ofstream(dir_ / "unrelated.txt") << "ignore me"; }

    nixlMultiprocessCollector collector(dir_, std::chrono::seconds(30), /*reap_stale=*/false);
    const auto fams = collector.Collect();

    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 2u);
    const auto *m1 = findByLabel(*tx, "agent_name", "agent-1");
    const auto *m2 = findByLabel(*tx, "agent_name", "agent-2");
    ASSERT_NE(m1, nullptr);
    ASSERT_NE(m2, nullptr);
    EXPECT_DOUBLE_EQ(m1->counter.value, 500.0);
    EXPECT_DOUBLE_EQ(m2->counter.value, 700.0);
}

TEST_F(MpCollectorFileTest, CollectOnEmptyDirYieldsNoFamilies) {
    nixlMultiprocessCollector collector(dir_, std::chrono::seconds(30), /*reap_stale=*/false);
    EXPECT_TRUE(collector.Collect().empty());
}

} // namespace
