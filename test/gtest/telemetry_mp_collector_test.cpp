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
#include "mp_collector.h"
#include "mp_store.h"
#include "mp_telemetry_fixture.h"

#include "common.h"

#include <gtest/gtest.h>

#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <prometheus/metric_type.h>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using nixl::telemetry::mp::buildMetricFamilies;
using nixl::telemetry::mp::isSnapshotLive;
using nixl::telemetry::mp::makeStoreFileName;
using nixl::telemetry::mp::monotonicNs;
using nixl::telemetry::mp::scannedStore;
using nixl::telemetry::mp::storeSnapshot;
using nixl::telemetry::mp::storeWriter;
using nixl::telemetry::mp::nixlMultiprocessCollector;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto ERR_BACKEND = nixl_telemetry_event_type_t::AGENT_ERR_BACKEND;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

const std::vector<double> kBuckets = {10, 100, 1000};

[[nodiscard]] storeSnapshot
makeSnap(const std::string &agent, const std::string &rank, int64_t pid = ::getpid()) {
    storeSnapshot s;
    s.pid = pid;
    s.lastUpdateNs = monotonicNs();
    s.agentName = agent;
    s.hostname = "host";
    s.localRank = rank;
    s.bucketCount = static_cast<uint32_t>(kBuckets.size());
    std::copy(kBuckets.begin(), kBuckets.end(), s.bucketBounds.begin());
    return s;
}

[[nodiscard]] scannedStore
live(const storeSnapshot &s) {
    return {s, true};
}

[[nodiscard]] scannedStore
departed(const storeSnapshot &s) {
    return {s, false};
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

[[nodiscard]] std::map<std::string, std::string>
labelMap(const prometheus::ClientMetric &m) {
    std::map<std::string, std::string> labels;
    for (const auto &l : m.label) {
        labels.emplace(l.name, l.value);
    }
    return labels;
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

    const auto fams = buildMetricFamilies({{a}, {b}});

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
    // Two processes that (mis)use the same agent name and no local_rank must still
    // produce distinct series, keyed by pid, rather than a duplicate series.
    auto a = makeSnap("dup", "");
    a.pid = 1001;
    a.counters[idx(TX_BYTES)] = 10;
    auto b = makeSnap("dup", "");
    b.pid = 1002;
    b.counters[idx(TX_BYTES)] = 20;

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup';");
    const auto fams = buildMetricFamilies(collector.selectStores({live(a), live(b)}));
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
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
    EXPECT_TRUE(hasLabel(*m_a, "agent_instance"));
}

TEST(MpCollectorTest, AgentInstanceLabelDisambiguatesSameProcessSameName) {
    // Two agents in the SAME process (same pid) with the same name must still
    // produce distinct series, keyed by agent_instance, rather than colliding.
    auto a = makeSnap("dup", "");
    a.pid = 1001;
    a.instance = 0;
    a.counters[idx(TX_BYTES)] = 10;
    auto b = makeSnap("dup", "");
    b.pid = 1001;
    b.instance = 1;
    b.counters[idx(TX_BYTES)] = 20;

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup';");
    const auto fams = buildMetricFamilies(collector.selectStores({live(a), live(b)}));
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 2u);
    const auto *m_a = findByLabel(*tx, "agent_instance", "0");
    const auto *m_b = findByLabel(*tx, "agent_instance", "1");
    ASSERT_NE(m_a, nullptr);
    ASSERT_NE(m_b, nullptr);
    EXPECT_DOUBLE_EQ(m_a->counter.value, 10.0);
    EXPECT_DOUBLE_EQ(m_b->counter.value, 20.0);
}

TEST(MpCollectorTest, DuplicateLiveNameIsReportedOncePerCollector) {
    std::vector<scannedStore> stores;
    for (const std::string name : {"dup-a", "dup-b"}) {
        for (const int64_t pid : {1001, 1002}) {
            stores.push_back(live(makeSnap(name, "", pid)));
        }
    }

    const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup-[ab]';");
    const nixlMultiprocessCollector collector{std::filesystem::path()};
    EXPECT_EQ(collector.selectStores(stores).size(), 4u);
    EXPECT_EQ(collector.selectStores(stores).size(), 4u);
    EXPECT_EQ(lig.getIgnoredCount(), 1u);

    const nixlMultiprocessCollector successor{std::filesystem::path()};
    EXPECT_EQ(successor.selectStores(stores).size(), 4u);
    EXPECT_EQ(lig.getIgnoredCount(), 2u);
}

TEST(MpCollectorTest, ConcurrentScrapesReportADuplicateLiveNameOnce) {
    const std::vector stores{live(makeSnap("dup", "", 1001)), live(makeSnap("dup", "", 1002))};

    const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup';");
    const nixlMultiprocessCollector collector{std::filesystem::path()};
    {
        std::vector<std::jthread> scrapers;
        for (int i = 0; i < 8; ++i) {
            scrapers.emplace_back([&collector, &stores] {
                for (int scrape = 0; scrape < 100; ++scrape) {
                    EXPECT_EQ(collector.selectStores(stores).size(), 2u);
                }
            });
        }
    }
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
}

TEST(MpCollectorTest, DuplicateLiveNameDropsDepartedStoresOfThatName) {
    auto gone = makeSnap("dup", "3", 1000);
    auto a = makeSnap("dup", "3", 1001);
    auto b = makeSnap("dup", "3", 1002);

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    const gtest::LogIgnoreGuard lig(
        "2 live agents on host 'host' are named 'dup' with local_rank '3';");
    const auto fams =
        buildMetricFamilies(collector.selectStores({departed(gone), live(a), live(b)}));
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 2u);
    EXPECT_NE(findByLabel(*tx, "pid", "1001"), nullptr);
    EXPECT_NE(findByLabel(*tx, "pid", "1002"), nullptr);
}

TEST(MpCollectorTest, SingleStoreCarriesOnlyItsIdentityLabels) {
    const nixlMultiprocessCollector collector{std::filesystem::path()};
    for (const std::string rank : {"", "3"}) {
        SCOPED_TRACE("local_rank '" + rank + "'");
        std::map<std::string, std::string> expected{{"hostname", "host"},
                                                    {"agent_name", "agent-a"}};
        if (!rank.empty()) {
            expected.emplace("local_rank", rank);
        }

        const auto fams =
            buildMetricFamilies(collector.selectStores({live(makeSnap("agent-a", rank))}));
        for (const std::string name : {"agent_tx_bytes_total",
                                       "agent_tx_last_bytes",
                                       "agent_xfer_time_us",
                                       "agent_errors_total"}) {
            EXPECT_NE(findFamily(fams, name), nullptr) << name;
        }
        for (const auto &family : fams) {
            for (const auto &metric : family.metric) {
                auto labels = labelMap(metric);
                EXPECT_EQ(labels.size(), metric.label.size()) << family.name;
                if (family.name == "agent_errors_total") {
                    EXPECT_EQ(labels.erase("status"), 1u) << family.name;
                }
                EXPECT_EQ(labels, expected) << family.name;
            }
        }
    }
}

TEST(MpCollectorTest, LiveStoresWithDistinctIdentitiesCarryNoProcessLabels) {
    auto a = makeSnap("agent-a", "", 1001);
    auto b = makeSnap("agent-b", "", 1002);
    auto ranked = makeSnap("agent-a", "1", 1003);
    auto elsewhere = makeSnap("agent-a", "", 1004);
    elsewhere.hostname = "other-host";
    auto dup_first = makeSnap("dup", "", 1005);
    auto dup_second = makeSnap("dup", "", 1006);

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup';");
    const auto fams = buildMetricFamilies(collector.selectStores(
        {live(a), live(b), live(ranked), live(elsewhere), live(dup_first), live(dup_second)}));
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 6u);
    for (const auto &m : tx->metric) {
        const bool duplicate = labelMap(m).at("agent_name") == "dup";
        EXPECT_EQ(hasLabel(m, "pid"), duplicate);
        EXPECT_EQ(hasLabel(m, "agent_instance"), duplicate);
    }
}

TEST(MpCollectorTest, LiveStoreSupersedesAMoreRecentDepartedOneOfTheSameName) {
    auto gone = makeSnap("agent-a", "", 1001);
    gone.lastUpdateNs = 200;
    gone.counters[idx(TX_BYTES)] = 10;
    auto successor = makeSnap("agent-a", "", 1000);
    successor.lastUpdateNs = 100;
    successor.counters[idx(TX_BYTES)] = 20;

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    const auto fams =
        buildMetricFamilies(collector.selectStores({departed(gone), live(successor)}));
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 1u);
    EXPECT_DOUBLE_EQ(tx->metric.front().counter.value, 20.0);
    EXPECT_FALSE(hasLabel(tx->metric.front(), "pid"));
    EXPECT_FALSE(hasLabel(tx->metric.front(), "agent_instance"));
}

TEST(MpCollectorTest, OnlyTheMostRecentDepartedStoreOfANameIsPublished) {
    auto older = makeSnap("agent-a", "", 1002);
    older.lastUpdateNs = 100;
    older.counters[idx(TX_BYTES)] = 10;
    auto newer = makeSnap("agent-a", "", 1001);
    newer.lastUpdateNs = 200;
    newer.counters[idx(TX_BYTES)] = 20;

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    for (const auto &stores : {std::vector{departed(older), departed(newer)},
                               std::vector{departed(newer), departed(older)}}) {
        const auto fams = buildMetricFamilies(collector.selectStores(stores));
        const auto *tx = findFamily(fams, "agent_tx_bytes_total");
        ASSERT_NE(tx, nullptr);
        ASSERT_EQ(tx->metric.size(), 1u);
        EXPECT_DOUBLE_EQ(tx->metric.front().counter.value, 20.0);
        EXPECT_FALSE(hasLabel(tx->metric.front(), "pid"));
        EXPECT_FALSE(hasLabel(tx->metric.front(), "agent_instance"));
    }
}

TEST(MpCollectorTest, TiedDepartedStoresResolveTheSameInEveryScanOrder) {
    auto a = makeSnap("agent-a", "", 1001);
    a.instance = 2;
    a.lastUpdateNs = 100;
    a.counters[idx(TX_BYTES)] = 10;
    auto b = a;
    b.pid = 1002;
    b.instance = 0;
    b.counters[idx(TX_BYTES)] = 20;
    auto c = b;
    c.instance = 1;
    c.counters[idx(TX_BYTES)] = 30;
    const std::array tied{departed(a), departed(b), departed(c)};

    const nixlMultiprocessCollector collector{std::filesystem::path()};
    std::array<std::size_t, 3> order{0, 1, 2};
    do {
        const auto chosen =
            collector.selectStores({tied[order[0]], tied[order[1]], tied[order[2]]});
        ASSERT_EQ(chosen.size(), 1u);
        EXPECT_FALSE(chosen.front().withProcessLabels);
        EXPECT_EQ(chosen.front().snapshot.counters[idx(TX_BYTES)], 30u);
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST(MpCollectorTest, LocalRankLabelOnlyWhenPresent) {
    const auto with_rank = makeSnap("agent-a", "3");
    const auto without_rank = makeSnap("agent-b", "");

    const auto fams = buildMetricFamilies({{with_rank}, {without_rank}});
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);

    const auto *m_with = findByLabel(*tx, "agent_name", "agent-a");
    const auto *m_without = findByLabel(*tx, "agent_name", "agent-b");
    ASSERT_NE(m_with, nullptr);
    ASSERT_NE(m_without, nullptr);
    EXPECT_TRUE(hasLabel(*m_with, "local_rank"));
    EXPECT_FALSE(hasLabel(*m_without, "local_rank"));
}

TEST(MpCollectorTest, ErrorFamilyCarriesStatusLabel) {
    auto a = makeSnap("agent-a", "0");
    a.counters[idx(ERR_BACKEND)] = 5;

    const auto fams = buildMetricFamilies({{a}});
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

TEST(MpCollectorTest, HistogramFamilyIsCumulativeAndEndsAtInfinity) {
    auto a = makeSnap("agent-a", "0");
    // Per-bucket (non-cumulative) counts for bounds {10, 100, 1000} plus overflow.
    a.histBuckets[idx(XFER_TIME)] = {1, 2, 3, 4};
    a.histSums[idx(XFER_TIME)] = 9999;

    const auto fams = buildMetricFamilies({{a}});
    const auto *hist = findFamily(fams, "agent_xfer_time_us");
    ASSERT_NE(hist, nullptr);
    EXPECT_EQ(hist->type, prometheus::MetricType::Histogram);
    ASSERT_EQ(hist->metric.size(), 1u);

    const auto &buckets = hist->metric[0].histogram.bucket;
    ASSERT_EQ(buckets.size(), kBuckets.size() + 1);
    EXPECT_DOUBLE_EQ(buckets[0].upper_bound, 10.0);
    EXPECT_EQ(buckets[0].cumulative_count, 1u);
    EXPECT_DOUBLE_EQ(buckets[1].upper_bound, 100.0);
    EXPECT_EQ(buckets[1].cumulative_count, 3u);
    EXPECT_DOUBLE_EQ(buckets[2].upper_bound, 1000.0);
    EXPECT_EQ(buckets[2].cumulative_count, 6u);
    EXPECT_TRUE(std::isinf(buckets[3].upper_bound));
    EXPECT_EQ(buckets[3].cumulative_count, 10u);

    EXPECT_EQ(hist->metric[0].histogram.sample_count, 10u);
    EXPECT_DOUBLE_EQ(hist->metric[0].histogram.sample_sum, 9999.0);
}

TEST(MpCollectorTest, HistogramsAreNotAggregatedAcrossProcesses) {
    auto a = makeSnap("agent-a", "0");
    a.pid = 1001;
    a.histBuckets[idx(XFER_TIME)] = {1, 0, 0, 0};
    auto b = makeSnap("agent-b", "1");
    b.pid = 1002;
    b.histBuckets[idx(XFER_TIME)] = {0, 0, 0, 5};

    const auto fams = buildMetricFamilies({{a}, {b}});
    const auto *hist = findFamily(fams, "agent_xfer_time_us");
    ASSERT_NE(hist, nullptr);
    ASSERT_EQ(hist->metric.size(), 2u);

    const auto *m_a = findByLabel(*hist, "agent_name", "agent-a");
    const auto *m_b = findByLabel(*hist, "agent_name", "agent-b");
    ASSERT_NE(m_a, nullptr);
    ASSERT_NE(m_b, nullptr);
    EXPECT_EQ(m_a->histogram.sample_count, 1u);
    EXPECT_EQ(m_b->histogram.sample_count, 5u);
}

TEST(MpCollectorTest, SnapshotLivenessByWriterThenTtl) {
    const auto ttl = std::chrono::seconds(30);

    auto held = makeSnap("a", "");
    held.lastUpdateNs = 0; // a writer that still holds its store stays live
    EXPECT_TRUE(isSnapshotLive(held, ttl, /*writer_alive=*/true));

    auto gone_fresh = makeSnap("b", "");
    gone_fresh.lastUpdateNs = monotonicNs();
    EXPECT_TRUE(isSnapshotLive(gone_fresh, ttl, /*writer_alive=*/false));

    // A zero heartbeat is as old as this boot, and a zero TTL makes that stale on
    // any host. Subtracting a fixed age from monotonicNs() instead would wrap on
    // a host that booted seconds ago, and then pass as a future timestamp rather
    // than as an expired one.
    auto gone_stale = makeSnap("c", "");
    gone_stale.lastUpdateNs = 0;
    EXPECT_FALSE(isSnapshotLive(gone_stale, std::chrono::seconds(0), /*writer_alive=*/false));
}

class MpCollectorFileTest : public mpTempDirTest {};

TEST_F(MpCollectorFileTest, CollectReadsLiveStoresAndIgnoresOthers) {
    // Two distinct store files; both headers stamp this (live) process.
    storeWriter w1(dir_ / makeStoreFileName(111, 1, 0), "agent-1", "host", "0", 0, kBuckets);
    w1.addCounter(TX_BYTES, 500);
    w1.setGauge(TX_BYTES, 500);
    storeWriter w2(dir_ / makeStoreFileName(222, 2, 0), "agent-2", "host", "1", 0, kBuckets);
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

TEST_F(MpCollectorFileTest, ReapsUnparsableFilesNobodyHoldsAndKeepsHeldOnes) {
    const auto writeZeroFile = [](const std::filesystem::path &p) {
        std::ofstream f(p, std::ios::binary);
        const std::string zeros(64 * 1024, '\0');
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    };

    // Unlocked: nothing will ever write it again, so age does not enter into it.
    const auto orphan = dir_ / makeStoreFileName(999, 1, 0);
    writeZeroFile(orphan);

    // Locked by somebody, however unparsable: not ours to remove.
    const auto held = dir_ / makeStoreFileName(998, 1, 0);
    writeZeroFile(held);
    const nixl::scopedFd held_fd(::open(held.c_str(), O_RDONLY | O_CLOEXEC));
    ASSERT_TRUE(held_fd.valid());
    ASSERT_EQ(::flock(held_fd.get(), LOCK_EX | LOCK_NB), 0);

    nixlMultiprocessCollector collector(dir_, std::chrono::seconds(0), /*reap_stale=*/true);
    const auto fams = collector.Collect();

    EXPECT_TRUE(fams.empty());
    EXPECT_FALSE(std::filesystem::exists(orphan));
    EXPECT_TRUE(std::filesystem::exists(held));
}

TEST_F(MpCollectorFileTest, LiveWriterOutlivesAZeroTtl) {
    // The heartbeat says nothing about a writer that is simply idle, so a store
    // its writer still holds must survive even a TTL that expires everything.
    storeWriter writer(dir_ / makeStoreFileName(444, 1, 0), "agent-idle", "host", "", 0, kBuckets);
    writer.addCounter(TX_BYTES, 7);

    nixlMultiprocessCollector collector(dir_, std::chrono::seconds(0), /*reap_stale=*/true);
    const auto fams = collector.Collect();

    ASSERT_NE(findFamily(fams, "agent_tx_bytes_total"), nullptr);
    EXPECT_TRUE(std::filesystem::exists(writer.path()));
}

TEST_F(MpCollectorFileTest, DepartedWriterIsPublishedOnceMoreThenReaped) {
    const auto path = dir_ / makeStoreFileName(333, 1, 0);
    {
        storeWriter writer(path, "agent-gone", "host", "", 0, kBuckets);
        writer.addCounter(TX_BYTES, 42);
    }

    nixlMultiprocessCollector within_ttl(dir_, std::chrono::seconds(30), /*reap_stale=*/true);
    const auto fams = within_ttl.Collect();
    const auto *tx = findFamily(fams, "agent_tx_bytes_total");
    ASSERT_NE(tx, nullptr);
    ASSERT_EQ(tx->metric.size(), 1u);
    EXPECT_DOUBLE_EQ(tx->metric.front().counter.value, 42.0)
        << "the values recorded after the last scrape were lost";
    EXPECT_TRUE(std::filesystem::exists(path));

    nixlMultiprocessCollector expired(dir_, std::chrono::seconds(0), /*reap_stale=*/true);
    EXPECT_TRUE(expired.Collect().empty());
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(MpCollectorFileTest, RecreatedWriterContinuesTheSeriesOfItsName) {
    const auto previous = dir_ / makeStoreFileName(555, 1, 0);
    {
        storeWriter writer(previous, "agent-a", "host", "", 0, kBuckets);
        writer.addCounter(TX_BYTES, 42);
    }
    storeWriter writer(dir_ / makeStoreFileName(555, 1, 1), "agent-a", "host", "", 1, kBuckets);
    writer.addCounter(TX_BYTES, 7);

    for (const auto ttl : {std::chrono::seconds(30), std::chrono::seconds(0)}) {
        SCOPED_TRACE("stale TTL " + std::to_string(ttl.count()) + "s");
        nixlMultiprocessCollector collector(dir_, ttl);
        const auto fams = collector.Collect();
        const auto *tx = findFamily(fams, "agent_tx_bytes_total");
        ASSERT_NE(tx, nullptr);
        ASSERT_EQ(tx->metric.size(), 1u);
        EXPECT_DOUBLE_EQ(tx->metric.front().counter.value, 7.0);
        EXPECT_EQ(
            labelMap(tx->metric.front()),
            (std::map<std::string, std::string>{{"hostname", "host"}, {"agent_name", "agent-a"}}));
        EXPECT_EQ(std::filesystem::exists(previous), ttl.count() > 0);
    }
}

TEST_F(MpCollectorFileTest, CoexistingWritersOfOneNameAreToldApartAndReportedOnce) {
    storeWriter first(dir_ / makeStoreFileName(666, 1, 0), "dup", "host", "", 0, kBuckets);
    first.addCounter(TX_BYTES, 1);
    nixlMultiprocessCollector collector(dir_);

    const auto expectOnlyFirstWithBaseLabels = [&collector] {
        const auto fams = collector.Collect();
        const auto *tx = findFamily(fams, "agent_tx_bytes_total");
        ASSERT_NE(tx, nullptr);
        ASSERT_EQ(tx->metric.size(), 1u);
        EXPECT_DOUBLE_EQ(tx->metric.front().counter.value, 1.0);
        EXPECT_FALSE(hasLabel(tx->metric.front(), "pid"));
        EXPECT_FALSE(hasLabel(tx->metric.front(), "agent_instance"));
    };

    ASSERT_NO_FATAL_FAILURE(expectOnlyFirstWithBaseLabels());
    {
        storeWriter second(dir_ / makeStoreFileName(666, 1, 1), "dup", "host", "", 1, kBuckets);
        second.addCounter(TX_BYTES, 2);

        const gtest::LogIgnoreGuard lig("2 live agents on host 'host' are named 'dup';");
        const auto fams = collector.Collect();
        const auto *tx = findFamily(fams, "agent_tx_bytes_total");
        ASSERT_NE(tx, nullptr);
        ASSERT_EQ(tx->metric.size(), 2u);
        const auto *m_first = findByLabel(*tx, "agent_instance", "0");
        const auto *m_second = findByLabel(*tx, "agent_instance", "1");
        ASSERT_NE(m_first, nullptr);
        ASSERT_NE(m_second, nullptr);
        EXPECT_DOUBLE_EQ(m_first->counter.value, 1.0);
        EXPECT_DOUBLE_EQ(m_second->counter.value, 2.0);
        EXPECT_TRUE(hasLabel(*m_first, "pid"));
        EXPECT_EQ(lig.getIgnoredCount(), 1u);
    }
    ASSERT_NO_FATAL_FAILURE(expectOnlyFirstWithBaseLabels());
}

TEST_F(MpCollectorFileTest, CollectOnEmptyDirYieldsNoFamilies) {
    nixlMultiprocessCollector collector(dir_, std::chrono::seconds(30), /*reap_stale=*/false);
    EXPECT_TRUE(collector.Collect().empty());
}

} // namespace
