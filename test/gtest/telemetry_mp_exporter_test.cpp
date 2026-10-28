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
#include "prometheus_mp_exporter.h"
#include "histogram_buckets.h"
#include "mp_store.h"
#include "mp_telemetry_fixture.h"
#include "owner_election.h"
#include "scrape_endpoint.h"
#include "telemetry.h"

#include "common.h"

#include "scrape_util.h"
#include "timeseries.h"

#include <gtest/gtest.h>

#include <prometheus/exposer.h>

#include <fcntl.h>
#include <sys/file.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using nixl::metrics_test::scrapeMetrics;
using nixl::telemetry::mp::heldAddressesExcept;
using nixl::telemetry::mp::ownerLockFileName;
using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

TEST_F(MpExporterTest, OwnerBindsAndRecordsToStore) {
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-owner"));
    EXPECT_TRUE(exporter.isExporter());

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());

    exporter.exportEvent({TX_BYTES, 1234});

    const auto snap = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-owner");
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1234u);
    EXPECT_EQ(snap->gauges[idx(TX_BYTES)], 1234u);
}

TEST_F(MpExporterTest, WriterModeWhenPortTaken) {
    // A stranger on the port: the elected rank cannot bind, so no rank of this
    // directory serves and the demotion is warned about rather than routine.
    prometheus::Exposer blocker("127.0.0.1:" + std::to_string(port_));

    const gtest::LogIgnoreGuard lig("is held by a process outside this run");
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-writer"));
    EXPECT_FALSE(exporter.isExporter());
    EXPECT_EQ(lig.getIgnoredCount(), 1);

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());

    exporter.exportEvent({RX_BYTES, 77});

    const auto snap = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-writer");
    EXPECT_EQ(snap->counters[idx(RX_BYTES)], 77u);
}

TEST_F(MpExporterTest, AgentsOfOneProcessShareTheEndpoint) {
    nixlTelemetryPrometheusMpExporter first(initParams("agent-first"));
    nixlTelemetryPrometheusMpExporter second(initParams("agent-second"));
    EXPECT_TRUE(first.isExporter());
    EXPECT_TRUE(second.isExporter());

    first.exportEvent({TX_BYTES, 1});
    second.exportEvent({TX_BYTES, 2});
    const auto metrics = scrapeMetrics(port_);
    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", {{"agent_name", "agent-first"}}), 1.0);
    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", {{"agent_name", "agent-second"}}), 2.0);
}

TEST_F(MpExporterTest, SpellingsOfOneDirectoryShareTheEndpoint) {
    const auto link = dir_ / "link";
    std::filesystem::create_directory_symlink(dir_, link);

    auto first = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-first"));
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", link.string());
    nixlTelemetryPrometheusMpExporter second(initParams("agent-second"));
    first.reset();
    EXPECT_TRUE(second.isExporter());
}

TEST_F(MpExporterTest, EndpointOutlivesTheAgentThatCreatedIt) {
    auto first = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-first"));
    nixlTelemetryPrometheusMpExporter survivor(initParams("agent-survivor"));
    first->exportEvent({TX_BYTES, 5});

    first.reset();
    EXPECT_TRUE(survivor.isExporter());
    const auto metrics = scrapeMetrics(port_);
    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", {{"agent_name", "agent-survivor"}}), 0.0);
    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", {{"agent_name", "agent-first"}}), 5.0);
}

TEST_F(MpExporterTest, LastAgentReleasesTheEndpoint) {
    using nixl::telemetry::mp::scrapeEndpoint;

    auto first = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-first"));
    auto second = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-second"));
    ASSERT_TRUE(second->isExporter());
    first.reset();
    second.reset();

    scrapeEndpoint fresh(dir_, "127.0.0.1:" + std::to_string(port_), std::chrono::seconds(2));
    EXPECT_EQ(fresh.claim(), scrapeEndpoint::status::SERVING);
}

TEST_F(MpExporterTest, AgentCreatedWhileTheLastOneIsDestroyedServes) {
    auto last = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-last"));
    ASSERT_TRUE(last->isExporter());

    std::latch start(2);
    std::unique_ptr<nixlTelemetryPrometheusMpExporter> successor;
    std::thread creator([&start, &successor] {
        start.arrive_and_wait();
        successor =
            std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-successor"));
    });
    start.arrive_and_wait();
    last.reset();
    creator.join();

    ASSERT_NE(successor, nullptr);
    EXPECT_TRUE(successor->isExporter());
}

TEST_F(MpExporterTest, ConcurrentConstructionSharesOneEndpoint) {
    constexpr std::size_t kAgents = 8;
    std::vector<std::unique_ptr<nixlTelemetryPrometheusMpExporter>> exporters(kAgents);
    std::latch start(kAgents);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < kAgents; ++i) {
        threads.emplace_back([&exporters, &start, i] {
            start.arrive_and_wait();
            exporters[i] = std::make_unique<nixlTelemetryPrometheusMpExporter>(
                initParams("agent-" + std::to_string(i)));
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (const auto &exporter : exporters) {
        ASSERT_NE(exporter, nullptr);
        EXPECT_TRUE(exporter->isExporter());
    }
}

TEST_F(MpExporterTest, ConcurrentReclaimServesEveryAgentOnceThePortFrees) {
    constexpr std::size_t kAgents = 4;
    auto blocker = std::make_unique<prometheus::Exposer>("127.0.0.1:" + std::to_string(port_));

    std::vector<std::unique_ptr<nixlTelemetryPrometheusMpExporter>> exporters;
    {
        const gtest::LogIgnoreGuard lig("is held by a process outside this run");
        for (std::size_t i = 0; i < kAgents; ++i) {
            exporters.push_back(std::make_unique<nixlTelemetryPrometheusMpExporter>(
                initParams("agent-" + std::to_string(i))));
        }
        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }
    blocker.reset();

    std::latch start(kAgents);
    std::vector<std::thread> threads;
    for (const auto &exporter : exporters) {
        threads.emplace_back([&start, agent = exporter.get()] {
            start.arrive_and_wait();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!agent->isExporter() && std::chrono::steady_clock::now() < deadline) {
                agent->exportEvent({TX_BYTES, 1});
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (const auto &exporter : exporters) {
        EXPECT_TRUE(exporter->isExporter());
    }
}

TEST_F(MpExporterTest, TakeoverServesOnTheFirstAttemptAfterTheOwnerGoes) {
    using nixl::telemetry::mp::monotonicNs;
    using nixl::telemetry::mp::scrapeEndpoint;

    const std::string bind = "127.0.0.1:" + std::to_string(port_);
    auto owner = std::make_unique<scrapeEndpoint>(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(owner->claim(), scrapeEndpoint::status::SERVING);

    scrapeEndpoint writer(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(writer.claim(), scrapeEndpoint::status::SIBLING_OWNS);

    owner.reset();
    writer.reclaim(std::chrono::nanoseconds(monotonicNs()));
    EXPECT_TRUE(writer.serving()) << "the first re-election after the owner left did not serve";
}

TEST_F(MpExporterTest, ReclaimWhileServingKeepsTheElection) {
    using nixl::telemetry::mp::monotonicNs;
    using nixl::telemetry::mp::scrapeEndpoint;

    const std::string bind = "127.0.0.1:" + std::to_string(port_);
    scrapeEndpoint endpoint(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(endpoint.claim(), scrapeEndpoint::status::SERVING);

    endpoint.reclaim(std::chrono::nanoseconds(monotonicNs()) + std::chrono::seconds(10));
    ASSERT_TRUE(endpoint.serving());

    scrapeEndpoint intruder(dir_, bind, std::chrono::seconds(2));
    EXPECT_EQ(intruder.claim(), scrapeEndpoint::status::SIBLING_OWNS)
        << "the serving process gave its election away";
}

TEST_F(MpExporterTest, ReclaimWaitsTheRetryIntervalAfterLosingToASibling) {
    using nixl::telemetry::mp::monotonicNs;
    using nixl::telemetry::mp::scrapeEndpoint;

    const std::string bind = "127.0.0.1:" + std::to_string(port_);
    auto owner = std::make_unique<scrapeEndpoint>(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(owner->claim(), scrapeEndpoint::status::SERVING);
    scrapeEndpoint writer(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(writer.claim(), scrapeEndpoint::status::SIBLING_OWNS);

    const auto start = std::chrono::nanoseconds(monotonicNs());
    writer.reclaim(start);
    owner.reset();

    writer.reclaim(start + std::chrono::milliseconds(100));
    EXPECT_FALSE(writer.serving()) << "retried inside the retry interval";
    writer.reclaim(start + std::chrono::milliseconds(300));
    EXPECT_TRUE(writer.serving());
}

TEST_F(MpExporterTest, ReclaimBacksOffAfterABindFailure) {
    using nixl::telemetry::mp::monotonicNs;
    using nixl::telemetry::mp::scrapeEndpoint;

    const std::string bind = "127.0.0.1:" + std::to_string(port_);
    auto blocker = std::make_unique<prometheus::Exposer>(bind);
    scrapeEndpoint endpoint(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(endpoint.claim(), scrapeEndpoint::status::PORT_TAKEN);

    const auto start = std::chrono::nanoseconds(monotonicNs());
    endpoint.reclaim(start);
    blocker.reset();

    endpoint.reclaim(start + std::chrono::seconds(1));
    EXPECT_FALSE(endpoint.serving()) << "retried inside the backoff";
    endpoint.reclaim(start + std::chrono::seconds(6));
    EXPECT_TRUE(endpoint.serving());
}

TEST_F(MpExporterTest, ReclaimsFromSeveralThreadsElectOneAtATime) {
    using nixl::telemetry::mp::monotonicNs;
    using nixl::telemetry::mp::scrapeEndpoint;

    const std::string bind = "127.0.0.1:" + std::to_string(port_);
    auto blocker = std::make_unique<prometheus::Exposer>(bind);
    scrapeEndpoint endpoint(dir_, bind, std::chrono::seconds(2));
    ASSERT_EQ(endpoint.claim(), scrapeEndpoint::status::PORT_TAKEN);

    constexpr int kThreads = 4;
    constexpr int kHours = 20;
    const auto base = std::chrono::nanoseconds(monotonicNs());
    std::latch start(kThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&endpoint, &start, base] {
            start.arrive_and_wait();
            for (int hour = 1; hour <= kHours; ++hour) {
                endpoint.reclaim(base + std::chrono::hours(hour));
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(endpoint.serving());

    blocker.reset();
    endpoint.reclaim(base + std::chrono::hours(kHours + 1));
    EXPECT_TRUE(endpoint.serving());
}

TEST_F(MpExporterTest, RanksConfiguredForAnotherAddressServeItAndWarn) {
    nixlTelemetryPrometheusMpExporter owner(initParams("agent-owner"));
    ASSERT_TRUE(owner.isExporter());

    env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT",
                std::to_string(gtest::PortAllocator::next_tcp_port()));
    {
        // Elections are per address, so this rank contends with nobody and
        // serves what it was configured with; the directory being served twice
        // is what gets reported.
        const gtest::LogIgnoreGuard lig("is not the only address serving it");
        nixlTelemetryPrometheusMpExporter second(initParams("agent-second"));
        EXPECT_TRUE(second.isExporter());
        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }
    env_.popVar();
}

TEST_F(MpExporterTest, LeftoverLockFileIsNotASecondOwner) {
    // Lock files outlive the run that created them, so a second address counts
    // only while someone holds its lock. Any warning here fails the test.
    { std::ofstream(dir_ / ownerLockFileName("127.0.0.1:1")); }

    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-owner"));
    EXPECT_TRUE(exporter.isExporter());
}

TEST_F(MpExporterTest, LeftoverLockFileIsNotASecondOwnerWhileAnotherOwnerProbesIt) {
    const auto leftover = dir_ / ownerLockFileName("127.0.0.1:1");
    { std::ofstream(leftover).put('\0'); }

    const nixl::scopedFd second_owner_probe(::open(leftover.c_str(), O_RDONLY | O_CLOEXEC));
    ASSERT_TRUE(second_owner_probe.valid());
    ASSERT_EQ(::flock(second_owner_probe.get(), LOCK_SH | LOCK_NB), 0);

    EXPECT_TRUE(heldAddressesExcept(dir_, "127.0.0.1:2").empty());
}

TEST_F(MpExporterTest, ForeignOwnedLockFileCannotSilenceTheRun) {
    const auto lock = dir_ / ownerLockFileName("127.0.0.1:" + std::to_string(port_));
    { std::ofstream(lock).put('\0'); }
    if (const auto why = giveAwayOwnership(lock)) {
        GTEST_SKIP() << *why;
    }

    // A planted lock must not read as a sibling win, or every rank of a shared
    // directory ends up writer-only and nothing serves.
    const gtest::LogIgnoreGuard lig("not a regular file owned by this user");
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-owner"));
    EXPECT_TRUE(exporter.isExporter());
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpExporterTest, DurationEventFeedsCounterGaugeAndHistogram) {
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-hist"));

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());

    exporter.exportEvent({XFER_TIME, 42});

    const auto snap = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->counters[idx(XFER_TIME)], 42u);
    EXPECT_EQ(snap->gauges[idx(XFER_TIME)], 42u);
    EXPECT_EQ(snap->histSums[idx(XFER_TIME)], 42u);

    // Default bounds start at 10us, so 42us lands in the 50us bucket (index 2).
    ASSERT_EQ(snap->bucketCount, nixl::telemetry::defaultHistogramBucketsUs().size());
    EXPECT_DOUBLE_EQ(snap->bucketBounds[2], 50.0);
    EXPECT_EQ(snap->histBuckets[idx(XFER_TIME)][2], 1u);
}

TEST_F(MpExporterTest, ExportEventRefreshesHeartbeat) {
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-beat"));

    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());
    const auto before = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(before.has_value());

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    exporter.exportEvent({TX_BYTES, 1});

    const auto after = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(after.has_value());
    EXPECT_GT(after->lastUpdateNs, before->lastUpdateNs);
}

TEST_F(MpExporterTest, LoadsThroughPluginManager) {
    // The plugin manager probes every registered plugin directory, so it warns
    // about the ones that do not hold this plugin before finding the one that does.
    const gtest::LogIgnoreGuard lig("Plugin file does not exist");
    nixlTelemetry telemetry("agent-loader", "prometheus_mp");
    EXPECT_FALSE(singleStoreFile().empty());
}

TEST_F(MpExporterTest, PrometheusIdLoadsTheSamePlugin) {
    const gtest::LogIgnoreGuard lig("Plugin file does not exist");
    auto &plugin_manager = nixlPluginManager::getInstance();
    const auto handle = plugin_manager.loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle, plugin_manager.loadTelemetryPlugin("prometheus_mp"));
    EXPECT_STREQ(handle->getName(), "prometheus_mp");

    nixlTelemetry telemetry("agent-alias", "prometheus");
    EXPECT_FALSE(singleStoreFile().empty());
}

TEST_F(MpExporterTest, CreatedTelemetryDirIsPrivate) {
    const auto sub = dir_ / "created";
    ASSERT_FALSE(std::filesystem::exists(sub));
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", sub.string());

    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-private"));

    ASSERT_TRUE(std::filesystem::is_directory(sub));
    EXPECT_EQ(std::filesystem::status(sub).permissions() & std::filesystem::perms::mask,
              std::filesystem::perms::owner_all);
}

TEST_F(MpExporterTest, GroupWritableTelemetryDirWarns) {
    const auto sub = dir_ / "group-loose";
    std::filesystem::create_directory(sub);
    std::filesystem::permissions(sub,
                                 std::filesystem::perms::owner_all |
                                     std::filesystem::perms::group_write,
                                 std::filesystem::perm_options::replace);
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", sub.string());

    const gtest::LogIgnoreGuard lig("is writable by group or other");
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-group-loose"));
    EXPECT_TRUE(exporter.isExporter());
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpExporterTest, WorldWritableTelemetryDirWarns) {
    const auto sub = dir_ / "world-loose";
    std::filesystem::create_directory(sub);
    std::filesystem::permissions(sub,
                                 std::filesystem::perms::owner_all |
                                     std::filesystem::perms::others_write,
                                 std::filesystem::perm_options::replace);
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", sub.string());

    const gtest::LogIgnoreGuard lig("is writable by group or other");
    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-world-loose"));
    EXPECT_TRUE(exporter.isExporter());
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpExporterTest, EmptyMultiprocDirDefaultsToAPerUserPerPortDir) {
    const auto dir = scopedDefaultDir();
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", "");

    nixlTelemetryPrometheusMpExporter exporter(initParams("agent-default"));
    EXPECT_TRUE(exporter.isExporter());
    EXPECT_EQ(std::filesystem::status(dir).permissions() & std::filesystem::perms::mask,
              std::filesystem::perms::owner_all);

    const auto file = singleStoreFile(dir);
    ASSERT_FALSE(file.empty());
    const auto snap = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->agentName, "agent-default");
}

TEST_F(MpExporterTest, UncreatableMultiprocDirThrows) {
    const auto file = dir_ / "not-a-dir";
    { std::ofstream(file).put('\0'); }
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", (file / "telemetry").string());

    EXPECT_THROW(
        { nixlTelemetryPrometheusMpExporter exporter(initParams("agent-uncreatable")); },
        std::runtime_error);
}

TEST_F(MpExporterTest, SquattedDefaultDirRunsWithoutTelemetry) {
    { std::ofstream(scopedDefaultDir()).put('\0'); }
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", "");
    env_.addVar(telemetryExporterVar, "prometheus_mp");

    const gtest::LogIgnoreGuard probe("Plugin file does not exist");
    const gtest::LogIgnoreGuard lig("will run without a telemetry sink");
    EXPECT_EQ(nixlTelemetry::create("agent-squatted"), nullptr);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(MpExporterTest, OverlongBucketListStaysFatalWithTheDefaultDir) {
    scopedDefaultDir();
    env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", "");
    env_.addVar(telemetryExporterVar, "prometheus_mp");
    std::string bounds = "1";
    for (std::size_t i = 2; i <= nixl::telemetry::mp::MP_STORE_MAX_BUCKETS + 1; ++i) {
        bounds += "," + std::to_string(i);
    }
    env_.addVar("NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US", bounds);

    const gtest::LogIgnoreGuard probe("Plugin file does not exist");
    const gtest::LogIgnoreGuard lig("Failed to create exporter");
    EXPECT_THROW(static_cast<void>(nixlTelemetry::create("agent-overlong")), std::runtime_error);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

} // namespace
