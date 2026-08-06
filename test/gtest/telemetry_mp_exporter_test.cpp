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
#include "owner_election.h"
#include "plugin_manager.h"
#include "scrape_endpoint.h"
#include "telemetry.h"

#include "common.h"

#include <gtest/gtest.h>

#include <prometheus/exposer.h>

#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

namespace {

using nixl::telemetry::mp::ownerLockFileName;
using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

[[nodiscard]] std::size_t
idx(nixl_telemetry_event_type_t t) {
    return static_cast<std::size_t>(t);
}

[[nodiscard]] nixlTelemetryExporterInitParams
initParams(const std::string &agent) {
    return nixlTelemetryExporterInitParams{agent, 4096};
}

class MpExporterTest : public ::testing::Test {
protected:
    // A build tree has no <libnixl.so dir>/plugins, so LoadsThroughPluginManager
    // finds the plugin only if the build path is registered. Registered once per
    // process (--gtest_repeat re-enters this hook), and only when it exists:
    // re-registering, or registering a missing directory, logs a warning/error
    // that the gtest main counts as a failure. When it is absent (a binary run
    // from an install tree) NIXL_PLUGIN_DIR is what supplies the plugin.
    static void
    SetUpTestSuite() {
        [[maybe_unused]] static const bool registered = [] {
            const std::string build_plugin_dir =
                std::string(BUILD_DIR) + "/src/plugins/telemetry/prometheus_mp";
            if (std::filesystem::is_directory(build_plugin_dir)) {
                nixlPluginManager::getInstance().addPluginDirectory(build_plugin_dir);
            }
            return true;
        }();
    }

    void
    SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        dir_ = std::filesystem::path(::testing::TempDir()) /
            ("nixl_mp_exporter_" + std::to_string(::getpid()) + "_" + info->name());
        std::filesystem::create_directories(dir_);
        // What the exporter asks operators for; without it a permissive umask
        // makes it warn about the directory and the gtest main counts that.
        std::filesystem::permissions(
            dir_, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
        env_.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", dir_.string());
    }

    void
    TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] std::filesystem::path
    singleStoreFile() const {
        std::error_code ec;
        for (const auto &entry : std::filesystem::directory_iterator(dir_, ec)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("nixl.", 0) == 0 && name.size() > 5 &&
                name.substr(name.size() - 5) == ".mmap") {
                return entry.path();
            }
        }
        return {};
    }

    gtest::ScopedEnv env_;
    uint16_t port_ = 0;
    std::filesystem::path dir_;
};

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

TEST_F(MpExporterTest, WriterBehindSiblingOwnerIsQuiet) {
    // Losing the election to a sibling on the same address is the routine case:
    // no warning, otherwise the gtest problem counter fails this test.
    nixlTelemetryPrometheusMpExporter owner(initParams("agent-owner"));
    ASSERT_TRUE(owner.isExporter());

    nixlTelemetryPrometheusMpExporter writer(initParams("agent-writer"));
    EXPECT_FALSE(writer.isExporter());
}

TEST_F(MpExporterTest, WriterTakesOverWhenTheOwnerExits) {
    auto owner = std::make_unique<nixlTelemetryPrometheusMpExporter>(initParams("agent-owner"));
    ASSERT_TRUE(owner->isExporter());

    nixlTelemetryPrometheusMpExporter writer(initParams("agent-writer"));
    ASSERT_FALSE(writer.isExporter());

    writer.exportEvent({TX_BYTES, 1});
    ASSERT_FALSE(writer.isExporter()) << "took the address from a live owner";

    owner.reset();
    // Export until the re-election throttle the export above armed has passed,
    // rather than assuming how long that is.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!writer.isExporter() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        writer.exportEvent({TX_BYTES, 1});
    }
    EXPECT_TRUE(writer.isExporter());
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

TEST_F(MpExporterTest, ForeignOwnedLockFileCannotSilenceTheRun) {
    if (::geteuid() != 0) {
        GTEST_SKIP() << "needs privileges to give the lock file another owner";
    }
    const auto lock = dir_ / ownerLockFileName("127.0.0.1:" + std::to_string(port_));
    { std::ofstream(lock).put('\0'); }
    constexpr uid_t nobody = 65534;
    if (::chown(lock.c_str(), nobody, static_cast<gid_t>(-1)) != 0) {
        GTEST_SKIP() << "cannot give the lock file another owner: " << strerror(errno);
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

TEST(MpExporterStandaloneTest, MissingMultiprocDirThrows) {
    gtest::ScopedEnv env;
    env.addVar("NIXL_TELEMETRY_MULTIPROC_DIR", "");
    EXPECT_THROW(
        { nixlTelemetryPrometheusMpExporter exporter(nixlTelemetryExporterInitParams{"a", 4096}); },
        std::runtime_error);
}

} // namespace
