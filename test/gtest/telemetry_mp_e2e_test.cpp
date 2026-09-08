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

#include "common.h"
#include "mp_telemetry_fixture.h"

#include "common/scoped_fd.h"

#include "scrape_util.h"
#include "timeseries.h"

#include <absl/strings/str_join.h>

#include <gtest/gtest.h>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

using nixl::scopedFd;
using nixl::metrics_test::labelSet;
using nixl::metrics_test::scrapeMetrics;
using nixl::metrics_test::timeSeries;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

// The `metric` series of every agent in the scrape, keyed by agent_name.
[[nodiscard]] std::map<std::string, double>
seriesByAgent(const timeSeries &metrics, const std::string &metric) {
    std::map<std::string, double> out;
    for (const auto &[id, samples] : metrics.series()) {
        const auto agent = id.labels.find("agent_name");
        if (id.name != metric || agent == id.labels.end() || samples.empty()) {
            continue;
        }
        out[agent->second] = samples.back().value;
    }
    return out;
}

void
runWriterChild(int go_fd, int ready_fd, int quit_fd, const std::string &agent, uint64_t tx_value) {
    char c = 0;
    while (::read(go_fd, &c, 1) > 0) {}

    int rc = 0;
    try {
        nixlTelemetryPrometheusMpExporter exporter(initParams(agent));
        exporter.exportEvent({TX_BYTES, tx_value});
        const char ok = 1;
        if (::write(ready_fd, &ok, 1) != 1) {
            ::_exit(4);
        }
        // Block until the parent closes the quit pipe.
        while (::read(quit_fd, &c, 1) > 0) {}
    }
    catch (...) {
        rc = 3;
    }
    ::_exit(rc);
}

class MpE2ETest : public MpExporterTest {
protected:
    void
    SetUp() override {
        MpExporterTest::SetUp();
        // Dead processes become stale immediately so the reaping check is prompt.
        env_.addVar("NIXL_TELEMETRY_MP_STALE_TTL", "0");
    }

    // Runs even when a fatal assertion aborts the test body mid-fork.
    void
    TearDown() override {
        for (const pid_t pid : children_) {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        }
        MpExporterTest::TearDown();
    }

    [[nodiscard]] std::size_t
    countStores(const std::string &prefix) const {
        std::size_t n = 0;
        for (const auto &entry : std::filesystem::directory_iterator(dir_)) {
            n += entry.path().filename().string().rfind(prefix, 0) == 0 ? 1 : 0;
        }
        return n;
    }

    std::vector<pid_t> children_;
};

TEST_F(MpE2ETest, AllRankProcessesAggregateBehindOneEndpointAndStaleAreDropped) {
    constexpr int kChildren = 3;

    int go_pipe[2];
    int ready_pipe[2];
    int quit_pipe[2];
    ASSERT_EQ(::pipe(go_pipe), 0);
    scopedFd go_read(go_pipe[0]);
    scopedFd go_write(go_pipe[1]);
    ASSERT_EQ(::pipe(ready_pipe), 0);
    scopedFd ready_read(ready_pipe[0]);
    scopedFd ready_write(ready_pipe[1]);
    ASSERT_EQ(::pipe(quit_pipe), 0);
    scopedFd quit_read(quit_pipe[0]);
    scopedFd quit_write(quit_pipe[1]);

    // Fork children while the parent is still single-threaded (before it builds
    // the owner exporter, which starts civetweb threads).
    for (int i = 0; i < kChildren; ++i) {
        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            go_write.reset();
            ready_read.reset();
            quit_write.reset();
            runWriterChild(go_read.get(),
                           ready_write.get(),
                           quit_read.get(),
                           "agent-" + std::to_string(i),
                           static_cast<uint64_t>((i + 1) * 100));
        }
        children_.push_back(pid);
    }

    go_read.reset();
    ready_write.reset();
    quit_read.reset();

    // Parent wins the election and serves the endpoint.
    nixlTelemetryPrometheusMpExporter owner(initParams("agent-parent"));
    ASSERT_TRUE(owner.isExporter());
    owner.exportEvent({TX_BYTES, 999});
    owner.exportEvent({XFER_TIME, 1234});

    // Release the children (they now become writers) and wait for readiness.
    go_write.reset();
    for (int i = 0; i < kChildren; ++i) {
        pollfd pfd{ready_read.get(), POLLIN, 0};
        ASSERT_GT(::poll(&pfd, 1, 30000), 0) << "writer " << i << " never signalled readiness";
        char c = 0;
        ASSERT_EQ(::read(ready_read.get(), &c, 1), 1);
    }

    // Phase 1: every process must appear behind the single owner endpoint.
    const auto metrics = scrapeMetrics(port_);
    const auto phase1 = seriesByAgent(metrics, "agent_tx_bytes_total");
    ASSERT_EQ(phase1.size(), static_cast<std::size_t>(kChildren + 1))
        << absl::StrJoin(phase1, ", ", absl::PairFormatter("="));
    EXPECT_DOUBLE_EQ(phase1.at("agent-parent"), 999.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-0"), 100.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-1"), 200.0);
    EXPECT_DOUBLE_EQ(phase1.at("agent-2"), 300.0);

    const auto hist_count = seriesByAgent(metrics, "agent_xfer_time_us_count");
    const auto hist_sum = seriesByAgent(metrics, "agent_xfer_time_us_sum");
    ASSERT_EQ(hist_count.size(), static_cast<std::size_t>(kChildren + 1))
        << absl::StrJoin(hist_count, ", ", absl::PairFormatter("="));
    ASSERT_EQ(hist_sum.size(), static_cast<std::size_t>(kChildren + 1))
        << absl::StrJoin(hist_sum, ", ", absl::PairFormatter("="));
    EXPECT_DOUBLE_EQ(hist_count.at("agent-parent"), 1.0);
    EXPECT_DOUBLE_EQ(hist_sum.at("agent-parent"), 1234.0);

    // Buckets are cumulative and carry one series per bound, so where the
    // parent's single 1234us sample landed is only visible per `le`.
    const auto bucket = [&](const std::string &agent, const std::string &le) {
        return metrics.latestValue("agent_xfer_time_us_bucket",
                                   labelSet{{"agent_name", agent}, {"le", le}});
    };
    EXPECT_EQ(bucket("agent-parent", "1000"), std::optional<double>(0.0));
    EXPECT_EQ(bucket("agent-parent", "2500"), std::optional<double>(1.0));
    EXPECT_EQ(bucket("agent-parent", "+Inf"), std::optional<double>(1.0));

    // Writers that observed nothing still expose the family, at zero.
    for (int i = 0; i < kChildren; ++i) {
        const std::string agent = "agent-" + std::to_string(i);
        EXPECT_DOUBLE_EQ(hist_count.at(agent), 0.0) << agent;
        EXPECT_DOUBLE_EQ(hist_sum.at(agent), 0.0) << agent;
        EXPECT_EQ(bucket(agent, "+Inf"), std::optional<double>(0.0)) << agent;
    }

    // Kill one child and reap it so its pid is truly gone before the next scrape.
    // It stays in children_ until reaped, so TearDown() finishes the job if an
    // assertion below leaves the test body first, and stops tracking it once
    // reaped, so a recycled pid is never signalled.
    const pid_t dead = children_.front();
    const std::string dead_prefix =
        std::string(nixl::telemetry::mp::MP_STORE_FILE_PREFIX) + std::to_string(dead) + ".";
    ASSERT_EQ(countStores(dead_prefix), 1u);
    ASSERT_EQ(::kill(dead, SIGKILL), 0);
    ASSERT_EQ(::waitpid(dead, nullptr, 0), dead);
    children_.erase(children_.begin());

    // Phase 2: the dead child's series is dropped (and its store reaped).
    const auto phase2 = seriesByAgent(scrapeMetrics(port_), "agent_tx_bytes_total");
    EXPECT_EQ(phase2.count("agent-0"), 0u);
    EXPECT_EQ(phase2.count("agent-1"), 1u);
    EXPECT_EQ(phase2.count("agent-2"), 1u);
    EXPECT_EQ(phase2.count("agent-parent"), 1u);

    // Dropping the series is not enough: the store itself must be unlinked.
    EXPECT_EQ(countStores(dead_prefix), 0u);

    // The remaining children are released and reaped by TearDown().
}

} // namespace
