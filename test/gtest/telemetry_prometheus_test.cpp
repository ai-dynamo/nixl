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

#include "common.h"
#include "plugin_manager.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Minimal HTTP/1.1 GET over 127.0.0.1:<port>. Returns response body (empty
// string on any failure). Keeps the test free of a curl dependency.
std::string
httpGet(uint16_t port, const std::string &path) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return {};

    const struct timeval tv{3, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return {};
    }

    const std::string req =
        "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);

    std::string response;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, n);
    }
    ::close(fd);

    const auto pos = response.find("\r\n\r\n");
    return pos == std::string::npos ? std::string{} : response.substr(pos + 4);
}

std::string
waitForMetricsBody(uint16_t port) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    std::string body;
    do {
        body = httpGet(port, "/metrics");
        if (!body.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    return body;
}

} // namespace

class prometheusTelemetryTest : public ::testing::Test {
protected:
    // Register the freshly built plugin directory exactly once for the whole
    // test suite. Doing it in SetUp() instead would re-register on every
    // test and trip the plugin manager's "already registered" warning, which
    // the gtest main() treats as a test failure.
    static void
    SetUpTestSuite() {
        nixlPluginManager::getInstance().addPluginDirectory(std::string(BUILD_DIR) +
                                                            "/src/plugins/telemetry/prometheus");
    }

    void
    SetUp() override {
        port_ = gtest::PortAllocator::next_tcp_port();
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_LOCAL", "y");
        env_.addVar("NIXL_TELEMETRY_PROMETHEUS_PORT", std::to_string(port_));
    }

    void
    TearDown() override {
        env_.popVar();
        env_.popVar();
    }

    gtest::ScopedEnv env_;
    uint16_t port_ = 0;
};

// Regression test for a bug where the pre-registered per-agent metric
// families were immediately wiped from the shared prometheus::Registry by
// the dtor of a temporary CounterEntry/GaugeEntry created during
// `counters_[name] = {&family, &metric}`. Before the fix, this scrape body
// contained ONLY exposer_* self-metrics; `agent_*` families were absent,
// and the cached metric* pointers were left dangling (UB on first event).
TEST_F(prometheusTelemetryTest, AgentMetricsAppearInScrape) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr) << "Failed to load prometheus telemetry plugin";

    const std::string agent_name = "prometheus_test_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    const std::string body = waitForMetricsBody(port_);
    ASSERT_FALSE(body.empty()) << "Got empty /metrics response on port " << port_;

    // The 8 counter families that initializeMetrics() must publish.
    const std::vector<std::string> expected_counters = {
        "agent_tx_bytes_total",
        "agent_rx_bytes_total",
        "agent_tx_requests_num_total",
        "agent_rx_requests_num_total",
        "agent_memory_registered_total",
        "agent_memory_deregistered_total",
        "agent_xfer_time_total",
        "agent_xfer_post_time_total",
    };
    for (const auto &c : expected_counters) {
        EXPECT_NE(body.find(c), std::string::npos)
            << "Missing counter family \"" << c << "\" in /metrics body";
    }

    // Gauges share a name with their counter; they are serialized without the
    // "_total" suffix, so match via the opening label brace.
    EXPECT_NE(body.find("\nagent_memory_registered{"), std::string::npos)
        << "Missing agent_memory_registered gauge";
    EXPECT_NE(body.find("\nagent_memory_deregistered{"), std::string::npos)
        << "Missing agent_memory_deregistered gauge";

    // Each metric must carry the three labels the exporter attaches.
    EXPECT_NE(body.find("agent_name=\"" + agent_name + "\""), std::string::npos)
        << "agent_name label missing";
    EXPECT_NE(body.find("category=\"NIXL_TELEMETRY_TRANSFER\""), std::string::npos);
    EXPECT_NE(body.find("category=\"NIXL_TELEMETRY_MEMORY\""), std::string::npos);
    EXPECT_NE(body.find("category=\"NIXL_TELEMETRY_PERFORMANCE\""), std::string::npos);
    EXPECT_NE(body.find("hostname=\""), std::string::npos);
}

// Drives the hot path to surface the dangling-pointer consequence of the
// same root-cause bug. On the buggy code:
//   counters_["agent_tx_bytes"].metric points into freed heap (the Counter
//   that Family::Add() created was Remove()d by a temporary CounterEntry's
//   dtor just after map insertion).
// exportEvent() then reaches that pointer and calls Counter::Increment on
// freed memory. Under AddressSanitizer this is a reliable heap-use-after-
// free; unsanitized, it is either a silent no-op (if the slot has not been
// recycled) or observable via the scrape check below — the family has no
// remaining Counter instance, so Family::Collect returns {} and the metric
// is missing from /metrics entirely.
TEST_F(prometheusTelemetryTest, ExportEventIncrementReflectedInScrape) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_ub_test_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    // Five increments of 1000 bytes each → cumulative total must be 5000 in
    // the scrape body for AGENT_TX_BYTES. On buggy code, each Increment()
    // call dereferences a dangling Counter*; even if it returns without
    // crashing, the Family has no metric instance so the scrape below will
    // not contain "agent_tx_bytes_total{" at all.
    constexpr uint64_t kIncrement = 1000;
    constexpr int kEventCount = 5;
    for (int i = 0; i < kEventCount; ++i) {
        const nixlTelemetryEvent event{nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER,
                                       nixl_telemetry_event_type_t::AGENT_TX_BYTES,
                                       kIncrement};
        EXPECT_EQ(exporter->exportEvent(event), NIXL_SUCCESS);
    }

    const std::string body = waitForMetricsBody(port_);
    ASSERT_FALSE(body.empty()) << "Got empty /metrics response on port " << port_;

    // Locate the specific labeled line: agent_tx_bytes_total{...agent_name="..."} <value>
    const std::string needle = "agent_tx_bytes_total{agent_name=\"" + agent_name +
        "\",category=\"NIXL_TELEMETRY_TRANSFER\",hostname=\"";
    const auto line_pos = body.find(needle);
    ASSERT_NE(line_pos, std::string::npos)
        << "agent_tx_bytes_total for this agent is not in scrape body.\n"
        << "On buggy code, counters_ map holds a dangling Counter* and "
        << "Family::metrics_ is empty, so Family::Collect() returns {} and "
        << "TextSerializer emits nothing for this family.";

    // Find end-of-line and extract the numeric value after "} "
    const auto eol = body.find('\n', line_pos);
    ASSERT_NE(eol, std::string::npos);
    const std::string line = body.substr(line_pos, eol - line_pos);
    const auto brace_close = line.find("} ");
    ASSERT_NE(brace_close, std::string::npos) << "line shape unexpected: " << line;
    const std::string value_str = line.substr(brace_close + 2);
    const double value = std::stod(value_str);

    EXPECT_EQ(value, static_cast<double>(kIncrement * kEventCount))
        << "Counter value after " << kEventCount << " × Increment(" << kIncrement << ") should be "
        << (kIncrement * kEventCount) << " but the scraped line is: " << line;
}
