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

#include "histogram_parity.h"
#include "scrape_util.h"

#include "doca_exporter.h"
#include "prometheus_exporter.h"
#include "telemetry_event.h"

#include <array>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

using nixl::doca_test::loopbackConnection;
using nixl::doca_test::scrapeUntilValue;
using nixl::telemetry_test::histogramSeriesLines;

namespace {

constexpr char prometheusPortVar[] = "NIXL_TELEMETRY_PROMETHEUS_PORT";
constexpr char prometheusLocalVar[] = "NIXL_TELEMETRY_PROMETHEUS_LOCAL";
constexpr char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
constexpr char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

} // namespace

class histogramExporterParityTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        prometheus_port_ = loopbackConnection::findFreePort();
        doca_port_ = loopbackConnection::findFreePort();
        ASSERT_NE(prometheus_port_, 0);
        ASSERT_NE(doca_port_, 0);
        ASSERT_NE(prometheus_port_, doca_port_);

        ASSERT_EQ(::setenv(prometheusLocalVar, "y", 1), 0);
        ASSERT_EQ(::setenv(prometheusPortVar, std::to_string(prometheus_port_).c_str(), 1), 0);
        ASSERT_EQ(::setenv(docaPrometheusLocalVar, "y", 1), 0);
        ASSERT_EQ(::setenv(docaPrometheusPortVar, std::to_string(doca_port_).c_str(), 1), 0);
    }

    void
    TearDown() override {
        ::unsetenv(prometheusLocalVar);
        ::unsetenv(prometheusPortVar);
        ::unsetenv(docaPrometheusLocalVar);
        ::unsetenv(docaPrometheusPortVar);
    }

    uint16_t prometheus_port_ = 0;
    uint16_t doca_port_ = 0;
};

TEST_F(histogramExporterParityTest, XferDurationHistogramPayloadMatchesAcrossExporters) {
    constexpr char agentName[] = "histogram_parity_agent";
    const nixlTelemetryExporterInitParams params{agentName, 4096};

    nixlTelemetryPrometheusExporter prometheus_exporter(params);
    nixlTelemetryDocaExporter doca_exporter(params);

    constexpr std::array<uint64_t, 4> xfer_time_values{5, 30, 200, 700};
    for (const uint64_t value : xfer_time_values) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_XFER_TIME, value);
        ASSERT_EQ(prometheus_exporter.exportEvent(event), NIXL_SUCCESS);
        ASSERT_EQ(doca_exporter.exportEvent(event), NIXL_SUCCESS);
    }

    constexpr std::array<uint64_t, 4> xfer_post_time_values{2, 15, 80, 300};
    for (const uint64_t value : xfer_post_time_values) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_XFER_POST_TIME, value);
        ASSERT_EQ(prometheus_exporter.exportEvent(event), NIXL_SUCCESS);
        ASSERT_EQ(doca_exporter.exportEvent(event), NIXL_SUCCESS);
    }

    ASSERT_EQ(doca_exporter.flush(), NIXL_SUCCESS);

    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};
    const auto doca_metrics = scrapeUntilValue(
        doca_port_, "agent_xfer_time_us_count", 4.0, std::chrono::seconds(12), labels);
    ASSERT_EQ(doca_metrics.latestValue("agent_xfer_time_us_count", labels),
              std::optional<double>(4.0));

    const std::string prometheus_body = loopbackConnection::httpGet(prometheus_port_, "/metrics");
    ASSERT_FALSE(prometheus_body.empty()) << "empty Prometheus /metrics body";

    const auto prometheus_metrics =
        nixl::doca_test::timeSeries(nixl::doca_test::open_metrics_text::parse(prometheus_body));
    const auto prometheus_histograms = histogramSeriesLines(prometheus_metrics, agentName);
    const auto doca_histograms = histogramSeriesLines(doca_metrics, agentName);

    ASSERT_FALSE(doca_histograms.empty())
        << "DOCA exporter published no histogram series; the parity check would be vacuous";

    EXPECT_EQ(prometheus_histograms, doca_histograms);
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
