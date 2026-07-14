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

#include "prometheus_telemetry_fixture.h"

#include "plugin_manager.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"

#include "histogram_buckets.h"
#include "loopback_connection.h"
#include "open_metrics_text_parser.h"
#include "timeseries.h"

#include <array>
#include <chrono>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using nixl::doca_test::loopbackConnection;
using nixl::doca_test::timeSeries;

// Poll /metrics until it serves a non-empty body (the exposer may not be ready
// the instant the exporter is constructed), then parse it into the time-series
// model shared with the doca-telemetry tests.
timeSeries
scrapeMetrics(uint16_t port) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        const std::string body = loopbackConnection::httpGet(port, "/metrics");
        if (!body.empty()) {
            return timeSeries(nixl::doca_test::open_metrics_text::parse(body));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    return timeSeries(nixl::doca_test::seriesMap{});
}

// Cumulative count at the exact `le` boundary of this agent's bucket series.
// Parses `le` as a double so the check does not depend on boundary formatting.
std::optional<double>
bucketCount(const timeSeries &metrics,
            const std::string &bucket_metric,
            const std::string &agent_name,
            double le) {
    for (const auto &[id, samples] : metrics.series()) {
        if (id.name != bucket_metric || samples.empty()) {
            continue;
        }
        const auto agent = id.labels.find("agent_name");
        const auto le_it = id.labels.find("le");
        if (agent == id.labels.end() || agent->second != agent_name || le_it == id.labels.end()) {
            continue;
        }
        try {
            size_t pos = 0;
            if (std::stod(le_it->second, &pos) == le && pos == le_it->second.size()) {
                return samples.back().value;
            }
        }
        catch (const std::exception &) {
        }
    }
    return std::nullopt;
}

// The full set of `le` boundaries emitted for this agent's bucket series
// ("+Inf" becomes +infinity).
std::set<double>
bucketBoundaries(const timeSeries &metrics,
                 const std::string &bucket_metric,
                 const std::string &agent_name) {
    std::set<double> boundaries;
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        if (id.name != bucket_metric) {
            continue;
        }
        const auto agent = id.labels.find("agent_name");
        const auto le_it = id.labels.find("le");
        if (agent == id.labels.end() || agent->second != agent_name || le_it == id.labels.end()) {
            continue;
        }
        if (le_it->second == "+Inf") {
            boundaries.insert(std::numeric_limits<double>::infinity());
            continue;
        }
        try {
            size_t pos = 0;
            const double le = std::stod(le_it->second, &pos);
            if (pos == le_it->second.size()) {
                boundaries.insert(le);
            }
        }
        catch (const std::exception &) {
        }
    }
    return boundaries;
}

} // namespace

// AGENT_XFER_TIME events drive the agent_xfer_time_us histogram: _count is the
// number of observations, _sum is their total, and each cumulative _bucket{le}
// counts the observations at or below that boundary. Values land in distinct
// default buckets so the per-bucket counts are unambiguous.
TEST_F(prometheusTelemetryTest, XferTimeHistogramRecordsObservations) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_histogram_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    constexpr std::array<uint64_t, 4> values{5, 30, 200, 700}; // sum 935
    for (const uint64_t v : values) {
        EXPECT_EQ(exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_XFER_TIME, v}),
                  NIXL_SUCCESS);
    }

    const auto metrics = scrapeMetrics(port_);
    const nixl::doca_test::labelSet labels{{"agent_name", agent_name}};

    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_count", labels), std::optional<double>(4.0))
        << "histogram _count must equal the number of observations";
    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_sum", labels), std::optional<double>(935.0))
        << "histogram _sum must equal the sum of observations";

    // Cumulative counts at representative default boundaries: 5<=10 (1); +30<=100
    // (2); +200<=250 (3); +700<=1000 (4).
    EXPECT_EQ(bucketCount(metrics, "agent_xfer_time_us_bucket", agent_name, 10.0),
              std::optional<double>(1.0));
    EXPECT_EQ(bucketCount(metrics, "agent_xfer_time_us_bucket", agent_name, 100.0),
              std::optional<double>(2.0));
    EXPECT_EQ(bucketCount(metrics, "agent_xfer_time_us_bucket", agent_name, 250.0),
              std::optional<double>(3.0));
    EXPECT_EQ(bucketCount(metrics, "agent_xfer_time_us_bucket", agent_name, 1000.0),
              std::optional<double>(4.0));
}

// NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US replaces the default boundaries for the xfer
// histograms. prometheus-cpp always appends the implicit +Inf bucket, so the
// emitted boundary set is the configured list plus +Inf.
TEST_F(prometheusTelemetryTest, HistogramBucketsEnvOverride) {
    gtest::ScopedEnv bucket_env;
    bucket_env.addVar(nixl::telemetry::histogramBucketsUsVar, "100,200,300");

    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_histogram_env_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    const auto metrics = scrapeMetrics(port_);
    const std::set<double> expected{100.0, 200.0, 300.0, std::numeric_limits<double>::infinity()};
    EXPECT_EQ(bucketBoundaries(metrics, "agent_xfer_time_us_bucket", agent_name), expected)
        << "histogram buckets must reflect NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US plus +Inf";
}

TEST(histogramBucketsTest, ValidListIsParsedInOrder) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "1, 2.5, 10, 100");
    EXPECT_EQ(nixl::telemetry::resolveHistogramBucketsUs(),
              (std::vector<double>{1.0, 2.5, 10.0, 100.0}));
}

TEST(histogramBucketsTest, NonNumericThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "10,not_a_number,30");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, UnsortedThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "10,5,20");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, NonPositiveThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "0,10,20");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, AbsentUsesDefaults) {
    ::unsetenv(nixl::telemetry::histogramBucketsUsVar);
    EXPECT_EQ(nixl::telemetry::resolveHistogramBucketsUs(),
              nixl::telemetry::defaultHistogramBucketsUs());
}
