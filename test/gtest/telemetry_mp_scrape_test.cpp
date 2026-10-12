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

#include "mp_telemetry_fixture.h"
#include "plugin_manager.h"
#include "telemetry.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_event.h"

#include "common.h"
#include "common/hostname.h"

#include "histogram_buckets.h"
#include "scrape_util.h"
#include "timeseries.h"

#include <prometheus/exposer.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <ranges>
#include <set>
#include <string>

namespace {

using nixl::metrics_test::labelSet;
using nixl::metrics_test::scrapeMetrics;
using nixl::metrics_test::seriesId;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;
constexpr auto XFER_TIME = nixl_telemetry_event_type_t::AGENT_XFER_TIME;

constexpr char PLUGIN_PROBE_WARNING[] = "Plugin file does not exist";

[[nodiscard]] std::unique_ptr<nixlTelemetryExporter>
createExporter(const std::string &agent_name) {
    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    const auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus_mp");
    return handle == nullptr ? nullptr : handle->createExporter(initParams(agent_name));
}

TEST_F(MpExporterTest, ScrapeEmitsExactlyTheDescriptorSeries) {
    const auto exporter = createExporter("mp_parity_agent");
    ASSERT_NE(exporter, nullptr);

    std::set<std::string> expected{"agent_errors_total"};
    for (const auto type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(type);
        for (const char *name : {descriptor.counterName, descriptor.gaugeName}) {
            if (name != nullptr) {
                expected.insert(name);
            }
        }
        if (descriptor.histogramName != nullptr) {
            for (const char *suffix : {"_bucket", "_sum", "_count"}) {
                expected.insert(std::string(descriptor.histogramName) + suffix);
            }
        }
    }

    const auto metrics = scrapeMetrics(port_);
    std::set<std::string> actual;
    for (const auto &id : std::views::keys(metrics.series())) {
        if (id.labels.contains("agent_name")) {
            actual.insert(id.name);
        }
    }
    EXPECT_EQ(actual, expected);
}

TEST_F(MpExporterTest, AgentMetricsAppearInScrape) {
    env_.unsetVar("NIXL_TELEMETRY_RANK_ENV");
    env_.unsetVar("LOCAL_RANK");
    const std::string agent_name = "mp_scrape_agent";
    const auto exporter = createExporter(agent_name);
    ASSERT_NE(exporter, nullptr);

    const std::string host = nixl::getHostname().value_or("unknown");
    const auto tx_series = [&](const std::string &agent) {
        return seriesId{"agent_tx_bytes_total", {{"agent_name", agent}, {"hostname", host}}};
    };
    const auto lone = scrapeMetrics(port_);
    EXPECT_TRUE(lone.series().contains(tx_series(agent_name)))
        << "a lone agent must carry only agent_name and hostname";
    for (const auto &id : std::views::keys(lone.series())) {
        EXPECT_FALSE(id.labels.contains("pid") || id.labels.contains("agent_instance")) << id.name;
    }

    const std::string peer_name = "mp_scrape_agent_peer";
    {
        const auto peer = createExporter(peer_name);
        ASSERT_NE(peer, nullptr);
        const auto both = scrapeMetrics(port_);
        EXPECT_TRUE(both.series().contains(tx_series(agent_name)));
        EXPECT_TRUE(both.series().contains(tx_series(peer_name)));
    }
    EXPECT_TRUE(scrapeMetrics(port_).series().contains(tx_series(agent_name)))
        << "the first agent's series must survive the peer's teardown";
}

TEST_F(MpExporterTest, ByteCounterSumsWhileLastGaugeTracksFinalOp) {
    const std::string agent_name = "mp_last_gauge_agent";
    const auto exporter = createExporter(agent_name);
    ASSERT_NE(exporter, nullptr);
    for (const uint64_t bytes : {1000, 2000, 3500}) {
        exporter->exportEvent({TX_BYTES, bytes});
    }
    for (const uint64_t bytes : {500, 1500}) {
        exporter->exportEvent({RX_BYTES, bytes});
    }

    const auto metrics = scrapeMetrics(port_);
    const labelSet agent{{"agent_name", agent_name}};
    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", agent), 6500.0);
    EXPECT_EQ(metrics.latestValue("agent_tx_last_bytes", agent), 3500.0);
    EXPECT_EQ(metrics.latestValue("agent_rx_bytes_total", agent), 2000.0);
    EXPECT_EQ(metrics.latestValue("agent_rx_last_bytes", agent), 1500.0);
}

TEST_F(MpExporterTest, ErrorCountersUseBoundedStatusLabel) {
    const std::string agent_name = "mp_error_counter_agent";
    const auto exporter = createExporter(agent_name);
    ASSERT_NE(exporter, nullptr);
    exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1});
    exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1});
    exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_BACKEND, 1});

    const auto metrics = scrapeMetrics(port_);
    EXPECT_EQ(metrics.latestValue("agent_errors_total",
                                  {{"agent_name", agent_name}, {"status", "invalid_param"}}),
              2.0);
    EXPECT_EQ(metrics.latestValue("agent_errors_total",
                                  {{"agent_name", agent_name}, {"status", "backend"}}),
              1.0);
}

TEST_F(MpExporterTest, XferTimeHistogramRecordsObservations) {
    const std::string agent_name = "mp_histogram_agent";
    const auto exporter = createExporter(agent_name);
    ASSERT_NE(exporter, nullptr);
    for (const uint64_t us : {5, 30, 200, 700}) {
        exporter->exportEvent({XFER_TIME, us});
    }

    const auto metrics = scrapeMetrics(port_);
    const auto bucket = [&](const std::string &le) {
        return metrics.latestValue("agent_xfer_time_us_bucket",
                                   {{"agent_name", agent_name}, {"le", le}});
    };
    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_count", {{"agent_name", agent_name}}), 4.0);
    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_sum", {{"agent_name", agent_name}}), 935.0);
    EXPECT_EQ(bucket("10"), 1.0);
    EXPECT_EQ(bucket("100"), 2.0);
    EXPECT_EQ(bucket("250"), 3.0);
    EXPECT_EQ(bucket("1000"), 4.0);
}

TEST_F(MpExporterTest, HistogramBucketsEnvOverride) {
    env_.addVar(nixl::telemetry::histogramBucketsUsVar, "100,200,300");
    const auto exporter = createExporter("mp_histogram_env_agent");
    ASSERT_NE(exporter, nullptr);

    const auto metrics = scrapeMetrics(port_);
    std::set<std::string> bounds;
    for (const auto &id : std::views::keys(metrics.series())) {
        if (id.name == "agent_xfer_time_us_bucket") {
            bounds.insert(id.labels.at("le"));
        }
    }
    EXPECT_EQ(bounds, (std::set<std::string>{"100", "200", "300", "+Inf"}));
}

TEST_F(MpExporterTest, BindCollisionCreateIsNonFatalWarn) {
    prometheus::Exposer blocker("127.0.0.1:" + std::to_string(port_));
    env_.addVar(telemetryExporterVar, "prometheus_mp");

    const gtest::LogIgnoreGuard probe(PLUGIN_PROBE_WARNING);
    const gtest::LogIgnoreGuard lig("is held by a process outside this run");
    std::unique_ptr<nixlTelemetry> telemetry;
    EXPECT_NO_THROW(telemetry = nixlTelemetry::create("mp_bind_collision_create_agent"));
    EXPECT_NE(telemetry, nullptr);
    EXPECT_EQ(lig.getIgnoredCount(), 1u);
    EXPECT_FALSE(singleStoreFile().empty());
}

} // namespace
