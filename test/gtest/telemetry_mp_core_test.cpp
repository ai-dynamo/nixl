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

// The multi-process exporter driven through nixlTelemetry rather than by direct
// exportEvent calls, so the staging queue, the drain task, per-metric gating and
// the drop accounting are all in the path. Mirrors the single-process coverage
// in telemetry_prometheus_test.cpp.

#include "mp_store.h"
#include "mp_telemetry_fixture.h"
#include "plugin_manager.h"
#include "telemetry.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_core_scrape.h"
#include "telemetry_event.h"

#include "common.h"

#include "scrape_util.h"
#include "timeseries.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace {

using nixl::metrics_test::labelSet;
using nixl::metrics_test::scrapeMetrics;
using nixl::metrics_test::scrapeUntilValue;
using nixl::telemetry::mp::readStoreSnapshot;

constexpr auto TX_BYTES = nixl_telemetry_event_type_t::AGENT_TX_BYTES;
constexpr auto RX_BYTES = nixl_telemetry_event_type_t::AGENT_RX_BYTES;

// The plugin manager probes every registered plugin directory, so it warns about
// the ones that do not hold this plugin before finding the one that does.
constexpr char PLUGIN_PROBE_WARNING[] = "Plugin file does not exist";

TEST_F(MpExporterTest, MetricAllowlistDeactivatesMetric) {
    gtest::ScopedEnv telemetry_env;
    telemetry_env.addVar(TELEMETRY_ENABLED_METRICS_VAR, "agent_tx_bytes");
    telemetry_env.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    const std::string agent_name = "mp_allowlist_agent";
    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    nixlTelemetry telemetry(agent_name, "prometheus_mp");
    telemetry.updateTxBytes(1000);
    telemetry.updateRxBytes(2000);

    const labelSet agent{{"agent_name", agent_name}};
    const auto metrics =
        scrapeUntilValue(port_, "agent_tx_bytes_total", 1000.0, std::chrono::seconds(5), agent);

    ASSERT_EQ(metrics.latestValue("agent_tx_bytes_total", agent), std::optional<double>(1000.0))
        << "allowed metric agent_tx_bytes_total never reached 1000";
    EXPECT_EQ(metrics.latestValue("agent_rx_bytes_total", agent), std::optional<double>(0.0))
        << "filtered metric must not be exported";

    // The store is the stronger assertion: an untouched slot proves the event
    // never reached the exporter, rather than merely not being rendered.
    const auto file = singleStoreFile();
    ASSERT_FALSE(file.empty());
    const auto snap = readStoreSnapshot(file).snapshot;
    ASSERT_TRUE(snap.has_value());
    EXPECT_EQ(snap->counters[idx(TX_BYTES)], 1000u);
    EXPECT_EQ(snap->counters[idx(RX_BYTES)], 0u) << "filtered metric must not reach the store";
    EXPECT_EQ(snap->gauges[idx(RX_BYTES)], 0u) << "filtered metric must not reach the store";
}

TEST_F(MpExporterTest, DroppedEventsCounterAccumulates) {
    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus_mp");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "mp_dropped_events_agent";
    auto exporter = handle->createExporter(initParams(agent_name));
    ASSERT_NE(exporter, nullptr);

    // Two flush deltas (7 then 5) as the core would emit them; the counter must
    // read their sum.
    constexpr std::array<uint64_t, 2> dropped_deltas{7, 5};
    uint64_t expected_total = 0;
    for (const uint64_t delta : dropped_deltas) {
        EXPECT_EQ(exporter->exportEvent(
                      {nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED, delta}),
                  NIXL_SUCCESS);
        expected_total += delta;
    }

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    EXPECT_EQ(metrics.latestValue("agent_telemetry_events_dropped_total",
                                  labelSet{{"agent_name", agent_name}}),
              std::optional<double>(static_cast<double>(expected_total)))
        << "dropped-events counter must sum every emitted delta (7+5)";
}

TEST_F(MpExporterTest, CoreUpdateDataOverflowConservation) {
    const std::string agent_name = "mp_core_update_overflow_agent";
    constexpr uint64_t kProduced = 100000; // far exceeds the 256-slot staging queue

    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    const auto scrape = gtest::scrapeCoreOverflow({.port = port_,
                                                   .exporter = "prometheus_mp",
                                                   .agent = agent_name,
                                                   .accepted_metric = "agent_tx_requests_num_total",
                                                   .accepted_event_weight = 1,
                                                   .expected_total_events = kProduced},
                                                  [](nixlTelemetry &telemetry) {
                                                      for (uint64_t i = 0; i < kProduced; ++i) {
                                                          telemetry.updateTxRequestsNum(1);
                                                      }
                                                  });

    ASSERT_TRUE(scrape.ok) << "accepted + dropped must reach produced (" << kProduced
                           << ") -- no silent loss";
    EXPECT_GT(scrape.dropped, 0.0) << "flooding a 256-slot staging queue must drop events";
}

TEST_F(MpExporterTest, CoreAddXferStatsOverflowConservation) {
    const std::string agent_name = "mp_core_xfer_overflow_agent";
    constexpr uint64_t kCalls = 100000;
    constexpr uint64_t kEventsPerCall = 4;

    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    const auto scrape = gtest::scrapeCoreOverflow(
        {.port = port_,
         .exporter = "prometheus_mp",
         .agent = agent_name,
         .accepted_metric = "agent_tx_requests_num_total",
         .accepted_event_weight = kEventsPerCall,
         .expected_total_events = kCalls * kEventsPerCall},
        [](nixlTelemetry &telemetry) {
            for (uint64_t i = 0; i < kCalls; ++i) {
                telemetry.addXferStats(
                    std::chrono::microseconds(10), true, 2000, std::chrono::microseconds(1));
            }
        });

    ASSERT_TRUE(scrape.ok) << "accepted*4 + dropped must reach produced events ("
                           << kCalls * kEventsPerCall << ") -- no silent loss";
    EXPECT_GT(scrape.dropped, 0.0) << "flooding the staging queue must drop xfer batches";
    EXPECT_EQ(std::fmod(scrape.dropped, static_cast<double>(kEventsPerCall)), 0.0)
        << "addXferStats drops the whole 4-event batch, so drops are multiples of 4";
}

} // namespace
