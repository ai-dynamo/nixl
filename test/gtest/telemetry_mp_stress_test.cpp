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
#include "telemetry_core_scrape.h"

#include "common.h"
#include "telemetry.h"
#include "telemetry_event.h"

#include "scrape_util.h"
#include "timeseries.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <latch>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using nixl::metrics_test::labelSet;
using nixl::metrics_test::timeSeries;
using testing::AnyOf;
using testing::Optional;

constexpr char PLUGIN_PROBE_WARNING[] = "Plugin file does not exist";

void
runConcurrently(uint64_t iterations, const std::vector<std::function<void()>> &ops) {
    std::latch start_gate(1);
    std::vector<std::thread> threads;
    for (const auto &op : ops) {
        threads.emplace_back([&start_gate, &op, iterations] {
            start_gate.wait();
            for (uint64_t i = 0; i < iterations; ++i) {
                op();
            }
        });
    }
    start_gate.count_down();
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST_F(MpExporterTest, ConcurrentAddXferStatsOverflowConservation) {
    constexpr uint64_t kThreads = 4;
    constexpr uint64_t kCallsPerThread = 5000;
    constexpr uint64_t kEventsPerCall = 4;

    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    const auto scrape = gtest::scrapeCoreOverflow(
        {.port = port_,
         .exporter = "prometheus_mp",
         .agent = "mp_concurrent_xfer_overflow_agent",
         .accepted_metric = "agent_tx_requests_num_total",
         .accepted_event_weight = kEventsPerCall,
         .expected_total_events = kThreads * kCallsPerThread * kEventsPerCall,
         .flush_interval = std::chrono::milliseconds(50),
         .settle_timeout = std::chrono::seconds(20)},
        [](nixlTelemetry &telemetry) {
            runConcurrently(
                kCallsPerThread, std::vector<std::function<void()>>(kThreads, [&telemetry] {
                    telemetry.addXferStats(
                        std::chrono::microseconds(10), true, 2000, std::chrono::microseconds(1));
                }));
        });

    ASSERT_TRUE(scrape.ok) << "accepted*4 + dropped must equal the produced events";
    EXPECT_GT(scrape.dropped, 0.0);
    EXPECT_EQ(std::fmod(scrape.dropped, static_cast<double>(kEventsPerCall)), 0.0);
}

TEST_F(MpExporterTest, ConcurrentMixedWorkloadAggregation) {
    constexpr uint64_t kIters = 500;
    constexpr uint64_t kTxA = 1000;
    constexpr uint64_t kTxB = 3000;
    constexpr uint64_t kRxA = 500;
    constexpr uint64_t kRxB = 1500;
    constexpr uint64_t kMem = 4096;
    const std::string agent_name = "mp_concurrent_mixed_agent";

    env_.addVar(TELEMETRY_BUFFER_SIZE_VAR, "65536");
    env_.addVar(TELEMETRY_RUN_INTERVAL_VAR, "5");
    const gtest::LogIgnoreGuard lig(PLUGIN_PROBE_WARNING);
    nixlTelemetry telemetry(agent_name, "prometheus_mp");
    runConcurrently(kIters,
                    {[&] { telemetry.updateTxBytes(kTxA); },
                     [&] { telemetry.updateTxBytes(kTxB); },
                     [&] { telemetry.updateRxBytes(kRxA); },
                     [&] { telemetry.updateRxBytes(kRxB); },
                     [&] { telemetry.updateTxRequestsNum(1); },
                     [&] { telemetry.updateMemoryRegistered(kMem); },
                     [&] { telemetry.updateErrorCount(NIXL_ERR_BACKEND); },
                     [&] { telemetry.updateErrorCount(NIXL_ERR_INVALID_PARAM); }});

    const labelSet agent{{"agent_name", agent_name}};
    const std::vector<std::tuple<std::string, labelSet, double>> totals{
        {"agent_tx_bytes_total", agent, kIters * (kTxA + kTxB)},
        {"agent_rx_bytes_total", agent, kIters * (kRxA + kRxB)},
        {"agent_tx_requests_num_total", agent, kIters},
        {"agent_memory_registered_total", agent, kIters * kMem},
        {"agent_errors_total", {{"agent_name", agent_name}, {"status", "backend"}}, kIters},
        {"agent_errors_total", {{"agent_name", agent_name}, {"status", "invalid_param"}}, kIters},
        {"agent_telemetry_events_dropped_total", agent, 0}};
    const auto metrics = nixl::metrics_test::scrapeUntil(
        port_, std::chrono::seconds(30), [&](const timeSeries &scrape) {
            return std::ranges::all_of(totals, [&](const auto &total) {
                return scrape.latestValue(std::get<0>(total), std::get<1>(total)) ==
                    std::get<2>(total);
            });
        });
    for (const auto &[name, labels, value] : totals) {
        EXPECT_EQ(metrics.latestValue(name, labels), value) << name;
    }

    EXPECT_THAT(metrics.latestValue("agent_tx_last_bytes", agent),
                AnyOf(Optional(kTxA), Optional(kTxB)));
    EXPECT_THAT(metrics.latestValue("agent_rx_last_bytes", agent),
                AnyOf(Optional(kRxA), Optional(kRxB)));
    EXPECT_THAT(metrics.latestValue("agent_memory_registered_last_bytes", agent), Optional(kMem));
}

} // namespace
