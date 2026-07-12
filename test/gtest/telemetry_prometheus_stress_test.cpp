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

// Concurrent Prometheus stress coverage. Kept separate from the functional
// Prometheus suite (telemetry_prometheus_test.cpp) but sharing its fixture via
// prometheus_telemetry_fixture.h and scraping through the same OpenMetrics
// parser/time-series helpers the DOCA and histogram tests use. The Prometheus
// exporter is lossless and ring-free, so concurrent staging overflow and mixed
// aggregation resolve to exact, hardware-independent conservation identities.

#include "prometheus_telemetry_fixture.h"

#include "common.h"
#include "telemetry.h"
#include "telemetry_event.h"

#include "loopback_connection.h"
#include "open_metrics_text_parser.h"
#include "timeseries.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using nixl::doca_test::labelSet;
using nixl::doca_test::loopbackConnection;
using nixl::doca_test::timeSeries;

namespace {

// Sanitizer builds run the same concurrent aggregation with fewer producer
// iterations so TSan/ASan stay within a CI-friendly runtime. Compile-time
// detection in test code only.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool kSanitizerBuild = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
constexpr bool kSanitizerBuild = true;
#else
constexpr bool kSanitizerBuild = false;
#endif
#else
constexpr bool kSanitizerBuild = false;
#endif

timeSeries
scrapeMetrics(uint16_t port) {
    return timeSeries(
        nixl::doca_test::open_metrics_text::parse(loopbackConnection::httpGet(port, "/metrics")));
}

struct OverflowScrape {
    bool ok = false; // all produced events were accounted for before the timeout
    double accepted = 0;
    double dropped = 0;
};

// Drives `produce` against a fresh nixlTelemetry backed by the Prometheus
// exporter with a small (256-slot) staging buffer, then polls /metrics until
// every produced event is accounted for: accepted (`accepted_metric`, weighted by
// `accepted_event_weight` events per sample) plus dropped
// (`agent_telemetry_events_dropped_total`) equals `expected_total_events`. Polling
// for that exact end state (not a fixed sleep) makes the result timing
// independent. The instance stays alive through the scrape so the exporter keeps
// serving the port.
OverflowScrape
scrapeCoreOverflow(uint16_t port,
                   const std::string &agent_name,
                   const std::string &accepted_metric,
                   uint64_t accepted_event_weight,
                   uint64_t expected_total_events,
                   const std::function<void(nixlTelemetry &)> &produce) {
    gtest::ScopedEnv telemetry_env;
    telemetry_env.addVar(TELEMETRY_BUFFER_SIZE_VAR, "256");
    telemetry_env.addVar(TELEMETRY_RUN_INTERVAL_VAR, "5");

    nixlTelemetry telemetry(agent_name, "prometheus");
    produce(telemetry);

    const labelSet labels{{"agent_name", agent_name}};
    OverflowScrape result;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kSanitizerBuild ? 20 : 5);
    do {
        const timeSeries metrics = scrapeMetrics(port);
        const auto dropped = metrics.latestValue("agent_telemetry_events_dropped_total", labels);
        const auto accepted = metrics.latestValue(accepted_metric, labels);
        if (dropped && accepted) {
            result.dropped = *dropped;
            result.accepted = *accepted;
            if (result.accepted * static_cast<double>(accepted_event_weight) + result.dropped ==
                static_cast<double>(expected_total_events)) {
                result.ok = true;
                return result;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    return result; // ok stays false: full accounting was not observed in time
}

} // namespace

// Concurrent staging overflow conservation.
//
// Many producer threads flood a small (256-slot) staging queue with all-or-none
// addXferStats batches. Driving the lossless, ring-free Prometheus exporter
// makes the outcome exact and hardware independent: every 4-event batch is
// either fully accepted (advancing agent_tx_requests_num_total by one per call,
// weight 4 slots) or fully dropped (four staging drops). Thread scheduling may
// change the accepted/dropped split but not the total, so
// accepted*4 + dropped must equal the produced event slots, drops must be
// nonzero, and drops must remain a multiple of four.
TEST_F(prometheusTelemetryTest, ConcurrentAddXferStatsOverflowConservation) {
    const std::string agent_name = "prometheus_concurrent_xfer_overflow_agent";
    constexpr uint64_t kThreads = kSanitizerBuild ? 4 : 8;
    constexpr uint64_t kCallsPerThread = kSanitizerBuild ? 5000 : 20000;
    constexpr uint64_t kEventsPerCall = 4;
    constexpr uint64_t kProducedEvents = kThreads * kCallsPerThread * kEventsPerCall;

    const auto scrape =
        scrapeCoreOverflow(port_,
                           agent_name,
                           "agent_tx_requests_num_total",
                           kEventsPerCall,
                           kProducedEvents,
                           [](nixlTelemetry &telemetry) {
                               std::vector<std::thread> producers;
                               producers.reserve(kThreads);
                               for (uint64_t t = 0; t < kThreads; ++t) {
                                   producers.emplace_back([&telemetry] {
                                       for (uint64_t i = 0; i < kCallsPerThread; ++i) {
                                           telemetry.addXferStats(std::chrono::microseconds(10),
                                                                  true,
                                                                  2000,
                                                                  std::chrono::microseconds(1));
                                       }
                                   });
                               }
                               for (auto &producer : producers) {
                                   producer.join();
                               }
                           });

    ASSERT_TRUE(scrape.ok) << "accepted*4 + dropped must reach produced events (" << kProducedEvents
                           << ") under concurrent overflow -- no silent loss";
    EXPECT_GT(scrape.dropped, 0.0)
        << "flooding a 256-slot staging queue from many threads must drop batches";
    EXPECT_EQ(std::fmod(scrape.dropped, static_cast<double>(kEventsPerCall)), 0.0)
        << "addXferStats drops the whole 4-event batch, so drops are multiples of 4";
}

// Concurrent Prometheus mixed-workload aggregation.
//
// Multiple producers drive distinct metric families concurrently into one
// Prometheus-backed instance whose staging queue is sized above the total, so
// nothing is dropped. Deterministic per-thread values make every cumulative
// counter's final total exact; the last-operation gauges must each hold one of
// the submitted values; error counters must stay on the bounded status-labeled
// series; and the staging-drop counter must remain zero (full conservation).
TEST_F(prometheusTelemetryTest, ConcurrentMixedWorkloadAggregation) {
    const std::string agent_name = "prometheus_concurrent_mixed_agent";
    const uint64_t iters = kSanitizerBuild ? 500 : 4000;

    gtest::ScopedEnv telemetry_env;
    telemetry_env.addVar(TELEMETRY_BUFFER_SIZE_VAR, "65536"); // above the total: no staging drops
    telemetry_env.addVar(TELEMETRY_RUN_INTERVAL_VAR, "5");

    nixlTelemetry telemetry(agent_name, "prometheus");

    constexpr uint64_t kTxA = 1000;
    constexpr uint64_t kTxB = 3000;
    constexpr uint64_t kRxA = 500;
    constexpr uint64_t kRxB = 1500;
    constexpr uint64_t kMem = 4096;

    std::vector<std::thread> producers;
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateTxBytes(kTxA);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateTxBytes(kTxB);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateRxBytes(kRxA);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateRxBytes(kRxB);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateTxRequestsNum(1);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateMemoryRegistered(kMem);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateErrorCount(NIXL_ERR_BACKEND);
        }
    });
    producers.emplace_back([&] {
        for (uint64_t i = 0; i < iters; ++i) {
            telemetry.updateErrorCount(NIXL_ERR_INVALID_PARAM);
        }
    });
    for (auto &producer : producers) {
        producer.join();
    }

    const double tx_expected = static_cast<double>(iters) * (kTxA + kTxB);
    const double rx_expected = static_cast<double>(iters) * (kRxA + kRxB);
    const double tx_requests_expected = static_cast<double>(iters);
    const double mem_expected = static_cast<double>(iters) * kMem;
    const double err_expected = static_cast<double>(iters);

    const labelSet labels{{"agent_name", agent_name}};

    // Poll until every deterministic cumulative counter has reached its exact
    // total; with no drops the periodic flush drains all events, so the end state
    // is reached regardless of how flushes interleave (timing independent).
    timeSeries metrics{nixl::doca_test::seriesMap{}};
    bool converged = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kSanitizerBuild ? 30 : 10);
    do {
        metrics = scrapeMetrics(port_);
        const auto tx = metrics.latestValue("agent_tx_bytes_total", labels);
        const auto rx = metrics.latestValue("agent_rx_bytes_total", labels);
        const auto tx_req = metrics.latestValue("agent_tx_requests_num_total", labels);
        const auto mem = metrics.latestValue("agent_memory_registered_total", labels);
        if (tx == tx_expected && rx == rx_expected && tx_req == tx_requests_expected &&
            mem == mem_expected) {
            converged = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);

    ASSERT_TRUE(converged) << "concurrent cumulative counters must reach their exact totals; "
                              "a lost event would leave a counter short of its deterministic sum";

    // Last-operation gauges must each hold one of that direction's submitted
    // values (the winner is scheduling dependent, the validity is not).
    const auto tx_last = metrics.latestValue("agent_tx_last_bytes", labels);
    ASSERT_TRUE(tx_last.has_value());
    EXPECT_TRUE(*tx_last == static_cast<double>(kTxA) || *tx_last == static_cast<double>(kTxB))
        << "tx last-op gauge must be a submitted tx value, got " << *tx_last;
    const auto rx_last = metrics.latestValue("agent_rx_last_bytes", labels);
    ASSERT_TRUE(rx_last.has_value());
    EXPECT_TRUE(*rx_last == static_cast<double>(kRxA) || *rx_last == static_cast<double>(kRxB))
        << "rx last-op gauge must be a submitted rx value, got " << *rx_last;
    EXPECT_EQ(metrics.latestValue("agent_memory_registered_last_bytes", labels),
              std::optional<double>(static_cast<double>(kMem)));

    // Error counters stay on the bounded status-labeled series, correctly mapped.
    EXPECT_EQ(metrics.latestValue("agent_errors_total",
                                  {{"agent_name", agent_name}, {"status", "backend"}}),
              std::optional<double>(err_expected));
    EXPECT_EQ(metrics.latestValue("agent_errors_total",
                                  {{"agent_name", agent_name}, {"status", "invalid_param"}}),
              std::optional<double>(err_expected));
    for (const auto &[id, series_samples] : metrics.series()) {
        (void)series_samples;
        EXPECT_NE(id.name.rfind("agent_err_", 0), 0u)
            << "legacy per-type error counter must not be published: " << id.name;
    }

    // Full conservation: with the queue sized above the total, nothing is
    // staging-dropped.
    EXPECT_EQ(metrics.latestValue("agent_telemetry_events_dropped_total", labels),
              std::optional<double>(0.0))
        << "no event may be dropped when the queue exceeds the total";
}
