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
#include "prometheus_telemetry_fixture.h"
#include "telemetry.h"
#include "telemetry/telemetry_exporter.h"
#include "telemetry_core_scrape.h"
#include "telemetry_event.h"

#include "common/scoped_fd.h"
#include "open_metrics_text_parser.h"
#include "scrape_util.h"
#include "timeseries.h"

#include <absl/log/log_sink_registry.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using nixl::metrics_test::labelSet;
using nixl::metrics_test::scrapeMetrics;
using nixl::metrics_test::scrapeUntilValue;
using nixl::metrics_test::timeSeries;
using nixl::scopedFd;

[[nodiscard]] labelSet
agentLabel(const std::string &agent_name) {
    return labelSet{{"agent_name", agent_name}};
}

// The labels of the single series named `name` carrying `agent_name`, so a test
// can assert on labels the lookup did not key on.
[[nodiscard]] std::optional<labelSet>
agentSeriesLabels(const timeSeries &metrics,
                  const std::string &name,
                  const std::string &agent_name) {
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        const auto agent = id.labels.find("agent_name");
        if (id.name == name && agent != id.labels.end() && agent->second == agent_name) {
            return id.labels;
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool
hasAnyAgentSeries(const timeSeries &metrics, const std::string &agent_name) {
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        const auto agent = id.labels.find("agent_name");
        if (id.name.rfind("agent_", 0) == 0 && agent != id.labels.end() &&
            agent->second == agent_name) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] scopedFd
occupyLocalPort(uint16_t port) {
    scopedFd fd(::socket(AF_INET, SOCK_STREAM, 0));
    if (!fd.valid()) {
        return {};
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    if (::bind(fd.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd.get(), 1) != 0) {
        return {};
    }
    return fd;
}

class SeverityCountingLogSink : public absl::LogSink {
public:
    SeverityCountingLogSink() {
        absl::AddLogSink(this);
    }

    ~SeverityCountingLogSink() override {
        absl::RemoveLogSink(this);
    }

    void
    Send(const absl::LogEntry &entry) override {
        const std::string msg(entry.text_message());
        if (entry.log_severity() == absl::LogSeverity::kWarning &&
            msg.find("could not be bound") != std::string::npos) {
            bindWarnings_.fetch_add(1, std::memory_order_relaxed);
        } else if (entry.log_severity() >= absl::LogSeverity::kWarning) {
            otherProblems_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::size_t
    bindWarnings() const {
        return bindWarnings_.load(std::memory_order_relaxed);
    }

    std::size_t
    otherProblems() const {
        return otherProblems_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::size_t> bindWarnings_{0};
    std::atomic<std::size_t> otherProblems_{0};
};

} // namespace

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

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    const labelSet agent = agentLabel(agent_name);

    // The counter families that initializeMetrics() must publish.
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
        EXPECT_TRUE(metrics.latestValue(c, agent).has_value())
            << "Missing counter family \"" << c << "\" in /metrics body";
    }
    EXPECT_TRUE(metrics
                    .latestValue("agent_errors_total",
                                 labelSet{{"agent_name", agent_name}, {"status", "invalid_param"}})
                    .has_value())
        << "Missing counter family \"agent_errors_total\" in /metrics body";

    // All last-operation gauges use the distinct "_last_bytes" series name, kept
    // separate from the cumulative "_total" counter of the same subject.
    EXPECT_TRUE(metrics.latestValue("agent_memory_registered_last_bytes", agent).has_value())
        << "Missing agent_memory_registered_last_bytes gauge";
    EXPECT_TRUE(metrics.latestValue("agent_memory_deregistered_last_bytes", agent).has_value())
        << "Missing agent_memory_deregistered_last_bytes gauge";
    EXPECT_TRUE(metrics.latestValue("agent_tx_last_bytes", agent).has_value())
        << "Missing agent_tx_last_bytes gauge";
    EXPECT_TRUE(metrics.latestValue("agent_rx_last_bytes", agent).has_value())
        << "Missing agent_rx_last_bytes gauge";
    EXPECT_FALSE(metrics.latestValue("agent_err_invalid_param_total", agent).has_value())
        << "Error counters must use the labeled agent_errors_total series";

    // Each metric must carry the two labels the exporter attaches.
    const auto labels = agentSeriesLabels(metrics, "agent_tx_bytes_total", agent_name);
    ASSERT_TRUE(labels.has_value()) << "agent_name label missing";
    EXPECT_FALSE(labels->at("hostname").empty());
    EXPECT_EQ(labels->count("category"), 0u);

    const std::string peer_agent_name = "prometheus_test_agent_peer";
    {
        const nixlTelemetryExporterInitParams peer_params{peer_agent_name, 4096};
        auto peer_exporter = handle->createExporter(peer_params);
        ASSERT_NE(peer_exporter, nullptr);

        const auto both_agents = scrapeMetrics(port_);
        ASSERT_FALSE(both_agents.empty()) << "Got empty /metrics response on port " << port_;
        EXPECT_TRUE(hasAnyAgentSeries(both_agents, agent_name))
            << "Missing metrics for first agent";
        EXPECT_TRUE(hasAnyAgentSeries(both_agents, peer_agent_name))
            << "Missing metrics for peer agent";
    }

    const auto after_peer_teardown = scrapeMetrics(port_);
    ASSERT_FALSE(after_peer_teardown.empty()) << "Got empty /metrics response on port " << port_;
    EXPECT_TRUE(hasAnyAgentSeries(after_peer_teardown, agent_name))
        << "First agent metrics were removed when peer exporter was destroyed";
    EXPECT_FALSE(hasAnyAgentSeries(after_peer_teardown, peer_agent_name))
        << "Peer agent metrics remained after peer exporter was destroyed";
}

TEST_F(prometheusTelemetryTest, ScrapeEmitsExactlyTheDescriptorSeries) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr) << "Failed to load prometheus telemetry plugin";

    const std::string agent_name = "prometheus_parity_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    std::set<std::string> expected;
    for (const auto event_type : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(event_type);
        if (descriptor.counterName != nullptr) {
            expected.insert(descriptor.counterName);
        }
        if (descriptor.gaugeName != nullptr) {
            expected.insert(descriptor.gaugeName);
        }
        if (descriptor.histogramName != nullptr) {
            const std::string base = descriptor.histogramName;
            expected.insert(base + "_bucket");
            expected.insert(base + "_sum");
            expected.insert(base + "_count");
        }
    }
    expected.insert("agent_errors_total");

    std::set<std::string> actual;
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        const auto agent = id.labels.find("agent_name");
        if (agent != id.labels.end() && agent->second == agent_name &&
            id.name.rfind("agent_", 0) == 0) {
            actual.insert(id.name);
        }
    }

    EXPECT_EQ(actual, expected)
        << "native Prometheus scrape must emit exactly the shared-descriptor series set";
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

    const std::string peer_agent_name = "prometheus_ub_test_agent_peer";
    const nixlTelemetryExporterInitParams peer_params{peer_agent_name, 4096};
    auto peer_exporter = handle->createExporter(peer_params);
    ASSERT_NE(peer_exporter, nullptr);

    // Five increments of 1000 bytes each → cumulative total must be 5000 in
    // the scrape body for AGENT_TX_BYTES. On buggy code, each Increment()
    // call dereferences a dangling Counter*; even if it returns without
    // crashing, the Family has no metric instance so the scrape below will
    // not contain "agent_tx_bytes_total{" at all.
    constexpr uint64_t kIncrement = 1000;
    constexpr int kEventCount = 5;
    for (int i = 0; i < kEventCount; ++i) {
        const nixlTelemetryEvent event{nixl_telemetry_event_type_t::AGENT_TX_BYTES, kIncrement};
        EXPECT_EQ(exporter->exportEvent(event), NIXL_SUCCESS);
    }

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    const auto total = metrics.latestValue("agent_tx_bytes_total", agentLabel(agent_name));
    ASSERT_TRUE(total.has_value())
        << "agent_tx_bytes_total for this agent is not in scrape body.\n"
        << "On buggy code, counters_ map holds a dangling Counter* and "
        << "Family::metrics_ is empty, so Family::Collect() returns {} and "
        << "TextSerializer emits nothing for this family.";
    EXPECT_EQ(*total, static_cast<double>(kIncrement * kEventCount))
        << "Counter value after " << kEventCount << " × Increment(" << kIncrement << ") should be "
        << (kIncrement * kEventCount);

    const auto labels = agentSeriesLabels(metrics, "agent_tx_bytes_total", agent_name);
    ASSERT_TRUE(labels.has_value());
    EXPECT_EQ(labels->count("category"), 0u);
    EXPECT_FALSE(labels->at("hostname").empty());

    EXPECT_TRUE(
        metrics.latestValue("agent_tx_bytes_total", agentLabel(peer_agent_name)).has_value())
        << "Missing metrics for peer agent before teardown";

    peer_exporter.reset();

    const auto after_peer_teardown = scrapeMetrics(port_);
    ASSERT_FALSE(after_peer_teardown.empty()) << "Got empty /metrics response on port " << port_;
    EXPECT_EQ(after_peer_teardown.latestValue("agent_tx_bytes_total", agentLabel(agent_name)),
              std::optional<double>(static_cast<double>(kIncrement * kEventCount)))
        << "First agent metrics were removed when peer exporter was destroyed";
    EXPECT_FALSE(hasAnyAgentSeries(after_peer_teardown, peer_agent_name))
        << "Peer agent metrics remained after peer exporter was destroyed";
}

// A byte event drives BOTH a cumulative "_total" counter and a last-operation
// "_last" gauge from the same per-op value. TX and RX are exercised with
// distinct values so the assertions also prove the two byte directions map to
// independent series (no cross-wiring): the counter must read the sum of its
// deltas while the gauge must read only the final op, not a running total.
TEST_F(prometheusTelemetryTest, ByteCounterSumsWhileLastGaugeTracksFinalOp) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_last_gauge_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    constexpr std::array<uint64_t, 3> tx_values{1000, 2000, 3500}; // sum 6500, last 3500
    for (const uint64_t v : tx_values) {
        const nixlTelemetryEvent event{nixl_telemetry_event_type_t::AGENT_TX_BYTES, v};
        EXPECT_EQ(exporter->exportEvent(event), NIXL_SUCCESS);
    }
    constexpr std::array<uint64_t, 2> rx_values{500, 1500}; // sum 2000, last 1500
    for (const uint64_t v : rx_values) {
        const nixlTelemetryEvent event{nixl_telemetry_event_type_t::AGENT_RX_BYTES, v};
        EXPECT_EQ(exporter->exportEvent(event), NIXL_SUCCESS);
    }

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    const labelSet agent = agentLabel(agent_name);

    EXPECT_EQ(metrics.latestValue("agent_tx_bytes_total", agent), std::optional<double>(6500.0))
        << "tx counter must sum every exported delta (1000+2000+3500)";
    EXPECT_EQ(metrics.latestValue("agent_tx_last_bytes", agent), std::optional<double>(3500.0))
        << "tx last-op gauge must equal the final exported value (3500), not the sum";
    EXPECT_EQ(metrics.latestValue("agent_rx_bytes_total", agent), std::optional<double>(2000.0))
        << "rx counter must sum every exported delta (500+1500)";
    EXPECT_EQ(metrics.latestValue("agent_rx_last_bytes", agent), std::optional<double>(1500.0))
        << "rx last-op gauge must equal the final exported value (1500), not the sum";
}

TEST_F(prometheusTelemetryTest, ErrorCountersUseBoundedStatusLabel) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_error_counter_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
    ASSERT_NE(exporter, nullptr);

    EXPECT_EQ(exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1}),
              NIXL_SUCCESS);
    EXPECT_EQ(exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1}),
              NIXL_SUCCESS);
    EXPECT_EQ(exporter->exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_BACKEND, 1}),
              NIXL_SUCCESS);

    const auto metrics = scrapeMetrics(port_);
    ASSERT_FALSE(metrics.empty()) << "Got empty /metrics response on port " << port_;

    EXPECT_EQ(
        metrics.latestValue("agent_errors_total",
                            labelSet{{"agent_name", agent_name}, {"status", "invalid_param"}}),
        std::optional<double>(2.0))
        << "agent_errors_total{status=\"invalid_param\"} for this agent is not in scrape body";
    EXPECT_EQ(metrics.latestValue("agent_errors_total",
                                  labelSet{{"agent_name", agent_name}, {"status", "backend"}}),
              std::optional<double>(1.0))
        << "agent_errors_total{status=\"backend\"} for this agent is not in scrape body";

    const auto labels = agentSeriesLabels(metrics, "agent_errors_total", agent_name);
    ASSERT_TRUE(labels.has_value());
    EXPECT_FALSE(labels->at("hostname").empty());

    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        EXPECT_NE(id.name.rfind("agent_err_", 0), 0u)
            << "legacy per-type error counter must not be published: " << id.name;
    }
}

// The synthetic AGENT_TELEMETRY_EVENTS_DROPPED event (emitted by the core on flush with
// the number of staging-queue drops since the last flush) must surface as the
// cumulative counter agent_telemetry_events_dropped_total, accumulating every delta.
TEST_F(prometheusTelemetryTest, DroppedEventsCounterAccumulates) {
    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const std::string agent_name = "prometheus_dropped_events_agent";
    const nixlTelemetryExporterInitParams params{agent_name, 4096};
    auto exporter = handle->createExporter(params);
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

    EXPECT_EQ(metrics.latestValue("agent_telemetry_events_dropped_total", agentLabel(agent_name)),
              std::optional<double>(static_cast<double>(expected_total)))
        << "dropped-events counter must sum every emitted delta (7+5)";
}

// End-to-end through the core: a per-event allowlist skips deactivated metrics at
// the source, so an enabled metric advances while a disabled one stays at its
// pre-registered 0. Families are always registered, so this asserts values, not
// series presence (event-type granularity, not per-series).
TEST_F(prometheusTelemetryTest, MetricAllowlistDeactivatesMetric) {
    gtest::ScopedEnv telemetry_env;
    telemetry_env.addVar(TELEMETRY_ENABLED_METRICS_VAR, "agent_tx_bytes");
    telemetry_env.addVar(TELEMETRY_RUN_INTERVAL_VAR, "1");

    const std::string agent_name = "prometheus_allowlist_agent";
    nixlTelemetry telemetry(agent_name, "prometheus");
    telemetry.updateTxBytes(1000); // allowed
    telemetry.updateRxBytes(2000); // filtered

    const labelSet agent = agentLabel(agent_name);
    const auto metrics =
        scrapeUntilValue(port_, "agent_tx_bytes_total", 1000.0, std::chrono::seconds(5), agent);

    ASSERT_EQ(metrics.latestValue("agent_tx_bytes_total", agent), std::optional<double>(1000.0))
        << "allowed metric agent_tx_bytes_total never reached 1000";
    EXPECT_EQ(metrics.latestValue("agent_rx_bytes_total", agent), std::optional<double>(0.0))
        << "filtered metric must not be exported";
}

// End-to-end through the core: flooding a small staging queue via updateData
// forces producer-side drops, which the core publishes as AGENT_TELEMETRY_EVENTS_DROPPED
// on flush. Driving the (lossless, ring-free) Prometheus exporter makes the
// result exact and hardware-independent: every produced event is either counted
// (agent_tx_requests_num_total) or dropped (agent_telemetry_events_dropped_total), so
// their sum must equal the number produced regardless of flush timing.
TEST_F(prometheusTelemetryTest, CoreUpdateDataOverflowConservation) {
    const std::string agent_name = "prometheus_core_update_overflow_agent";
    constexpr uint64_t kProduced = 100000; // far exceeds the 256-slot staging queue

    // Each accepted event adds 1 to agent_tx_requests_num_total (weight 1), so
    // ok == (accepted + dropped == produced): conservation with no silent loss.
    const auto scrape = gtest::scrapeCoreOverflow({.port = port_,
                                                   .exporter = "prometheus",
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

// Same conservation check for the all-or-none addXferStats batch path: each
// accepted call stages 4 events (weight 4) and each dropped call loses its whole
// 4-event batch, so the dropped counter is always a multiple of 4 and
// accepted*4 + dropped must equal the produced events.
TEST_F(prometheusTelemetryTest, CoreAddXferStatsOverflowConservation) {
    const std::string agent_name = "prometheus_core_xfer_overflow_agent";
    constexpr uint64_t kCalls = 100000;
    constexpr uint64_t kEventsPerCall = 4;

    const auto scrape = gtest::scrapeCoreOverflow(
        {.port = port_,
         .exporter = "prometheus",
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

TEST_F(prometheusTelemetryTest, BindCollisionThrowsBindFailed) {
    const scopedFd occupier = occupyLocalPort(port_);
    ASSERT_TRUE(occupier.valid()) << "could not occupy 127.0.0.1:" << port_;

    auto handle = nixlPluginManager::getInstance().loadTelemetryPlugin("prometheus");
    ASSERT_NE(handle, nullptr);

    const nixlTelemetryExporterInitParams params{"prometheus_bind_collision_agent", 4096};
    EXPECT_THROW(handle->createExporter(params), nixlTelemetryBindFailed);
}

TEST_F(prometheusTelemetryTest, BindCollisionCreateIsNonFatalWarn) {
    const scopedFd occupier = occupyLocalPort(port_);
    ASSERT_TRUE(occupier.valid()) << "could not occupy 127.0.0.1:" << port_;

    gtest::ScopedEnv exporter_env;
    exporter_env.addVar(telemetryExporterVar, "prometheus");

    gtest::LogIgnoreGuard ignore_bind_warning("could not be bound");
    SeverityCountingLogSink sink;

    std::unique_ptr<nixlTelemetry> telemetry;
    EXPECT_NO_THROW(telemetry = nixlTelemetry::create("prometheus_bind_collision_create_agent"));
    EXPECT_EQ(telemetry, nullptr)
        << "a scrape-port collision must disable telemetry, not fail agent construction";

    EXPECT_EQ(sink.bindWarnings(), std::size_t{1}) << "the collision must log exactly one WARNING";
    EXPECT_EQ(sink.otherProblems(), std::size_t{0})
        << "the collision must not log an ERROR or other problem";
    EXPECT_EQ(ignore_bind_warning.getIgnoredCount(), 1u);
}
