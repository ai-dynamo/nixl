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

#include "scrape_util.h"

#include "doca_exporter.h"
#include "telemetry_event.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

using nixl::doca_test::loopbackConnection;
using nixl::doca_test::scrapeUntilValue;

namespace {

constexpr char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
constexpr char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

} // namespace

class docaNixlExporterTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        port_ = loopbackConnection::findFreePort();
        ASSERT_NE(port_, 0) << "failed to allocate a free TCP port";

        // The exporter reads these on construction (getBindAddress); bind to
        // loopback on the just-allocated port so the test is self-contained.
        ASSERT_EQ(::setenv(docaPrometheusLocalVar, "y", 1), 0);
        ASSERT_EQ(::setenv(docaPrometheusPortVar, std::to_string(port_).c_str(), 1), 0);
    }

    void
    TearDown() override {
        // Pair with SetUp so the fixture does not leak process-wide env state.
        ::unsetenv(docaPrometheusLocalVar);
        ::unsetenv(docaPrometheusPortVar);
    }

    uint16_t port_ = 0;
};

// Drive the real NIXL DOCA exporter end-to-end and verify EVERY pushed value is
// accounted for at the DOCA Prometheus endpoint. The exported counter
// is cumulative (add_counter_increment), so the final total must equal the exact
// sum of all pushed deltas: any dropped or duplicated sample shifts that sum and
// fails the check. Deltas are deliberately distinct so the sum is sensitive to a
// single bad sample. One flush + one settle-and-scrape -- no per-iteration
// rescraping (the final total is the stable invariant; intermediate cumulative
// values aren't reliably observable through DOCA's async/coalescing flush).
TEST_F(docaNixlExporterTest, CounterAccumulatesAllPushedValues) {
    constexpr char agentName[] = "nixl_doca_exporter_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const std::string metric =
        nixlEnumStrings::telemetryMetricDescriptor(nixl_telemetry_event_type_t::AGENT_TX_BYTES)
            .counterName;
    // The exporter tags every sample with the agent_name label, so look the
    // series up by name + that label rather than by name alone.
    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};

    constexpr std::array<uint64_t, 5> deltas{1000, 250, 4096, 1, 75000};
    uint64_t expected_total = 0;
    for (const uint64_t delta : deltas) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TX_BYTES, delta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
        expected_total += delta;
    }
    // A handful of samples never fill the DOCA metrics buffer, so the endpoint
    // would stay empty without an explicit flush.
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const auto metrics = scrapeUntilValue(
        port_, metric, static_cast<double>(expected_total), std::chrono::seconds(12), labels);
    const std::optional<double> observed = metrics.latestValue(metric, labels);
    ASSERT_TRUE(observed.has_value())
        << metric << "{agent_name=" << agentName << "} not served at the endpoint after flush";
    EXPECT_EQ(*observed, static_cast<double>(expected_total))
        << "cumulative counter must equal the sum of all " << deltas.size() << " pushed deltas ("
        << expected_total << "); a drop or duplicate would miss this total";
}

// A gauge is absolute (add_gauge), unlike the cumulative counter above: the
// endpoint must reflect the LAST pushed value, not the sum. Push a sequence of
// distinct values and confirm only the final one is served -- if the exporter
// wrongly accumulated, the served value would be the sum and this would fail.
TEST_F(docaNixlExporterTest, GaugeReflectsLastPushedValue) {
    constexpr char agentName[] = "nixl_doca_exporter_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    // The memory gauge is served under its last-operation series name, which is
    // distinct from the AGENT_MEMORY_REGISTERED event string.
    const std::string metric = "agent_memory_registered_last_bytes";
    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};

    constexpr std::array<uint64_t, 4> values{4096, 65536, 1024, 8192};
    for (const uint64_t v : values) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_MEMORY_REGISTERED, v);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
    }
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const uint64_t expected = values.back();
    const auto metrics = scrapeUntilValue(
        port_, metric, static_cast<double>(expected), std::chrono::seconds(12), labels);
    const std::optional<double> observed = metrics.latestValue(metric, labels);
    ASSERT_TRUE(observed.has_value())
        << metric << "{agent_name=" << agentName << "} not served at the endpoint after flush";
    EXPECT_EQ(*observed, static_cast<double>(expected))
        << "a gauge must reflect the last pushed value (" << expected << "), not a sum";
}

// Two agents export the SAME metric name; the DOCA exporter tags each sample
// with its agent_name, producing two distinct series that differ only by label.
// A single scrape must keep them apart: a name + agent_name lookup returns that
// agent's value, while a name-only lookup is ambiguous. Both exporters share the
// process-wide DOCA context (one Prometheus endpoint), as in a real multi-agent
// process.
TEST_F(docaNixlExporterTest, DistinguishesSeriesByAgentLabel) {
    constexpr char agentAlpha[] = "agent_alpha";
    constexpr char agentBeta[] = "agent_beta";
    constexpr auto txBytes = nixl_telemetry_event_type_t::AGENT_TX_BYTES;

    nixlTelemetryDocaExporter exporterAlpha(nixlTelemetryExporterInitParams{agentAlpha, 4096});
    nixlTelemetryDocaExporter exporterBeta(nixlTelemetryExporterInitParams{agentBeta, 4096});

    ASSERT_EQ(exporterAlpha.exportEvent(nixlTelemetryEvent(txBytes, 700)), NIXL_SUCCESS);
    ASSERT_EQ(exporterBeta.exportEvent(nixlTelemetryEvent(txBytes, 4)), NIXL_SUCCESS);
    // Both exporters share the one process-wide DOCA context/source, so a single
    // flush on either drains both agents' buffered samples.
    ASSERT_EQ(exporterAlpha.flush(), NIXL_SUCCESS);

    const std::string metric = nixlEnumStrings::telemetryMetricDescriptor(txBytes).counterName;

    // Wait for alpha's series, then read both from that one parsed scrape.
    const auto metrics = scrapeUntilValue(
        port_, metric, 700.0, std::chrono::seconds(12), {{"agent_name", agentAlpha}});
    EXPECT_EQ(metrics.latestValue(metric, {{"agent_name", agentAlpha}}),
              std::optional<double>(700.0));
    EXPECT_EQ(metrics.latestValue(metric, {{"agent_name", agentBeta}}), std::optional<double>(4.0));
    // Same name, two label sets -> a name-only lookup is ambiguous and is
    // rejected with a throw (a too-loose query is a test mistake, not a miss).
    EXPECT_THROW((void)metrics.latestValue(metric), std::runtime_error)
        << metric << " is exported by two agents; a name-only lookup must be ambiguous";
}

// Both byte directions drive a cumulative counter (agent_{tx,rx}_bytes) AND a
// last-operation gauge (agent_{tx,rx}_last_bytes) from the same per-op value. TX
// and RX use distinct deltas so the assertions also prove the directions map to
// independent series (no cross-wiring): each counter reads the sum of its own
// deltas while each *_last gauge reads only that direction's final op.
TEST_F(docaNixlExporterTest, ByteEventsEmitCumulativeCountersAndLastGauges) {
    constexpr char agentName[] = "nixl_doca_last_gauge_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};
    const std::string txCounter =
        nixlEnumStrings::telemetryMetricDescriptor(nixl_telemetry_event_type_t::AGENT_TX_BYTES)
            .counterName;
    const std::string rxCounter =
        nixlEnumStrings::telemetryMetricDescriptor(nixl_telemetry_event_type_t::AGENT_RX_BYTES)
            .counterName;
    const std::string txLast = "agent_tx_last_bytes";
    const std::string rxLast = "agent_rx_last_bytes";

    constexpr std::array<uint64_t, 3> tx_deltas{10, 20, 35}; // sum 65, last 35
    for (const uint64_t delta : tx_deltas) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TX_BYTES, delta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
    }
    constexpr std::array<uint64_t, 2> rx_deltas{5, 15}; // sum 20, last 15
    for (const uint64_t delta : rx_deltas) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_RX_BYTES, delta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
    }
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    // RX is pushed last and each event appends its gauge sample before its
    // counter sample, so the RX counter is the final sample; once agent_rx_bytes
    // reads its total (20) every earlier sample (all TX, plus the RX gauge) has
    // been served too -- read them all from that one settled scrape.
    const auto metrics = scrapeUntilValue(port_, rxCounter, 20.0, std::chrono::seconds(12), labels);

    EXPECT_EQ(metrics.latestValue(txCounter, labels), std::optional<double>(65.0))
        << "tx counter must sum every pushed delta (10+20+35)";
    EXPECT_EQ(metrics.latestValue(txLast, labels), std::optional<double>(35.0))
        << "tx last-op gauge must reflect only the final pushed value (35), not a sum";
    EXPECT_EQ(metrics.latestValue(rxCounter, labels), std::optional<double>(20.0))
        << "rx counter must sum every pushed delta (5+15)";
    EXPECT_EQ(metrics.latestValue(rxLast, labels), std::optional<double>(15.0))
        << "rx last-op gauge must reflect only the final pushed value (15), not a sum";
}

TEST_F(docaNixlExporterTest, ErrorCountersUseBoundedStatusLabel) {
    constexpr char agentName[] = "nixl_doca_error_counter_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    ASSERT_EQ(exporter.exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1}),
              NIXL_SUCCESS);
    ASSERT_EQ(exporter.exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1}),
              NIXL_SUCCESS);
    ASSERT_EQ(exporter.exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_BACKEND, 1}),
              NIXL_SUCCESS);
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const std::string metric = "agent_errors_total";
    const nixl::doca_test::labelSet invalidParamLabels{{"agent_name", agentName},
                                                       {"status", "invalid_param"}};
    const nixl::doca_test::labelSet backendLabels{{"agent_name", agentName}, {"status", "backend"}};

    const auto metrics =
        scrapeUntilValue(port_, metric, 2.0, std::chrono::seconds(12), invalidParamLabels);
    EXPECT_EQ(metrics.latestValue(metric, invalidParamLabels), std::optional<double>(2.0))
        << "invalid_param errors must accumulate under a bounded status label";
    EXPECT_EQ(metrics.latestValue(metric, backendLabels), std::optional<double>(1.0))
        << "backend errors must use a distinct status label on the same metric";

    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        EXPECT_NE(id.name.rfind("agent_err_", 0), 0)
            << "legacy per-type error counter must not be published: " << id.name;
    }
}

TEST_F(docaNixlExporterTest, ScrapeEmitsExactlyTheDescriptorSeries) {
    constexpr char agentName[] = "nixl_doca_parity_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    std::set<std::string> expected;
    for (const auto eventType : telemetry_metric_event_types) {
        const auto descriptor = nixlEnumStrings::telemetryMetricDescriptor(eventType);
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
        ASSERT_EQ(exporter.exportEvent(nixlTelemetryEvent(eventType, 7)), NIXL_SUCCESS);
    }
    expected.insert("agent_errors_total");
    ASSERT_EQ(exporter.exportEvent({nixl_telemetry_event_type_t::AGENT_ERR_INVALID_PARAM, 1}),
              NIXL_SUCCESS);
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    // Anchor on a histogram series (flushed with everything else in one metrics
    // flush): once it is served, every other descriptor series is present too.
    const nixl::doca_test::labelSet histogramLabels{{"agent_name", agentName}};
    const auto metrics = scrapeUntilValue(
        port_, "agent_xfer_time_us_count", 1.0, std::chrono::seconds(12), histogramLabels);

    std::set<std::string> actual;
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        const auto agent = id.labels.find("agent_name");
        if (agent != id.labels.end() && agent->second == agentName &&
            id.name.rfind("agent_", 0) == 0) {
            actual.insert(id.name);
        }
    }

    EXPECT_EQ(actual, expected) << "DOCA scrape must emit exactly the shared-descriptor series set";
}

// The synthetic AGENT_TELEMETRY_EVENTS_DROPPED event flows through the standard counter
// path (add_counter_increment), so the DOCA endpoint must expose it as the
// cumulative series agent_telemetry_events_dropped_total whose value equals the sum
// of the emitted per-flush drop deltas.
TEST_F(docaNixlExporterTest, DroppedEventsCounterAccumulates) {
    constexpr char agentName[] = "nixl_doca_dropped_events_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const std::string metric = nixlEnumStrings::telemetryMetricDescriptor(
                                   nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED)
                                   .counterName;
    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};

    constexpr std::array<uint64_t, 3> deltas{7, 4, 13};
    uint64_t expected_total = 0;
    for (const uint64_t delta : deltas) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED,
                                       delta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
        expected_total += delta;
    }
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const auto metrics = scrapeUntilValue(
        port_, metric, static_cast<double>(expected_total), std::chrono::seconds(12), labels);
    const std::optional<double> observed = metrics.latestValue(metric, labels);
    ASSERT_TRUE(observed.has_value())
        << metric << "{agent_name=" << agentName << "} not served at the endpoint after flush";
    EXPECT_EQ(*observed, static_cast<double>(expected_total))
        << "dropped-events counter must equal the sum of all emitted deltas (7+4+13)";
}

// AGENT_XFER_TIME events drive the agent_xfer_time_us DOCA histogram. After
// flush (which snapshots the cached histogram into the export buffer before the
// general metrics flush transmits it), the endpoint must serve _count = number
// of observations, _sum = their total, and cumulative _bucket{le} counts. Values
// land in distinct default buckets so the per-bucket counts are unambiguous.
TEST_F(docaNixlExporterTest, XferTimeHistogramRecordsObservations) {
    constexpr char agentName[] = "nixl_doca_histogram_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};

    constexpr std::array<uint64_t, 4> values{5, 30, 200, 700}; // sum 935
    for (const uint64_t v : values) {
        ASSERT_EQ(exporter.exportEvent(
                      nixlTelemetryEvent(nixl_telemetry_event_type_t::AGENT_XFER_TIME, v)),
                  NIXL_SUCCESS);
    }
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const auto metrics =
        scrapeUntilValue(port_, "agent_xfer_time_us_count", 4.0, std::chrono::seconds(12), labels);

    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_count", labels), std::optional<double>(4.0))
        << "histogram _count must equal the number of observations";
    EXPECT_EQ(metrics.latestValue("agent_xfer_time_us_sum", labels), std::optional<double>(935.0))
        << "histogram _sum must equal the sum of observations";

    // Build le(double) -> cumulative count from the bucket series for this agent
    // so the assertions do not depend on the exact `le` string formatting.
    std::map<double, double> buckets;
    for (const auto &[id, samples] : metrics.series()) {
        if (id.name != "agent_xfer_time_us_bucket") {
            continue;
        }
        const auto agent = id.labels.find("agent_name");
        const auto le = id.labels.find("le");
        if (agent == id.labels.end() || agent->second != agentName || le == id.labels.end() ||
            samples.empty()) {
            continue;
        }
        try {
            size_t pos = 0;
            const double boundary = std::stod(le->second, &pos);
            if (pos == le->second.size()) {
                buckets[boundary] = samples.back().value;
            }
        }
        catch (const std::exception &) {
        }
    }

    // 5<=10 (1); +30<=100 (2); +200<=250 (3); +700<=1000 (4).
    EXPECT_EQ(buckets[10.0], 1.0);
    EXPECT_EQ(buckets[100.0], 2.0);
    EXPECT_EQ(buckets[250.0], 3.0);
    EXPECT_EQ(buckets[1000.0], 4.0);
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
