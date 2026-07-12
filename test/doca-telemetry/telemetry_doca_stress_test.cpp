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

// DOCA exporter volume within the standalone test architecture.
//
// test/doca-telemetry/doca_nixl_test intentionally does not link full NIXL core,
// so this exercises the NIXL DOCA exporter's volume handling only -- not the core
// staging queue, whose concurrency is validated through BUFFER and Prometheus in
// the gtest stress suite. Kept in its own source/binary so the sustained
// high-volume case stays isolated from the functional DOCA exporter tests.

#include "scrape_util.h"

#include "doca_exporter.h"
#include "telemetry_event.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gtest/gtest.h>

using nixl::doca_test::loopbackConnection;
using nixl::doca_test::scrapeUntilValue;

namespace {

constexpr char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
constexpr char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

} // namespace

class docaNixlExporterStressTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        port_ = loopbackConnection::findFreePort();
        ASSERT_NE(port_, 0) << "failed to allocate a free TCP port";

        ASSERT_EQ(::setenv(docaPrometheusLocalVar, "y", 1), 0);
        ASSERT_EQ(::setenv(docaPrometheusPortVar, std::to_string(port_).c_str(), 1), 0);
    }

    void
    TearDown() override {
        ::unsetenv(docaPrometheusLocalVar);
        ::unsetenv(docaPrometheusPortVar);
    }

    uint16_t port_ = 0;
};

// Push a large deterministic sequence directly through nixlTelemetryDocaExporter.
// Distinct increasing byte deltas make the cumulative counter sum and the
// last-operation gauge sensitive to any dropped or duplicated sample, and a long
// run of synthetic drop deltas exercises the drop counter's accumulation at
// volume. One flush after production, then one settle-and-scrape on the stable
// final totals.
TEST_F(docaNixlExporterStressTest, HighVolumeSequenceCounterGaugeAndDropAccumulate) {
    constexpr char agentName[] = "nixl_doca_volume_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const nixl::doca_test::labelSet labels{{"agent_name", agentName}};
    const std::string txCounter =
        nixlEnumStrings::telemetryMetricDescriptor(nixl_telemetry_event_type_t::AGENT_TX_BYTES)
            .counterName;
    const std::string txLast = "agent_tx_last_bytes";
    const std::string dropCounter = nixlEnumStrings::telemetryMetricDescriptor(
                                        nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED)
                                        .counterName;

    constexpr uint64_t kByteEvents = 4000;
    uint64_t counter_sum = 0;
    for (uint64_t i = 1; i <= kByteEvents; ++i) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TX_BYTES, i);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
        counter_sum += i;
    }
    const uint64_t last_byte_value = kByteEvents;

    // A long run of per-flush drop deltas (as the core would emit them under
    // sustained staging overflow), each a whole 4-event batch.
    constexpr uint64_t kDropDeltas = 1000;
    constexpr uint64_t kDropPerDelta = 4;
    uint64_t drop_total = 0;
    for (uint64_t i = 0; i < kDropDeltas; ++i) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TELEMETRY_EVENTS_DROPPED,
                                       kDropPerDelta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
        drop_total += kDropPerDelta;
    }

    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const auto metrics = scrapeUntilValue(
        port_, txCounter, static_cast<double>(counter_sum), std::chrono::seconds(15), labels);
    const std::optional<double> observed_counter = metrics.latestValue(txCounter, labels);
    ASSERT_TRUE(observed_counter.has_value())
        << txCounter << "{agent_name=" << agentName << "} not served after flush";
    EXPECT_EQ(*observed_counter, static_cast<double>(counter_sum))
        << "cumulative counter must equal the sum of all " << kByteEvents
        << " distinct pushed deltas; a drop or duplicate would miss this total";

    EXPECT_EQ(metrics.latestValue(txLast, labels), std::optional<double>(last_byte_value))
        << "last-op gauge must reflect the final pushed value at volume";

    EXPECT_EQ(metrics.latestValue(dropCounter, labels), std::optional<double>(drop_total))
        << "drop counter must accumulate every emitted delta at volume (" << kDropDeltas << "*"
        << kDropPerDelta << ")";
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
