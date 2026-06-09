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
#include <optional>
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
// accounted for at the CollectX-backed Prometheus endpoint. The exported counter
// is cumulative (add_counter_increment), so the final total must equal the exact
// sum of all pushed deltas: any dropped or duplicated sample shifts that sum and
// fails the check. Deltas are deliberately distinct so the sum is sensitive to a
// single bad sample. One flush + one settle-and-scrape -- no per-iteration
// rescraping (the final total is the stable invariant; intermediate cumulative
// values aren't reliably observable through CollectX's async/coalescing flush).
TEST_F(docaNixlExporterTest, CounterAccumulatesAllPushedValues) {
    constexpr char agentName[] = "nixl_doca_exporter_test";
    const nixlTelemetryExporterInitParams params{agentName, 4096};
    nixlTelemetryDocaExporter exporter(params);

    const std::string metric = std::string(
        nixlEnumStrings::telemetryEventTypeStr(nixl_telemetry_event_type_t::AGENT_TX_BYTES));
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
    // A handful of samples never fill the CollectX buffer, so the endpoint would
    // stay empty without an explicit flush.
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

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
