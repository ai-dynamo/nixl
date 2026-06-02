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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

using nixl::doca_test::findFreePort;
using nixl::doca_test::metricValue;
using nixl::doca_test::scrapeUntil;

namespace {

constexpr char docaPrometheusPortVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_PORT";
constexpr char docaPrometheusLocalVar[] = "NIXL_TELEMETRY_DOCA_PROMETHEUS_LOCAL";

} // namespace

class docaNixlExporterTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        port_ = findFreePort();
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

// Drive the real NIXL DOCA exporter end-to-end: push per-operation counter
// deltas through exportEvent, flush, then scrape the CollectX-backed Prometheus
// endpoint. A counter event (AGENT_TX_BYTES) must accumulate into a monotonic
// cumulative total (add_counter_increment), and flush must make the few samples
// visible (they would not fill the buffer to trigger an auto-flush).
TEST_F(docaNixlExporterTest, CounterAccumulatesAndFlushServes) {
    const nixlTelemetryExporterInitParams params{"nixl_doca_exporter_test", 4096};
    nixlTelemetryDocaExporter exporter(params);

    constexpr uint64_t delta = 1000;
    constexpr int iterations = 3;
    for (int i = 0; i < iterations; ++i) {
        const nixlTelemetryEvent event(nixl_telemetry_event_type_t::AGENT_TX_BYTES, delta);
        ASSERT_EQ(exporter.exportEvent(event), NIXL_SUCCESS);
    }

    // A handful of samples will not fill the CollectX buffer, so the endpoint
    // would stay empty without this explicit flush.
    ASSERT_EQ(exporter.flush(), NIXL_SUCCESS);

    const std::string metric = std::string(
        nixlEnumStrings::telemetryEventTypeStr(nixl_telemetry_event_type_t::AGENT_TX_BYTES));
    const std::string body = scrapeUntil(port_, metric, std::chrono::seconds(12));
    std::cout << "=== NIXL DOCA exporter /metrics scrape (port " << port_ << ") ===\n"
              << body << "\n=== end scrape ===" << std::endl;

    EXPECT_NE(body.find(metric), std::string::npos)
        << metric << " not served at the DOCA Prometheus endpoint after flush";
    EXPECT_EQ(metricValue(body, metric), static_cast<double>(delta * iterations))
        << "per-operation deltas must accumulate into a cumulative counter";
}

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
