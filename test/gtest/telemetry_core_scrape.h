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
#ifndef NIXL_TEST_GTEST_TELEMETRY_CORE_SCRAPE_H
#define NIXL_TEST_GTEST_TELEMETRY_CORE_SCRAPE_H

#include "common.h"
#include "telemetry.h"
#include "telemetry_event.h"

#include "scrape_util.h"
#include "timeseries.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace gtest {

struct overflowScrape {
    bool ok = false; // all produced events were accounted for before the timeout
    double accepted = 0;
    double dropped = 0;
};

struct coreOverflowSpec {
    uint16_t port = 0;
    std::string exporter;
    std::string agent;
    std::string accepted_metric;
    uint64_t accepted_event_weight = 1;
    uint64_t expected_total_events = 0;
    std::chrono::milliseconds flush_interval = std::chrono::milliseconds(5);
    std::chrono::seconds settle_timeout = std::chrono::seconds(5);
};

// Drives `produce` against a fresh nixlTelemetry backed by the `exporter`
// exporter with a small (256-slot) staging buffer, then polls /metrics until every
// produced event is accounted for -- accepted (`accepted_metric`, weighted by
// `accepted_event_weight` events per sample) plus dropped
// (`agent_telemetry_events_dropped_total`) equals `expected_total_events`. Polling for that
// exact end state (rather than a fixed sleep) is what makes the test timing
// independent: it waits for the staging queue to fully drain and the final drop
// delta to be published, no matter how flushes interleave. The instance stays
// alive through the scrape so the exporter keeps serving the port.
[[nodiscard]] inline overflowScrape
scrapeCoreOverflow(const coreOverflowSpec &spec,
                   const std::function<void(nixlTelemetry &)> &produce) {
    ScopedEnv telemetry_env;
    telemetry_env.addVar(TELEMETRY_BUFFER_SIZE_VAR, "256");
    telemetry_env.addVar(TELEMETRY_RUN_INTERVAL_VAR, std::to_string(spec.flush_interval.count()));

    nixlTelemetry telemetry(spec.agent, spec.exporter);
    produce(telemetry);

    const nixl::metrics_test::labelSet agent{{"agent_name", spec.agent}};
    overflowScrape result;
    static_cast<void>(nixl::metrics_test::scrapeUntil(
        spec.port, spec.settle_timeout, [&](const nixl::metrics_test::timeSeries &scrape) {
            const auto dropped = scrape.latestValue("agent_telemetry_events_dropped_total", agent);
            const auto accepted = scrape.latestValue(spec.accepted_metric, agent);
            if (!dropped || !accepted) {
                return false;
            }
            result.dropped = *dropped;
            result.accepted = *accepted;
            result.ok = result.accepted * static_cast<double>(spec.accepted_event_weight) +
                    result.dropped ==
                static_cast<double>(spec.expected_total_events);
            return result.ok;
        }));
    return result;
}

} // namespace gtest

#endif // NIXL_TEST_GTEST_TELEMETRY_CORE_SCRAPE_H
