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
#ifndef NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
#define NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H

#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include "loopback_connection.h"
#include "open_metrics_text_parser.h"
#include "timeseries.h"

namespace nixl::doca_test {

// Poll /metrics until the series `name` (matching optional label subset `where`)
// reads exactly `expected`, or until timeout. Each poll parses the body once into
// a timeSeries; the final scrape is returned so the caller can assert that series
// and any others without rescanning. Cumulative counters settle asynchronously
// after a flush, so a single read can observe a stale value.
[[nodiscard]] inline timeSeries
scrapeUntilValue(uint16_t port,
                 const std::string &name,
                 double expected,
                 std::chrono::seconds timeout,
                 const labelSet &where = {}) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    timeSeries metrics{seriesMap{}};
    do {
        metrics =
            timeSeries(OpenMetricsTextParser::parse(loopbackConnection::httpGet(port, "/metrics")));
        if (metrics.latestValue(name, where) == expected) {
            return metrics;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return metrics;
}

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
