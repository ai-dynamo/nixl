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
#ifndef NIXL_TEST_METRICS_SCRAPE_UTIL_H
#define NIXL_TEST_METRICS_SCRAPE_UTIL_H

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "loopback_connection.h"
#include "open_metrics_text_parser.h"
#include "timeseries.h"

namespace nixl::metrics_test {

// Poll /metrics until `ready` accepts the scrape, or until timeout. Each poll
// parses the body once into a timeSeries; the last scrape is returned either way,
// so the caller asserts on it (and on any other series) without rescanning and
// without a separate "did it settle" flag. An endpoint that is not up yet, or a
// cumulative counter that has not settled after a flush, both read as an
// unaccepted scrape and are simply retried.
[[nodiscard]] inline timeSeries
scrapeUntil(uint16_t port,
            std::chrono::seconds timeout,
            const std::function<bool(const timeSeries &)> &ready) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    timeSeries metrics{seriesMap{}};
    do {
        metrics =
            timeSeries(open_metrics_text::parse(loopbackConnection::httpGet(port, "/metrics")));
        if (ready(metrics)) {
            return metrics;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return metrics;
}

// Poll /metrics until the series `name` (matching optional label subset `where`)
// reads exactly `expected`, or until timeout.
[[nodiscard]] inline timeSeries
scrapeUntilValue(uint16_t port,
                 const std::string &name,
                 double expected,
                 std::chrono::seconds timeout,
                 const labelSet &where = {}) {
    return scrapeUntil(port, timeout, [&](const timeSeries &metrics) {
        return metrics.latestValue(name, where) == expected;
    });
}

// Poll /metrics until the endpoint answers with something parseable: an exposer
// is not necessarily serving the instant its exporter is constructed.
[[nodiscard]] inline timeSeries
scrapeMetrics(uint16_t port, std::chrono::seconds timeout = std::chrono::seconds(3)) {
    return scrapeUntil(port, timeout, [](const timeSeries &metrics) { return !metrics.empty(); });
}

} // namespace nixl::metrics_test

#endif // NIXL_TEST_METRICS_SCRAPE_UTIL_H
