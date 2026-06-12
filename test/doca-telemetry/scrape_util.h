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
#include <sstream>
#include <string>
#include <thread>

#include "loopback_connection.h"

namespace nixl::doca_test {

// Poll /metrics until it contains `needle`, or timeout.
inline std::string
scrapeUntil(uint16_t port, const std::string &needle, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string body;
    do {
        body = loopbackConnection::httpGet(port, "/metrics");
        if (body.find(needle) != std::string::npos) {
            return body;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return body;
}

// Value on the first non-comment exposition line that starts with `metric`.
// Exposition format is: name{labels} VALUE [TIMESTAMP]  (or  name VALUE [TS]).
// The value is the token right after the label set, NOT the trailing timestamp.
inline double
metricValue(const std::string &body, const std::string &metric) {
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind(metric, 0) != 0) {
            continue;
        }

        size_t value_start;
        const auto labels_end = line.find("} ");
        if (labels_end != std::string::npos) {
            value_start = labels_end + 2;
        } else {
            const auto sp = line.find(' ');
            if (sp == std::string::npos) {
                continue;
            }
            value_start = sp + 1;
        }

        const auto value_end = line.find(' ', value_start);
        const std::string token = line.substr(
            value_start,
            value_end == std::string::npos ? std::string::npos : value_end - value_start);
        try {
            return std::stod(token);
        }
        catch (const std::exception &) {
        }
    }
    return -1.0;
}

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
