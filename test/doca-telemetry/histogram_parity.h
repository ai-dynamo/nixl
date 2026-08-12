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
#ifndef NIXL_TEST_DOCA_TELEMETRY_HISTOGRAM_PARITY_H
#define NIXL_TEST_DOCA_TELEMETRY_HISTOGRAM_PARITY_H

#include "timeseries.h"

#include <cstdlib>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

namespace nixl::telemetry_test {

using seriesLines = std::set<std::string>;

// Reduce a label value to a canonical form so two exporters that render the same
// number differently compare equal. prometheus-cpp writes histogram boundaries as
// "1000000" while DOCA/CollectX writes "1e+06"; both are the same double. A value
// that is not fully numeric (agent_name, hostname, status, ...) is returned
// unchanged.
[[nodiscard]] inline std::string
canonicalLabelValue(const std::string &raw) {
    char *end = nullptr;
    const double value = std::strtod(raw.c_str(), &end);
    if (raw.empty() || end != raw.c_str() + raw.size()) {
        return raw;
    }
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

// Every histogram series emitted for `agent_name`, rendered as a canonical
// "name{sorted labels} value" line: numeric label values canonicalized (so
// "1e+06" and "1000000" match) and timestamps dropped (values only). Returned as
// a set of strings so two exporters' payloads compare with a plain EXPECT_EQ that
// gtest can print -- the same technique the descriptor-series parity tests use. A
// histogram <base> is exactly the series <base>_bucket / <base>_sum /
// <base>_count, so the families are discovered by their _bucket component and no
// metric name is hard-coded.
[[nodiscard]] inline seriesLines
histogramSeriesLines(const nixl::doca_test::timeSeries &metrics, const std::string &agent_name) {
    const auto endsWith = [](const std::string &name, const std::string &suffix) {
        return name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    std::set<std::string> bases;
    for (const auto &[id, samples] : metrics.series()) {
        (void)samples;
        if (endsWith(id.name, "_bucket")) {
            bases.insert(id.name.substr(0, id.name.size() - std::string("_bucket").size()));
        }
    }

    seriesLines lines;
    for (const auto &[id, samples] : metrics.series()) {
        const auto agent_it = id.labels.find("agent_name");
        if (agent_it == id.labels.end() || agent_it->second != agent_name || samples.empty()) {
            continue;
        }

        bool is_histogram = false;
        for (const std::string suffix : {"_bucket", "_sum", "_count"}) {
            if (endsWith(id.name, suffix) &&
                bases.count(id.name.substr(0, id.name.size() - suffix.size())) != 0) {
                is_histogram = true;
                break;
            }
        }
        if (!is_histogram) {
            continue;
        }

        std::ostringstream line;
        line << id.name << '{';
        bool first = true;
        for (const auto &[label, value] : id.labels) {
            if (!first) {
                line << ',';
            }
            line << label << "=\"" << canonicalLabelValue(value) << '"';
            first = false;
        }
        line << "} " << samples.back().value;
        lines.insert(line.str());
    }
    return lines;
}

} // namespace nixl::telemetry_test

#endif // NIXL_TEST_DOCA_TELEMETRY_HISTOGRAM_PARITY_H
