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
#ifndef NIXL_TEST_DOCA_TELEMETRY_TIMESERIES_H
#define NIXL_TEST_DOCA_TELEMETRY_TIMESERIES_H

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nixl::doca_test {

using labelSet = std::map<std::string, std::string>;

// A single sample of a time series: a value and, when the exposition carries one,
// a timestamp (stored verbatim). DOCA tracks time internally in microseconds, but
// the Prometheus text exposition DOCA serves uses milliseconds.
struct sample {
    double value = 0.0;
    std::optional<uint64_t> timestamp;
};

// Identity of a Prometheus time series: a metric name plus its label set. Every
// sample of a series shares this identity, so it is the natural key -- the name
// and labels live here once, not duplicated onto each sample.
struct seriesId {
    std::string name;
    labelSet labels;

    bool
    operator<(const seriesId &rhs) const {
        return std::tie(name, labels) < std::tie(rhs.name, rhs.labels);
    }
};

// The samples of one series, and the id -> samples map produced by one parse.
using samples = std::vector<sample>;
using seriesMap = std::map<seriesId, samples>;

// A queryable view over a set of Prometheus time series: each series (name+labels)
// maps to its samples. It is a pure view over an already-parsed seriesMap -- it
// does not know how the text became series (OpenMetricsTextParser does that), so a
// caller looks up any number of series without rescanning the raw text. A metric
// name alone is ambiguous when the same metric is exported with different label
// sets, so it is never the key. A single scrape yields one sample per series; the
// sample vector also accommodates repeated scrapes.
class timeSeries {
public:
    explicit timeSeries(seriesMap series) : series_(std::move(series)) {}

    // Latest sampled value of the series named `name` whose labels include every
    // pair in `where` (subset match -- pass only the labels you care about).
    // Returns nullopt if nothing matches, if more than one series matches
    // (ambiguous: add labels to disambiguate), or if the matched series is empty.
    [[nodiscard]] std::optional<double>
    latestValue(const std::string &name, const labelSet &where = {}) const {
        std::optional<double> result;
        bool matched = false;
        for (const auto &[id, seriesSamples] : series_) {
            if (id.name != name || !labelsContain(id.labels, where)) {
                continue;
            }
            if (matched) {
                return std::nullopt; // ambiguous: same name, different label sets
            }
            matched = true;
            if (!seriesSamples.empty()) {
                result = seriesSamples.back().value;
            }
        }
        return result;
    }

    // The full id -> samples map, for any assertion the helpers above do not
    // cover (e.g. timestamps or multi-sample series). Zero-copy.
    [[nodiscard]] const seriesMap &
    series() const noexcept {
        return series_;
    }

    [[nodiscard]] bool
    empty() const noexcept {
        return series_.empty();
    }

private:
    // true if `labels` contains every key/value pair in `required`.
    [[nodiscard]] static bool
    labelsContain(const labelSet &labels, const labelSet &required) {
        for (const auto &[key, value] : required) {
            const auto it = labels.find(key);
            if (it == labels.end() || it->second != value) {
                return false;
            }
        }
        return true;
    }

    seriesMap series_;
};

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_TIMESERIES_H
