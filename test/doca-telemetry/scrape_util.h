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
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "loopback_connection.h"

namespace nixl::doca_test {

using labelSet = std::map<std::string, std::string>;

// A single sample of a time series: a value and, when the exposition carries one,
// a timestamp (stored verbatim). DOCA tracks time internally in microseconds, but
// the Prometheus text exposition DOCA serves uses milliseconds.
struct sample {
    double value = 0.0;
    std::optional<uint64_t> timestamp;

    // Build a sample from its value token (required) and the optional timestamp
    // token (the exposition's trailing field). Returns nullopt if the value is
    // non-numeric; a missing or non-numeric timestamp is left unset, since
    // Prometheus timestamps are optional.
    [[nodiscard]] static std::optional<sample>
    parse(const std::string &valueToken, const std::optional<std::string> &timestampToken) {
        sample s;
        try {
            s.value = std::stod(valueToken);
        }
        catch (const std::exception &) {
            return std::nullopt;
        }
        if (timestampToken) {
            try {
                s.timestamp = static_cast<uint64_t>(std::stoull(*timestampToken));
            }
            catch (const std::exception &) {
                // Leave the timestamp unset on a non-numeric token.
            }
        }
        return s;
    }
};

// Identity of a Prometheus time series: a metric name plus its label set. Every
// sample of a series shares this identity, so it is the natural key -- the name
// and labels live here once, not duplicated onto each sample.
struct seriesId {
    std::string name;
    labelSet labels;

    friend bool
    operator<(const seriesId &lhs, const seriesId &rhs) {
        return std::tie(lhs.name, lhs.labels) < std::tie(rhs.name, rhs.labels);
    }
};

// The samples of one series, and the id -> samples map produced by one parse.
using samples = std::vector<sample>;
using seriesMap = std::map<seriesId, samples>;

// A set of Prometheus time series parsed from a /metrics body: each series
// (name+labels) maps to its samples. The body is parsed ONCE so a caller can
// look up any number of series without rescanning the raw text per metric. A
// metric name alone is ambiguous when the same metric is exported with different
// label sets, so it is never the key. A single scrape yields one sample per
// series; the sample vector also accommodates repeated scrapes.
class timeSeries {
public:
    explicit timeSeries(const std::string &body) {
        std::istringstream stream(body);
        std::string line;
        while (std::getline(stream, line)) {
            seriesId id;
            sample s;
            if (parseLine(line, id, s)) {
                series_[std::move(id)].push_back(s);
            }
        }
    }

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

    // Parse one exposition line into a series id + sample. Returns false for
    // comment/blank/malformed lines. Format: name[{k="v",...}] value [timestamp].
    // Label values are assumed free of embedded quotes (true for these metrics).
    [[nodiscard]] static bool
    parseLine(const std::string &line, seriesId &id, sample &out) {
        // name, optional brace-delimited label block, value, optional timestamp.
        static const std::regex lineRe(R"(^([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)(?:\s+(\S+))?\s*$)");

        std::smatch match;
        if (line.empty() || line[0] == '#' || !std::regex_match(line, match, lineRe)) {
            return false;
        }
        const std::optional<sample> parsed = sample::parse(
            match[3].str(),
            match[4].matched ? std::optional<std::string>(match[4].str()) : std::nullopt);
        if (!parsed) {
            return false;
        }
        out = *parsed;
        std::optional<labelSet> labels = parseLabels(match[2].str());
        if (!labels) {
            return false;
        }
        id.name = match[1].str();
        id.labels = std::move(*labels);
        return true;
    }

    // Extract the key="value" pairs from a label block (the text inside `{}`).
    // Returns nullopt if the block is malformed -- anything other than well-formed
    // pairs separated by commas/whitespace -- so a bad exposition line is rejected
    // rather than silently parsed into partial labels (these are regression tests).
    [[nodiscard]] static std::optional<labelSet>
    parseLabels(const std::string &block) {
        // Custom raw-string delimiter so the pattern's )" does not close it early.
        static const std::regex labelRe(R"re(([^=,\s]+)\s*=\s*"([^"]*)")re");
        labelSet labels;
        size_t cursor = 0;
        for (auto it = std::sregex_iterator(block.begin(), block.end(), labelRe);
             it != std::sregex_iterator();
             ++it) {
            const auto &match = *it;
            // Only commas/whitespace may separate (and precede) the matched pairs.
            if (block.find_first_not_of(", \t", cursor) < static_cast<size_t>(match.position())) {
                return std::nullopt;
            }
            labels.emplace(match[1].str(), match[2].str());
            cursor = static_cast<size_t>(match.position() + match.length());
        }
        // Any leftover after the last pair must also be only commas/whitespace.
        if (block.find_first_not_of(", \t", cursor) != std::string::npos) {
            return std::nullopt;
        }
        return labels;
    }

    seriesMap series_;
};

// Poll /metrics until the series `name` (matching optional label subset `where`)
// reads exactly `expected`, or until timeout. Returns the final parsed scrape
// (each poll parses the body once) so the caller can assert that series and any
// others without rescanning. Cumulative counters settle asynchronously after a
// flush, so a single read can observe a stale value.
[[nodiscard]] inline timeSeries
scrapeUntilValue(uint16_t port,
                 const std::string &name,
                 double expected,
                 std::chrono::seconds timeout,
                 const labelSet &where = {}) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    timeSeries metrics{std::string{}};
    do {
        metrics = timeSeries(loopbackConnection::httpGet(port, "/metrics"));
        if (metrics.latestValue(name, where) == expected) {
            return metrics;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return metrics;
}

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_SCRAPE_UTIL_H
