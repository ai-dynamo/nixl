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
#ifndef NIXL_TEST_DOCA_TELEMETRY_OPEN_METRICS_TEXT_PARSER_H
#define NIXL_TEST_DOCA_TELEMETRY_OPEN_METRICS_TEXT_PARSER_H

#include <algorithm>
#include <cstdint>
#include <exception>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>

#include "timeseries.h"

namespace nixl::doca_test {

// Parses a Prometheus / OpenMetrics text exposition body into time series.
//
// The grammar handled is the line format DOCA's endpoint serves:
//   name[{key="value"(,key="value")*}] value [timestamp]
// Comment (`#`) and blank lines are skipped. The parser is deliberately strict --
// a malformed label block (missing/doubled/trailing comma, repeated key, stray
// text) rejects the whole line rather than yielding partial labels -- so an
// exporter-format regression surfaces as a missing series in these tests instead
// of being silently absorbed.
class OpenMetricsTextParser {
public:
    // Parse a whole /metrics body once. Each well-formed line contributes one
    // sample to its (name+labels) series; the resulting map lets a caller query
    // any number of series without rescanning the raw text.
    [[nodiscard]] static seriesMap
    parse(const std::string &body) {
        seriesMap series;
        std::istringstream stream(body);
        std::string line;
        while (std::getline(stream, line)) {
            if (auto parsed = parseLine(line)) {
                series[std::move(parsed->first)].push_back(parsed->second);
            }
        }
        return series;
    }

private:
    // Parse one exposition line into a series id + sample. Returns nullopt for
    // comment/blank/malformed lines. Label values are assumed free of embedded
    // quotes (true for these metrics).
    [[nodiscard]] static std::optional<std::pair<seriesId, sample>>
    parseLine(const std::string &line) {
        // name, optional brace-delimited label block, value, optional timestamp.
        static const std::regex lineRe(R"(^([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)(?:\s+(\S+))?\s*$)");

        std::smatch match;
        if (line.empty() || line[0] == '#' || !std::regex_match(line, match, lineRe)) {
            return std::nullopt;
        }
        const std::optional<sample> value = parseSample(
            match[3].str(),
            match[4].matched ? std::optional<std::string>(match[4].str()) : std::nullopt);
        if (!value) {
            return std::nullopt;
        }
        std::optional<labelSet> labels = parseLabels(match[2].str());
        if (!labels) {
            return std::nullopt;
        }
        return std::make_pair(seriesId{match[1].str(), std::move(*labels)}, *value);
    }

    // Build a sample from its value token (required) and the optional timestamp
    // token (the exposition's trailing field). Returns nullopt if the value is
    // non-numeric; a missing or non-numeric timestamp is left unset, since
    // Prometheus timestamps are optional. std::stod/std::stoull stop at the first
    // non-numeric character (so "1abc" would parse as 1), so the whole token must
    // be consumed; std::stoull also wraps a leading '-' into a huge unsigned, so a
    // negative timestamp is treated as malformed.
    [[nodiscard]] static std::optional<sample>
    parseSample(const std::string &valueToken, const std::optional<std::string> &timestampToken) {
        sample s;
        try {
            size_t pos = 0;
            s.value = std::stod(valueToken, &pos);
            if (pos != valueToken.size()) {
                return std::nullopt;
            }
        }
        catch (const std::exception &) {
            return std::nullopt;
        }
        if (timestampToken && !timestampToken->empty() && timestampToken->front() != '-') {
            try {
                size_t pos = 0;
                const unsigned long long ts = std::stoull(*timestampToken, &pos);
                if (pos == timestampToken->size()) {
                    s.timestamp = static_cast<uint64_t>(ts);
                }
            }
            catch (const std::exception &) {
                // Leave the timestamp unset on a non-numeric token.
            }
        }
        return s;
    }

    // The gap between two matched pairs must hold exactly one comma (pairs are
    // comma-separated); the gap before the first pair must hold none. Whitespace
    // may pad either side, but no other character is allowed.
    [[nodiscard]] static bool
    validSeparator(const std::string &gap, bool leading) {
        if (gap.find_first_not_of(", \t") != std::string::npos) {
            return false;
        }
        return std::count(gap.begin(), gap.end(), ',') == (leading ? 0 : 1);
    }

    // Extract the key="value" pairs from a label block (the text inside `{}`).
    // Returns nullopt if the block is malformed -- a repeated key, a missing or
    // doubled comma between pairs, a trailing comma, or any other stray text -- so
    // a bad exposition line is rejected rather than silently parsed into partial
    // or first-wins labels (these are regression tests).
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
            const auto position = static_cast<size_t>(match.position());
            if (!validSeparator(block.substr(cursor, position - cursor), cursor == 0)) {
                return std::nullopt;
            }
            // A duplicate label key within one series is malformed; reject it.
            if (!labels.emplace(match[1].str(), match[2].str()).second) {
                return std::nullopt;
            }
            cursor = position + static_cast<size_t>(match.length());
        }
        // Any leftover after the last pair must be whitespace only (no trailing comma).
        if (block.find_first_not_of(" \t", cursor) != std::string::npos) {
            return std::nullopt;
        }
        return labels;
    }
};

} // namespace nixl::doca_test

#endif // NIXL_TEST_DOCA_TELEMETRY_OPEN_METRICS_TEXT_PARSER_H
