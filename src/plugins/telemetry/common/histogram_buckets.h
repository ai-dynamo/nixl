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
#ifndef NIXL_SRC_PLUGINS_TELEMETRY_COMMON_HISTOGRAM_BUCKETS_H
#define NIXL_SRC_PLUGINS_TELEMETRY_COMMON_HISTOGRAM_BUCKETS_H

#include "common/configuration.h"
#include "common/nixl_log.h"

#include <cmath>
#include <string>
#include <vector>

#include <absl/strings/ascii.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

namespace nixl::telemetry {

inline constexpr char histogramBucketsUsVar[] = "NIXL_TELEMETRY_HISTOGRAM_BUCKETS_US";

/**
 * @brief Built-in microsecond histogram bucket upper bounds (~10us..~10s).
 * @return Shared default boundaries, used when the env override is absent or invalid.
 */
[[nodiscard]] inline const std::vector<double> &
defaultHistogramBucketsUs() {
    static const std::vector<double> buckets = {10,
                                                25,
                                                50,
                                                100,
                                                250,
                                                500,
                                                1000,
                                                2500,
                                                5000,
                                                10000,
                                                25000,
                                                50000,
                                                100000,
                                                250000,
                                                500000,
                                                1000000,
                                                5000000,
                                                10000000};
    return buckets;
}

/**
 * @brief Parse a comma-separated list of strictly-increasing positive doubles.
 * @param spec Comma-separated bucket specification (whitespace around tokens is trimmed).
 * @return Parsed boundaries, or an empty vector when @p spec is malformed so the caller
 *         can fall back to the defaults (validation kept local to the telemetry plugins).
 */
[[nodiscard]] inline std::vector<double>
parseHistogramBucketsUs(const std::string &spec) {
    std::vector<double> buckets;
    double previous = 0.0;
    for (const absl::string_view raw : absl::StrSplit(spec, ',')) {
        const absl::string_view token = absl::StripAsciiWhitespace(raw);
        double value = 0.0;
        if (!absl::SimpleAtod(token, &value) || !std::isfinite(value) || value <= 0.0 ||
            value <= previous) {
            return {};
        }
        buckets.push_back(value);
        previous = value;
    }
    return buckets;
}

/**
 * @brief Resolve the histogram bucket boundaries (microsecond upper bounds) shared by
 *        both duration histograms in both exporters.
 * @return The parsed env override when valid; otherwise the built-in microsecond defaults.
 *         Emits a WARN only when a non-empty override was provided but is invalid.
 */
[[nodiscard]] inline std::vector<double>
resolveHistogramBucketsUs() {
    const auto spec = nixl::config::getValueOptional<std::string>(histogramBucketsUsVar);
    if (!spec || spec->empty()) {
        return defaultHistogramBucketsUs();
    }

    std::vector<double> buckets = parseHistogramBucketsUs(*spec);
    if (buckets.empty()) {
        NIXL_WARN << histogramBucketsUsVar
                  << " must be a comma-separated list of strictly-increasing positive numbers; "
                     "falling back to default microsecond buckets";
        return defaultHistogramBucketsUs();
    }
    return buckets;
}

} // namespace nixl::telemetry

#endif // NIXL_SRC_PLUGINS_TELEMETRY_COMMON_HISTOGRAM_BUCKETS_H
