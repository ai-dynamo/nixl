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

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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
 * @brief Parse a comma-separated list of strictly-increasing positive numbers.
 * @param spec Comma-separated bucket specification. Values are real numbers (fractional
 *             microseconds are allowed), and leading/trailing whitespace around each token
 *             is tolerated (absl::SimpleAtod ignores it).
 * @return Parsed boundaries, or an empty vector when @p spec is malformed (non-numeric,
 *         non-finite, non-positive, or not strictly increasing).
 */
[[nodiscard]] inline std::vector<double>
parseHistogramBucketsUs(const std::string &spec) {
    std::vector<double> buckets;
    double previous = 0.0;
    for (const absl::string_view token : absl::StrSplit(spec, ',')) {
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
 * @return The parsed env override when set and valid; the built-in microsecond defaults
 *         when the override is absent or empty.
 * @throws std::invalid_argument when a non-empty override is provided but malformed, so a
 *         user who specified buckets is not silently given the defaults instead.
 */
[[nodiscard]] inline std::vector<double>
resolveHistogramBucketsUs() {
    const auto spec = nixl::config::getValueOptional<std::string>(histogramBucketsUsVar);
    if (!spec || spec->empty()) {
        return defaultHistogramBucketsUs();
    }

    const std::vector<double> buckets = parseHistogramBucketsUs(*spec);
    if (buckets.empty()) {
        throw std::invalid_argument(
            std::string(histogramBucketsUsVar) +
            " must be a comma-separated list of strictly-increasing positive numbers");
    }
    return buckets;
}

} // namespace nixl::telemetry

#endif // NIXL_SRC_PLUGINS_TELEMETRY_COMMON_HISTOGRAM_BUCKETS_H
