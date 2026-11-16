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

#include "common.h"

#include "histogram_buckets.h"

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

TEST(histogramBucketsTest, ValidListIsParsedInOrder) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "1, 2.5, 10, 100");
    EXPECT_EQ(nixl::telemetry::resolveHistogramBucketsUs(),
              (std::vector<double>{1.0, 2.5, 10.0, 100.0}));
}

TEST(histogramBucketsTest, NonNumericThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "10,not_a_number,30");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, UnsortedThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "10,5,20");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, NonPositiveThrows) {
    gtest::ScopedEnv env;
    env.addVar(nixl::telemetry::histogramBucketsUsVar, "0,10,20");
    EXPECT_THROW(nixl::telemetry::resolveHistogramBucketsUs(), std::invalid_argument);
}

TEST(histogramBucketsTest, AbsentUsesDefaults) {
    ::unsetenv(nixl::telemetry::histogramBucketsUsVar);
    EXPECT_EQ(nixl::telemetry::resolveHistogramBucketsUs(),
              nixl::telemetry::defaultHistogramBucketsUs());
}
