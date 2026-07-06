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

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "nixl_duration.h"

using namespace std::chrono;

TEST(nixlDurationTest, ElapsedIsMonotonic) {
    nixlTime::nixlDuration d;
    auto prev = d.elapsed();
    for (int i = 0; i < 100000; ++i) {
        const auto cur = d.elapsed();
        ASSERT_GE(cur.count(), 0);
        ASSERT_GE(cur.count(), prev.count());
        prev = cur;
    }
}

TEST(nixlDurationTest, MeasuresIntervalAccurately) {
    constexpr auto window = milliseconds(20);

    nixlTime::nixlDuration d;
    const auto steady_start = steady_clock::now();
    std::this_thread::sleep_for(window);
    const auto steady_us = duration_cast<microseconds>(steady_clock::now() - steady_start).count();
    const auto measured_us = d.elapsed().count();

    EXPECT_GT(measured_us, 0);
    const auto tolerance = steady_us / 10;
    EXPECT_NEAR(measured_us, steady_us, tolerance)
        << "measured=" << measured_us << "us steady=" << steady_us
        << "us hw_counter=" << nixlTime::fastClockUsesHwCounter();
}

TEST(nixlDurationTest, StartRebasesTheStopwatch) {
    nixlTime::nixlDuration d;
    std::this_thread::sleep_for(milliseconds(20));
    d.start();
    EXPECT_LT(d.elapsed().count(), 10000);
}
