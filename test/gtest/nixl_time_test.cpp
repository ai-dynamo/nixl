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

#include "nixl_time.h"

using namespace std::chrono;

// The pure counter->steady mapping is exact for known calibration values, with
// no hardware dependency: ns = steadyRefNs + (counter - counterRef) * nsPerTick.
TEST(nixlTimeTest, SteadyFromCounterMappingIsExact) {
    nixlTime::detail::fastClockCal cal{};
    cal.useHwCounter = true;
    cal.counterRef = 1000;
    cal.steadyRefNs = 5'000'000'000; // 5 s
    cal.nsPerTick = 0.5; // 2 GHz counter

    // 2000 ticks past the reference -> +1000 ns.
    const auto tp = nixlTime::detail::steadyFromCounter(cal, 3000);
    EXPECT_EQ(duration_cast<nanoseconds>(tp.time_since_epoch()).count(), 5'000'001'000);

    // At the reference the result is exactly steadyRefNs.
    const auto at_ref = nixlTime::detail::steadyFromCounter(cal, cal.counterRef);
    EXPECT_EQ(duration_cast<nanoseconds>(at_ref.time_since_epoch()).count(), cal.steadyRefNs);

    // A counter slightly behind the reference maps just before steadyRefNs
    // (signed delta), not to a huge value from unsigned wraparound.
    const auto before = nixlTime::detail::steadyFromCounter(cal, cal.counterRef - 100);
    EXPECT_EQ(duration_cast<nanoseconds>(before.time_since_epoch()).count(), 4'999'999'950);
}

// fastSteadyNow() must never go backwards.
TEST(nixlTimeTest, FastSteadyNowMonotonic) {
    auto prev = nixlTime::fastSteadyNow();
    for (int i = 0; i < 100000; ++i) {
        const auto cur = nixlTime::fastSteadyNow();
        ASSERT_GE(cur, prev);
        prev = cur;
    }
}

// fastSteadyNow() lands on the steady_clock timeline: a reading taken next to
// steady_clock::now() must be close to it (the telemetry startTime is exposed as
// an absolute steady-epoch timestamp, so the epochs must match). Generous bound
// to tolerate scheduling and the calibration pairing skew.
TEST(nixlTimeTest, FastSteadyNowAgreesWithSteadyClockEpoch) {
    for (int i = 0; i < 5; ++i) {
        const auto s = steady_clock::now();
        const auto f = nixlTime::fastSteadyNow();
        const auto diff = std::abs(duration_cast<microseconds>(f - s).count());
        EXPECT_LT(diff, 5000) << "fastSteadyNow() drifted from steady_clock by " << diff << " us";
    }
}

// A measured interval must match steady_clock's over the same window.
TEST(nixlTimeTest, FastSteadyNowMeasuresIntervalAccurately) {
    constexpr auto window = milliseconds(20);

    const auto fast_start = nixlTime::fastSteadyNow();
    const auto steady_start = steady_clock::now();
    std::this_thread::sleep_for(window);
    const auto steady_end = steady_clock::now();
    const auto fast_end = nixlTime::fastSteadyNow();

    const auto fast_us = duration_cast<microseconds>(fast_end - fast_start).count();
    const auto steady_us = duration_cast<microseconds>(steady_end - steady_start).count();

    EXPECT_GT(fast_us, 0);
    // Within 10% of the steady_clock-measured interval.
    const auto tolerance = steady_us / 10;
    EXPECT_NEAR(fast_us, steady_us, tolerance)
        << "fast=" << fast_us << "us steady=" << steady_us
        << "us hw_counter=" << nixlTime::fastClockUsesHwCounter();
}
