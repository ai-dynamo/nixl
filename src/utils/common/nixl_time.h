/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _NIXL_TIME_H
#define _NIXL_TIME_H

#include <atomic>
#include <chrono>
#include <cstdint>

#if defined(__cpp_constinit)
#define NIXL_CONSTINIT constinit
#else
#define NIXL_CONSTINIT
#endif

namespace nixlTime {

using namespace std::chrono;

using ns_t = uint64_t;
using us_t = uint64_t;
using ms_t = uint64_t;
using sec_t = uint64_t;

static inline ns_t
getNs() {
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static inline us_t
getUs() {
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

static inline ms_t
getMs() {
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static inline sec_t
getSec() {
    return duration_cast<seconds>(steady_clock::now().time_since_epoch()).count();
}

/*
 * fastSteadyNow(): a lower-overhead drop-in for steady_clock::now() for hot
 * paths that take frequent timestamps.
 *
 * steady_clock::now() is a VDSO clock_gettime(CLOCK_MONOTONIC) call; reading
 * the CPU cycle counter directly (rdtsc on x86_64, cntvct_el0 on aarch64) is
 * several times cheaper. The result is a steady_clock::time_point on the same
 * timeline as steady_clock::now(): a one-time calibration pairs the counter
 * with steady_clock, and fastSteadyNow() maps counter ticks back onto that
 * timeline so the value stays interchangeable with steady_clock::now() (i.e.
 * usable as an absolute timestamp, not only for deltas). On Linux both derive
 * from the same hardware source, so the mapping does not drift. When no
 * invariant/usable counter is available it falls back to steady_clock::now().
 */

namespace detail {

    struct fastClockCal {
        bool useHwCounter; // false -> fastSteadyNow() uses steady_clock::now()
        uint64_t counterRef; // counter value captured at calibration
        int64_t steadyRefNs; // steady_clock::now() (ns since epoch) at calibration
        double nsPerTick; // counter period in nanoseconds
    };

    extern NIXL_CONSTINIT fastClockCal g_fastClockCal;
    extern NIXL_CONSTINIT std::atomic<bool> g_fastClockReady;

    void
    ensureFastClockCalibrated();

    [[nodiscard]] inline uint64_t
    readCpuCounter() {
#if defined(__x86_64__)
        uint32_t lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
        uint64_t cnt;
        __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(cnt));
        return cnt;
#else
        return 0;
#endif
    }

    [[nodiscard]] inline steady_clock::time_point
    steadyFromCounter(const fastClockCal &cal, uint64_t counter) {
        // Signed tick delta tolerates the tiny cross-core counter skew that can
        // make a read momentarily precede counterRef right after calibration.
        const int64_t dticks = static_cast<int64_t>(counter - cal.counterRef);
        const int64_t ns =
            cal.steadyRefNs + static_cast<int64_t>(static_cast<double>(dticks) * cal.nsPerTick);
        return steady_clock::time_point{nanoseconds{ns}};
    }

} // namespace detail

// Monotonic timestamp on the steady_clock timeline, backed by the CPU counter
// when it is invariant and calibrated, else steady_clock::now().
[[nodiscard]] inline steady_clock::time_point
fastSteadyNow() {
    if (!detail::g_fastClockReady.load(std::memory_order_acquire)) {
        detail::ensureFastClockCalibrated();
    }
    const detail::fastClockCal &cal = detail::g_fastClockCal;
    if (!cal.useHwCounter) {
        return steady_clock::now();
    }
    return detail::steadyFromCounter(cal, detail::readCpuCounter());
}

// True when fastSteadyNow() uses the CPU counter rather than
// steady_clock::now(). For tests/introspection, not the datapath.
[[nodiscard]] bool
fastClockUsesHwCounter();
} // namespace nixlTime

#endif
