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
#ifndef NIXL_SRC_UTILS_COMMON_NIXL_DURATION_H
#define NIXL_SRC_UTILS_COMMON_NIXL_DURATION_H

#include <chrono>
#include <cstdint>

namespace nixlTime {

using namespace std::chrono;

namespace internal {

    struct fastClockCal {
        bool useHwCounter; // false -> nixlDuration measures with steady_clock
        double nsPerTick; // counter period in nanoseconds
    };

    extern const fastClockCal g_fastClockCal;

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

    // Monotonic base reading for a duration measurement: the raw CPU counter when
    // it is usable, else steady_clock nanoseconds. The value has an unspecified
    // origin and is only ever subtracted from a later reading of the same kind, so
    // that origin never escapes as an absolute timestamp.
    [[nodiscard]] inline uint64_t
    baseReading() {
        if (g_fastClockCal.useHwCounter) {
            return readCpuCounter();
        }
        return static_cast<uint64_t>(
            duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
    }

    [[nodiscard]] inline microseconds
    elapsedSince(uint64_t base) {
        if (g_fastClockCal.useHwCounter) {
            const int64_t dticks = static_cast<int64_t>(readCpuCounter() - base);
            const int64_t ns =
                static_cast<int64_t>(static_cast<double>(dticks) * g_fastClockCal.nsPerTick);
            return duration_cast<microseconds>(nanoseconds{ns});
        }
        const int64_t dns =
            duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() -
            static_cast<int64_t>(base);
        return duration_cast<microseconds>(nanoseconds{dns});
    }

} // namespace internal

/*
 * nixlDuration: a low-overhead monotonic stopwatch for hot paths that measure
 * short intervals frequently.
 *
 * It reads the CPU cycle counter (rdtsc on x86_64, cntvct_el0 on aarch64), which
 * is several times cheaper than steady_clock::now() (a VDSO
 * clock_gettime(CLOCK_MONOTONIC) call), falling back to steady_clock when no
 * invariant/kernel-trusted counter is available. It captures a base reading on
 * construction (and on restart()) and reports only the elapsed() interval; it
 * never exposes an absolute time point, so it is not comparable to
 * steady_clock/wall time and cannot drift against any epoch -- only the measured
 * interval is meaningful.
 */
class nixlDuration {
public:
    nixlDuration() : base_(internal::baseReading()) {}

    void
    restart() {
        base_ = internal::baseReading();
    }

    [[nodiscard]] microseconds
    elapsed() const {
        return internal::elapsedSince(base_);
    }

private:
    uint64_t base_;
};

// True when nixlDuration measures with the CPU counter rather than
// steady_clock. For tests/introspection, not the datapath.
[[nodiscard]] bool
fastClockUsesHwCounter();

} // namespace nixlTime

#endif
