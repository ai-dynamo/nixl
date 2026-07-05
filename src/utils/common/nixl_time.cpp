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

#include "nixl_time.h"

#include <thread>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace nixlTime {
namespace detail {

    namespace {

        [[nodiscard]] int64_t
        steadyNowNs() {
            return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
        }

#if defined(__x86_64__)
        // Invariant TSC (constant + non-stop): CPUID.80000007H:EDX[8]. Required for the
        // counter to be a usable wall-time source across cores and P-states.
        [[nodiscard]] bool
        hasInvariantTsc() {
            unsigned eax, ebx, ecx, edx;
            if (__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx) == 0 || eax < 0x80000007u) {
                return false;
            }
            if (__get_cpuid(0x80000007u, &eax, &ebx, &ecx, &edx) == 0) {
                return false;
            }
            return (edx & (1u << 8)) != 0;
        }
#endif

        // Estimate the counter frequency (Hz) by bracketing a steady_clock window of the
        // given length with counter reads. Used on x86 where the TSC rate is not
        // otherwise exposed cheaply; runs once at load.
        [[nodiscard]] double
        measureCounterHz(milliseconds window) {
            const int64_t steady_start = steadyNowNs();
            const uint64_t counter_start = readCpuCounter();
            std::this_thread::sleep_for(window);
            const uint64_t counter_end = readCpuCounter();
            const int64_t steady_end = steadyNowNs();

            const int64_t elapsed_ns = steady_end - steady_start;
            if (elapsed_ns <= 0) {
                return 0.0;
            }
            const uint64_t elapsed_ticks = counter_end - counter_start;
            return static_cast<double>(elapsed_ticks) * 1e9 / static_cast<double>(elapsed_ns);
        }

        [[nodiscard]] fastClockCal
        calibrate() {
            fastClockCal cal{};
            cal.useHwCounter = false;

#if defined(__x86_64__)
            if (hasInvariantTsc()) {
                constexpr auto calibration_window = milliseconds(3);
                const double hz = measureCounterHz(calibration_window);
                if (hz > 0.0) {
                    cal.nsPerTick = 1e9 / hz;
                    cal.useHwCounter = true;
                }
            }
#elif defined(__aarch64__)
            uint64_t freq = 0;
            __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(freq));
            if (freq != 0) {
                cal.nsPerTick = 1e9 / static_cast<double>(freq);
                cal.useHwCounter = true;
            }
#endif

            if (cal.useHwCounter) {
                // Pair the counter with steady_clock so fastSteadyNow() lands on the
                // steady_clock timeline. Read the counter between two steady reads and
                // anchor to their midpoint to halve the pairing skew.
                const int64_t s0 = steadyNowNs();
                cal.counterRef = readCpuCounter();
                const int64_t s1 = steadyNowNs();
                cal.steadyRefNs = s0 + (s1 - s0) / 2;
            }

            return cal;
        }

    } // namespace

    const fastClockCal g_fastClockCal = calibrate();

} // namespace detail

[[nodiscard]] bool
fastClockUsesHwCounter() {
    return detail::g_fastClockCal.useHwCounter;
}

} // namespace nixlTime
