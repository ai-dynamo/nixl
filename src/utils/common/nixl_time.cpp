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

#include <fstream>
#include <mutex>
#include <string>
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
        // Invariant TSC (constant + non-stop): CPUID.80000007H:EDX[8]. Necessary but not
        // sufficient -- it does not prove cross-socket synchronization.
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

        // The kernel selects the TSC as the CLOCK_MONOTONIC source only after its
        // boot-time cross-core synchronization check (tsc_sync.c) passes; if it detects
        // warping it demotes to hpet/acpi_pm. Deferring to that decision is the reliable
        // way to know the TSC is safe to use as a wall-time source here.
        [[nodiscard]] bool
        kernelClocksourceIsTsc() {
            std::ifstream f("/sys/devices/system/clocksource/clocksource0/current_clocksource");
            std::string clocksource;
            return static_cast<bool>(std::getline(f, clocksource)) && clocksource == "tsc";
        }

        [[nodiscard]] double
        tscFreqHzFromSysfs() {
            std::ifstream f("/sys/devices/system/cpu/cpu0/tsc_freq_khz");
            long khz = 0;
            if (f >> khz && khz > 0) {
                return static_cast<double>(khz) * 1e3;
            }
            return 0.0;
        }

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
#endif

        [[nodiscard]] fastClockCal
        calibrate() {
            fastClockCal cal{};
            cal.useHwCounter = false;

#if defined(__x86_64__)
            if (hasInvariantTsc() && kernelClocksourceIsTsc()) {
                double hz = tscFreqHzFromSysfs();
                if (hz <= 0.0) {
                    constexpr auto calibration_window = milliseconds(3);
                    hz = measureCounterHz(calibration_window);
                }
                if (hz > 0.0) {
                    cal.nsPerTick = 1e9 / hz;
                    cal.useHwCounter = true;
                }
            }
#elif defined(__aarch64__)
            // The ARM generic timer (cntvct_el0) is architecturally a single system-wide
            // counter, so it has no cross-socket-sync caveat; cntfrq_el0 is its exact rate.
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

        NIXL_CONSTINIT std::once_flag g_calibrateOnce;

    } // namespace

    NIXL_CONSTINIT fastClockCal g_fastClockCal{};
    NIXL_CONSTINIT std::atomic<bool> g_fastClockReady{false};

    void
    ensureFastClockCalibrated() {
        std::call_once(g_calibrateOnce, []() {
            g_fastClockCal = calibrate();
            g_fastClockReady.store(true, std::memory_order_release);
        });
    }

} // namespace detail

[[nodiscard]] bool
fastClockUsesHwCounter() {
    detail::ensureFastClockCalibrated();
    return detail::g_fastClockCal.useHwCounter;
}

} // namespace nixlTime
