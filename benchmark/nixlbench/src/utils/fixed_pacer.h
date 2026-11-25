/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_FIXED_PACER_H
#define NIXLBENCH_FIXED_PACER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace nixlbench {

inline void
validatePacingConfiguration(uint64_t rate,
                            uint64_t max_request_bytes,
                            std::string_view worker,
                            std::string_view backend,
                            std::string_view mode,
                            std::string_view scheme,
                            int threads,
                            int depth,
                            std::string_view operation,
                            bool device_api) {
    if (!rate && !max_request_bytes) {
        return;
    }
    if (!rate || !max_request_bytes) {
        throw std::invalid_argument(
            "pacing_bytes_per_sec and pacing_max_request_bytes must both be positive or both zero");
    }
    if (worker != "nixl" || backend != "UCX" || mode != "SG" || threads != 1 || depth != 1 ||
        device_api || (scheme != "pairwise" && scheme != "manytoone" && scheme != "onetomany") ||
        (operation != "READ" && operation != "WRITE")) {
        throw std::invalid_argument(
            "Pacing requires the host nixl UCX worker, SG, num_threads=1, pipeline_depth=1, "
            "pairwise/manytoone/onetomany, and READ or WRITE");
    }
}

// Charge application descriptor lengths, rather than padding or backend packet bytes.
template<typename Descriptors>
uint64_t
boundedRequestBytes(const Descriptors &descriptors, uint64_t limit) {
    uint64_t bytes = 0;
    for (const auto &descriptor : descriptors) {
        if (descriptor.len > limit - bytes) {
            throw std::invalid_argument("Transfer exceeds pacing_max_request_bytes");
        }
        bytes += descriptor.len;
    }
    if (!bytes) {
        throw std::invalid_argument("Paced transfer must contain application bytes");
    }
    return bytes;
}

// One request outstanding: callers complete the prior request before waiting here.
// The injected clock and sleeper permit deterministic tests without wall-clock sleeps.
class FixedPacer {
public:
    using Clock = std::chrono::steady_clock;

    explicit FixedPacer(uint64_t bytes_per_second) : rate_(bytes_per_second) {}

    template<typename Now, typename Sleep, typename Cancelled>
    bool
    wait(uint64_t bytes, Now now, Sleep sleep, Cancelled cancelled) {
        if (!rate_) {
            return true;
        }
        const long double ticks = std::ceil(static_cast<long double>(bytes) * Clock::period::den /
                                            Clock::period::num / rate_);
        if (ticks <= 0 || ticks >= std::numeric_limits<Clock::rep>::max()) {
            throw std::invalid_argument("Pacing interval exceeds the supported clock range");
        }
        const auto interval = Clock::duration(static_cast<Clock::rep>(ticks));
        auto current = now();
        const auto anchor = started_ ? last_release_ : current;
        if (anchor.time_since_epoch() > Clock::duration::max() - interval) {
            throw std::invalid_argument("Pacing deadline exceeds the supported clock range");
        }
        const auto deadline = anchor + interval;
        while (current < deadline) {
            if (cancelled()) {
                return false;
            }
            // Keep cancellation responsive without requiring transport progress while idle.
            sleep(std::min(
                deadline - current,
                std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(1))));
            current = now();
        }
        if (cancelled()) {
            return false;
        }
        // A late wake earns no credit toward the following request.
        last_release_ = current;
        started_ = true;
        return true;
    }

private:
    uint64_t rate_;
    bool started_ = false;
    Clock::time_point last_release_{};
};

} // namespace nixlbench

#endif // NIXLBENCH_FIXED_PACER_H
