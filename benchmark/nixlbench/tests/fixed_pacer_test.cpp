/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fixed_pacer.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std::chrono_literals;

namespace {
void
check(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template<typename Action>
void
rejects(Action action) {
    bool rejected = false;
    try {
        action();
    }
    catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "Expected invalid argument rejection");
}

struct Descriptor {
    uint64_t len;
};

void
checkBytesAndConfig() {
    using namespace nixlbench;
    check(boundedRequestBytes(std::vector<Descriptor>{{123}, {77}}, 200) == 200,
          "Charge exact descriptor lengths");
    rejects([] { boundedRequestBytes(std::vector<Descriptor>{{123}, {78}}, 200); });
    rejects([] { boundedRequestBytes(std::vector<Descriptor>{}, 200); });
    rejects([] { boundedRequestBytes(std::vector<Descriptor>{{UINT64_MAX}, {1}}, UINT64_MAX); });
    validatePacingConfiguration(0, 0, "other", "other", "MG", "tp", 8, 8, "QUERY", true);
    for (const auto operation : {"READ", "WRITE"}) {
        for (const auto scheme : {"pairwise", "manytoone", "onetomany"}) {
            validatePacingConfiguration(
                1000, 100, "nixl", "UCX", "SG", scheme, 1, 1, operation, false);
        }
    }
    rejects(
        [] { validatePacingConfiguration(1, 1, "nixl", "UCX", "SG", "tp", 1, 1, "WRITE", false); });
    rejects([] {
        validatePacingConfiguration(1, 0, "nixl", "UCX", "SG", "pairwise", 1, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(0, 1, "nixl", "UCX", "SG", "pairwise", 1, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nvshmem", "UCX", "SG", "pairwise", 1, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "POSIX", "SG", "pairwise", 1, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "UCX", "MG", "pairwise", 1, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "UCX", "SG", "pairwise", 2, 1, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "UCX", "SG", "pairwise", 1, 2, "WRITE", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "UCX", "SG", "pairwise", 1, 1, "QUERY", false);
    });
    rejects([] {
        validatePacingConfiguration(1, 1, "nixl", "UCX", "SG", "pairwise", 1, 1, "WRITE", true);
    });
}

void
checkSchedule() {
    using nixlbench::FixedPacer;
    FixedPacer::Clock::time_point time{};
    auto now = [&] { return time; };
    int sleeps = 0;
    auto sleep = [&](auto duration) {
        ++sleeps;
        time += duration;
    };
    auto running = [] { return false; };
    FixedPacer pacer(1000);
    check(pacer.wait(10, now, sleep, running), "First request released");
    check(time.time_since_epoch() == 10ms, "First request is charged before release");
    check(pacer.wait(20, now, sleep, running), "Second request released");
    check(time.time_since_epoch() == 30ms, "Variable request bytes determine spacing");
    time += 100ms;
    check(pacer.wait(10, now, sleep, running), "Late request released");
    check(time.time_since_epoch() == 130ms, "Late request needs no redundant sleep");
    check(pacer.wait(10, now, sleep, running), "Following request released");
    check(time.time_since_epoch() == 140ms, "Idle time grants no catch-up burst");

    // Cancellation happens while waiting, before another request is posted.
    const auto before = time;
    int checks = 0;
    check(!pacer.wait(10, now, sleep, [&] { return ++checks == 3; }), "Cancelled wait");
    check(time - before == 2ms, "Cancellation checked at bounded intervals");
    const auto calls = sleeps;
    FixedPacer disabled(0);
    check(disabled.wait(UINT64_MAX, now, sleep, [] { return true; }), "Disabled path returns");
    check(sleeps == calls, "Disabled pacing never sleeps");
    rejects([&] { FixedPacer(1).wait(UINT64_MAX, now, sleep, running); });
    rejects([&] { pacer.wait(0, now, sleep, running); });

    FixedPacer rounded(3);
    const auto rounding_start = time;
    check(rounded.wait(1, now, sleep, running), "Fractional interval released");
    check(time - rounding_start == 333333334ns,
          "Fractional byte intervals round conservatively to the clock resolution");
    time = FixedPacer::Clock::time_point::max() - 1ms;
    rejects([&] { FixedPacer(1000).wait(10, now, sleep, running); });
    time = FixedPacer::Clock::time_point{};

    // A simulated oversleep cannot be recovered as a burst of later releases.
    FixedPacer late(1000);
    auto oversleep = [&](auto duration) { time += duration + 100ms; };
    check(late.wait(10, now, oversleep, running), "Oversleep release");
    const auto release = time;
    check(late.wait(10, now, sleep, running), "Release after oversleep");
    check(time - release == 10ms, "Oversleep does not accumulate credits");
}
} // namespace

int
main() {
    try {
        checkBytesAndConfig();
        checkSchedule();
        std::cout << "PASS: exact bytes/cap, configuration, first/repeated/late release, "
                     "cancellation, disabled\n";
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
