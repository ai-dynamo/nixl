/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_coalescing.h"

#include <iostream>

using nixl::gpunetio::ReadSegment;
using nixl::gpunetio::canCoalesceRead;

int
main() {
    constexpr size_t page = 4096;
    const ReadSegment current{0x10000, 0x20000, page, 11, 22};
    const ReadSegment adjacent{0x11000, 0x21000, page, 11, 22};

    auto require = [](bool condition, const char *message) {
        if (!condition) {
            std::cerr << message << std::endl;
        }
        return condition;
    };

    bool passed = true;
    passed &= require(canCoalesceRead(current, adjacent, 2 * page), "adjacent run rejected");
    passed &=
        require(!canCoalesceRead(current, ReadSegment{0x12000, 0x21000, page, 11, 22}, 2 * page),
                "local gap accepted");
    passed &=
        require(!canCoalesceRead(current, ReadSegment{0x11000, 0x22000, page, 11, 22}, 2 * page),
                "remote gap accepted");
    passed &=
        require(!canCoalesceRead(current, ReadSegment{0x11000, 0x21000, page, 12, 22}, 2 * page),
                "local key boundary accepted");
    passed &=
        require(!canCoalesceRead(current, ReadSegment{0x11000, 0x21000, page, 11, 23}, 2 * page),
                "remote key boundary accepted");
    passed &= require(!canCoalesceRead(current, adjacent, 2 * page - 1), "length limit exceeded");
    passed &= require(
        !canCoalesceRead(
            ReadSegment{std::numeric_limits<uintptr_t>::max() - 1024, 0x20000, page, 11, 22},
            adjacent,
            2 * page),
        "address overflow accepted");

    return passed ? 0 : 1;
}
