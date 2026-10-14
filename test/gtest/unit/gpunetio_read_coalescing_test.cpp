/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_read_coalescing.h"
#include <gtest/gtest.h>

namespace {

constexpr size_t max_read_size = 1ULL << 30;
const nixl::gpunetio::readRange previous{0x1000, 0x10000, 4096, 7, 11};
const nixl::gpunetio::readRange next{0x2000, 0x11000, 4096, 7, 11};

TEST(GpunetioReadCoalescing, AdjacentSameKeyRanges) {
    EXPECT_TRUE(nixl::gpunetio::canCoalesceRead(previous, next, max_read_size));
}

TEST(GpunetioReadCoalescing, EitherAddressGapStopsMerging) {
    auto range = next;
    range.localAddr += 1;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(previous, range, max_read_size));
    range = next;
    range.remoteAddr += 1;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(previous, range, max_read_size));
}

TEST(GpunetioReadCoalescing, EitherKeyChangeStopsMerging) {
    auto range = next;
    range.localKey += 1;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(previous, range, max_read_size));
    range = next;
    range.remoteKey += 1;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(previous, range, max_read_size));
}

TEST(GpunetioReadCoalescing, CombinedLengthMustFitSdkLimit) {
    auto first = previous;
    first.size = max_read_size - 4096;
    auto second = next;
    second.localAddr = first.localAddr + first.size;
    second.remoteAddr = first.remoteAddr + first.size;
    EXPECT_TRUE(nixl::gpunetio::canCoalesceRead(first, second, max_read_size));
    ++second.size;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(first, second, max_read_size));
    first.size = max_read_size + 1;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(first, second, max_read_size));
}

TEST(GpunetioReadCoalescing, AddressOverflowDoesNotAppearAdjacent) {
    const auto limit = std::numeric_limits<uintptr_t>::max();
    auto first = previous;
    auto second = next;
    first.localAddr = limit - first.size + 1;
    second.localAddr = 0;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(first, second, max_read_size));
    first = previous;
    second = next;
    first.remoteAddr = limit - first.size + 1;
    second.remoteAddr = 0;
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(first, second, max_read_size));
}

TEST(GpunetioReadCoalescing, LengthOverflowDoesNotFit) {
    auto first = previous;
    first.size = std::numeric_limits<size_t>::max();
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(first, next, max_read_size));
    auto second = next;
    second.size = std::numeric_limits<size_t>::max();
    EXPECT_FALSE(nixl::gpunetio::canCoalesceRead(previous, second, max_read_size));
}

} // namespace
