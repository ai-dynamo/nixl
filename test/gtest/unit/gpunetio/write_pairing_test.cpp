/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_write_pairing.h"
#include <gtest/gtest.h>

namespace {
constexpr size_t limit = 1UL << 30;

TEST(GpunetioWritePairing, AdjacentRemoteRangesWithSameKey) {
    EXPECT_TRUE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4608, 576, 7, 3, limit));
}

TEST(GpunetioWritePairing, GapsReverseOrderAndDifferentKeysStaySeparate) {
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4609, 512, 7, 3, limit));
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 512, 7, 3584, 512, 7, 3, limit));
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4608, 512, 8, 3, limit));
}

TEST(GpunetioWritePairing, FinalDescriptorRemainsSeparate) {
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4608, 8, 7, 2, limit));
    EXPECT_TRUE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4608, 8, 7, 3, limit));
}

TEST(GpunetioWritePairing, EmptyRangesStaySeparate) {
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 0, 7, 4096, 512, 7, 3, limit));
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(4096, 512, 7, 4608, 0, 7, 3, limit));
}

TEST(GpunetioWritePairing, CombinedLengthDoesNotExceedSdkLimit) {
    EXPECT_TRUE(nixl::gpunetio::canPairWrite(0, limit / 2, 7, limit / 2, limit / 2, 7, 3, limit));
    EXPECT_FALSE(
        nixl::gpunetio::canPairWrite(0, limit / 2, 7, limit / 2, limit / 2 + 1, 7, 3, limit));
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(0, 1, 7, 1, limit + 1, 7, 3, limit));
}

TEST(GpunetioWritePairing, RemoteAddressArithmeticDoesNotWrap) {
    const auto end = std::numeric_limits<uintptr_t>::max();
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(end - 7, 8, 7, 0, 8, 7, 3, limit));
    EXPECT_FALSE(nixl::gpunetio::canPairWrite(end - 15, 8, 7, end - 7, 8, 7, 3, limit));
    EXPECT_TRUE(nixl::gpunetio::canPairWrite(end - 15, 8, 7, end - 7, 7, 7, 3, limit));
}
} // namespace
