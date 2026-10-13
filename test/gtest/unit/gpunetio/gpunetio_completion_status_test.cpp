/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_completion_status.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>

TEST(GpunetioCompletionStatus, ReadsCurrentErrorLatch) {
    uint32_t flag = 0;
    EXPECT_EQ(nixl::gpunetio::completionStatus(&flag), NIXL_SUCCESS);
    for (const uint32_t value : {1u, 2u, std::numeric_limits<uint32_t>::max()}) {
        flag = value;
        EXPECT_EQ(nixl::gpunetio::completionStatus(&flag), NIXL_ERR_BACKEND);
    }
}

TEST(GpunetioCompletionStatus, ReservesMatchingSlotAndAddress) {
    std::atomic<uint32_t> producer{3};
    uint32_t slot = 0;
    std::array<uint8_t, 64> buffer{};
    auto *address = nixl::gpunetio::reserveNotificationSlot(producer, 8, buffer.data(), 8, slot);
    EXPECT_EQ(slot, 3u);
    EXPECT_EQ(address, buffer.data() + 24);
    EXPECT_EQ(producer.load(), 4u);
}

TEST(GpunetioCompletionStatus, WrapsWithoutCrossingBufferGuards) {
    std::atomic<uint32_t> producer{7};
    uint32_t slot = 0;
    std::array<uint8_t, 66> guarded;
    guarded.fill(0xa5);
    for (uint32_t iteration = 0; iteration < 17; ++iteration) {
        auto *address =
            nixl::gpunetio::reserveNotificationSlot(producer, 8, guarded.data() + 1, 8, slot);
        ASSERT_LT(slot, 8u);
        EXPECT_EQ(slot, (iteration + 7) % 8);
        EXPECT_EQ(address, guarded.data() + 1 + slot * 8);
        std::fill_n(address, 8, static_cast<uint8_t>(iteration));
    }
    EXPECT_EQ(producer.load(), 24u);
    EXPECT_EQ(guarded.front(), 0xa5);
    EXPECT_EQ(guarded.back(), 0xa5);
}

TEST(GpunetioCompletionStatus, KeepsPeerProducerCountersIndependent) {
    std::atomic<uint32_t> first{0}, second{0};
    uint32_t first_slot = 0, second_slot = 0;
    std::array<uint8_t, 64> first_buffer{}, second_buffer{};
    auto *first_address =
        nixl::gpunetio::reserveNotificationSlot(first, 8, first_buffer.data(), 8, first_slot);
    auto *second_address =
        nixl::gpunetio::reserveNotificationSlot(second, 8, second_buffer.data(), 8, second_slot);
    EXPECT_EQ(first_slot, 0u);
    EXPECT_EQ(second_slot, 0u);
    EXPECT_EQ(first_address, first_buffer.data());
    EXPECT_EQ(second_address, second_buffer.data());
    EXPECT_EQ(first.load(), 1u);
    EXPECT_EQ(second.load(), 1u);
}
