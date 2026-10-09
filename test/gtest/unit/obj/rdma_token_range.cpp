/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Address math for cuMemObjGetRDMAToken. cuObject accepts only the pointer a
// buffer was registered with; a transfer inside that buffer has to be expressed
// as (base, offset). These tests do not need libcuobjclient.

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "object/rdma/rdma_token_range.h"

namespace {

using namespace nixl_obj_rdma;

constexpr uintptr_t kArena = 0x1000;
constexpr size_t kArenaLen = 0x1000;

TEST(RdmaTokenRange, WholeRegistrationKeepsZeroOffset) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    const auto token = resolveRdmaTokenRange(regs, kArena, kArenaLen, 0);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, kArena);
    EXPECT_EQ(token->offset, 0u);
}

TEST(RdmaTokenRange, InteriorPointerBecomesBasePlusOffset) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    const uintptr_t ptr = kArena + 0x800;
    const auto token = resolveRdmaTokenRange(regs, ptr, 0x100, 0);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, kArena);
    EXPECT_EQ(token->offset, 0x800u);
}

TEST(RdmaTokenRange, CallerOffsetIsAddedToTheInteriorOffset) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    const uintptr_t ptr = kArena + 0x800;
    const auto token = resolveRdmaTokenRange(regs, ptr, 0x100, 0x20);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, kArena);
    EXPECT_EQ(token->offset, 0x820u);
}

TEST(RdmaTokenRange, RangePastTheEndIsRejected) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    EXPECT_FALSE(resolveRdmaTokenRange(regs, kArena + 0xF00, 0x200, 0).has_value());
}

TEST(RdmaTokenRange, PointerOutsideEveryRegistrationIsRejected) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    EXPECT_FALSE(resolveRdmaTokenRange(regs, 0x100, 16, 0).has_value());
    EXPECT_FALSE(resolveRdmaTokenRange({}, kArena, 16, 0).has_value());
}

TEST(RdmaTokenRange, ExactBaseWinsOverALargerArena) {
    const uintptr_t sub = kArena + 0x200;
    const std::vector<RdmaRegistration> regs{{kArena, 0x10000}, {sub, 0x100}};
    const auto token = resolveRdmaTokenRange(regs, sub, 0x100, 0);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, sub);
    EXPECT_EQ(token->offset, 0u);
}

TEST(RdmaTokenRange, InteriorOfANestedRegistrationUsesTheSmallerOne) {
    const uintptr_t sub = kArena + 0x200;
    const std::vector<RdmaRegistration> regs{{kArena, 0x10000}, {sub, 0x100}};
    const auto token = resolveRdmaTokenRange(regs, sub + 0x80, 0x10, 0);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, sub);
    EXPECT_EQ(token->offset, 0x80u);
}

TEST(RdmaTokenRange, SubRegistrationThatCannotHoldTheRangeFallsBackToTheArena) {
    const uintptr_t sub = kArena + 0x200;
    const std::vector<RdmaRegistration> regs{{kArena, 0x10000}, {sub, 0x40}};
    const auto token = resolveRdmaTokenRange(regs, sub, 0x100, 0);
    ASSERT_TRUE(token.has_value());
    EXPECT_EQ(token->base, kArena);
    EXPECT_EQ(token->offset, 0x200u);
}

TEST(RdmaTokenRange, OverflowIsRejected) {
    const std::vector<RdmaRegistration> regs{{kArena, kArenaLen}};
    const uintptr_t ptr = std::numeric_limits<uintptr_t>::max() - 8;
    EXPECT_FALSE(resolveRdmaTokenRange(regs, ptr, 16, 16).has_value());
    EXPECT_FALSE(
        resolveRdmaTokenRange(regs, kArena, std::numeric_limits<size_t>::max(), 0).has_value());
}

} // namespace
