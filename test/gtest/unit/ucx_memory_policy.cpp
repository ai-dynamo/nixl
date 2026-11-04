// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../../../src/plugins/ucx/memory_policy.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>

TEST(UcxMemoryPolicy, DefaultPreservesStockDescriptorHandling) {
    const nixl::ucx::MemoryRegistrationPolicy policy;
    const nixlBlobDesc desc(0, 0, 0);
    EXPECT_FALSE(policy.isStrict());
    EXPECT_EQ(policy.validateBeforeMap(desc, FILE_SEG), NIXL_SUCCESS);
}

TEST(UcxMemoryPolicy, NamedPolicyEnforcesCommonChecksWithoutCallback) {
    const nixl::ucx::MemoryRegistrationPolicy policy{"musa", {}};
    EXPECT_TRUE(policy.isStrict());
    EXPECT_EQ(policy.validateBeforeMap(nixlBlobDesc(0x1000, 64, 0), VRAM_SEG), NIXL_SUCCESS);
    EXPECT_EQ(policy.validateBeforeMap(nixlBlobDesc(0x1000, 64, 0), FILE_SEG),
              NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(policy.validateBeforeMap(nixlBlobDesc(0, 64, 0), DRAM_SEG), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(policy.validateBeforeMap(nixlBlobDesc(0x1000, 0, 0), VRAM_SEG), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(policy.validateBeforeMap(
                  nixlBlobDesc(std::numeric_limits<uintptr_t>::max() - 1, 2, 0), VRAM_SEG),
              NIXL_ERR_INVALID_PARAM);
}

TEST(UcxMemoryPolicy, ProviderCallbackRunsOnlyAfterCommonChecks) {
    int calls = 0;
    nixl::ucx::MemoryRegistrationPolicy policy{
        "musa", [&](const nixlBlobDesc &desc, nixl_mem_t type) {
            ++calls;
            EXPECT_EQ(desc.devId, 3);
            EXPECT_EQ(type, VRAM_SEG);
            return NIXL_ERR_MISMATCH;
        }};
    EXPECT_EQ(policy.validateBeforeMap(nixlBlobDesc(0, 64, 3), VRAM_SEG), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(calls, 0);
    auto moved_policy = std::move(policy);
    EXPECT_EQ(moved_policy.validateBeforeMap(nixlBlobDesc(0x1000, 64, 3), VRAM_SEG), NIXL_ERR_MISMATCH);
    EXPECT_EQ(calls, 1);
}

TEST(UcxMemoryPolicy, ProviderExceptionsReachRegistrationBoundary) {
    const nixl::ucx::MemoryRegistrationPolicy policy{
        "musa", [](const nixlBlobDesc &, nixl_mem_t) -> nixl_status_t {
            throw std::runtime_error("provider failure");
        }};
    EXPECT_THROW(policy.validateBeforeMap(nixlBlobDesc(0x1000, 64, 0), VRAM_SEG), std::runtime_error);
}
