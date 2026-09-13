/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "worker/nixl/nixl_topology.h"

#include <gtest/gtest.h>

#include <vector>

namespace nixlbench {
namespace {

    TEST(SgManyToOneTopologyTest, RoutesEveryInitiatorToTheSingleTarget) {
        constexpr int initiators = 4;
        constexpr int targets = 1;

        for (int rank = 0; rank < initiators; ++rank) {
            EXPECT_EQ(exchangePeerRanks(
                          true, XFERBENCH_SCHEME_MANY_TO_ONE, true, rank, initiators, targets),
                      std::vector<int>({initiators}));
        }

        EXPECT_EQ(exchangePeerRanks(
                      true, XFERBENCH_SCHEME_MANY_TO_ONE, false, initiators, initiators, targets),
                  std::vector<int>({0, 1, 2, 3}));
    }

    TEST(SgManyToOneTopologyTest, PreservesPairwiseRankMapping) {
        EXPECT_EQ(exchangePeerRanks(true, XFERBENCH_SCHEME_PAIRWISE, true, 2, 4, 4),
                  std::vector<int>({6}));
        EXPECT_EQ(exchangePeerRanks(true, XFERBENCH_SCHEME_PAIRWISE, false, 6, 4, 4),
                  std::vector<int>({2}));
    }

    TEST(SgManyToOneTopologyTest, PreservesMultiGpuPeerMapping) {
        EXPECT_EQ(exchangePeerRanks(false, XFERBENCH_SCHEME_MANY_TO_ONE, true, 0, 4, 1),
                  std::vector<int>({1}));
        EXPECT_EQ(exchangePeerRanks(false, XFERBENCH_SCHEME_MANY_TO_ONE, false, 1, 4, 1),
                  std::vector<int>({0}));
    }

    TEST(SgManyToOneTopologyTest, AssignsOneDescriptorPartitionPerInitiator) {
        constexpr size_t descriptor_count = 16;
        constexpr size_t peer_count = 4;

        for (size_t peer = 0; peer < peer_count; ++peer) {
            const auto range = manyToOneDescriptorRange(descriptor_count, peer_count, peer);
            ASSERT_TRUE(range);
            EXPECT_EQ(range->offset, peer * 4);
            EXPECT_EQ(range->count, 4U);
        }
    }

    TEST(SgManyToOneTopologyTest, RejectsAliasingDescriptorPartitions) {
        EXPECT_FALSE(manyToOneDescriptorRange(0, 4, 0));
        EXPECT_FALSE(manyToOneDescriptorRange(3, 4, 0));
        EXPECT_FALSE(manyToOneDescriptorRange(4, 0, 0));
        EXPECT_FALSE(manyToOneDescriptorRange(4, 4, 4));
    }

} // namespace
} // namespace nixlbench
