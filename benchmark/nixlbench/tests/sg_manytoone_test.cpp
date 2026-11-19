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

#include "utils/transfer_layout.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

// Each test runs in a child because benchmark configuration is process-global.
void
checkSgLayout(int workers,
              size_t block,
              size_t batch,
              int depth,
              int warmup,
              int measured,
              size_t region,
              bool hugepages) {
    xferBenchConfig::mode = XFERBENCH_MODE_SG;
    xferBenchConfig::scheme = XFERBENCH_SCHEME_MANY_TO_ONE;
    xferBenchConfig::num_initiator_dev = 4;
    xferBenchConfig::num_target_dev = 1;
    xferBenchConfig::num_threads = workers;
    xferBenchConfig::total_buffer_size = region * 4 * workers;
    xferBenchConfig::use_hugepages = hugepages;
    xferBenchConfig::use_device_api = false;
    xferBenchConfig::pipeline_depth = depth;
    xferBenchConfig::warmup_iter = warmup * workers;
    xferBenchConfig::num_iter = measured * workers;
    xferBenchConfig::large_blk_iter_ftr = 1;
    const auto plan = xferBenchConfig::iterationPlan(block);
    ASSERT_EQ(plan.descriptor_depth, std::min(depth, std::max(warmup, measured)));
    const size_t entries = batch * plan.descriptor_depth;
    ASSERT_LE(entries * block, region);
    ASSERT_EQ(nixlbench::deviceBufferSize(true, workers), region);
    ASSERT_EQ(nixlbench::deviceBufferSize(false, workers), region * 4);

    std::vector<std::vector<unsigned char>> buffers(workers,
                                                    std::vector<unsigned char>(region * 4, 0xcc));
    std::vector<std::vector<xferBenchIOV>> allocated;
    for (auto &buffer : buffers) {
        allocated.push_back(
            {xferBenchIOV(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(), 0, "target")});
    }
    const auto target =
        nixlbench::createTransferDescLists(false, allocated, block, entries, workers);
    const auto peers =
        nixlbench::exchangePeerRanks(true, XFERBENCH_SCHEME_MANY_TO_ONE, false, 4, 4, 1);
    ASSERT_EQ(peers, std::vector<int>({0, 1, 2, 3}));
    ASSERT_EQ(target.size(), static_cast<size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        ASSERT_EQ(target[worker].size(), 4 * entries);
        for (size_t peer = 0; peer < peers.size(); ++peer) {
            const auto range =
                nixlbench::manyToOneDescriptorRange(target[worker].size(), peers.size(), peer);
            ASSERT_TRUE(range);
            ASSERT_EQ(range->count, entries);
            const std::vector<xferBenchIOV> exchanged(target[worker].begin() + range->offset,
                                                      target[worker].begin() + range->offset +
                                                          range->count);
            std::vector<unsigned char> sender(region, static_cast<unsigned char>(peer + 1));
            const auto local = nixlbench::createTransferDescLists(
                true,
                {{xferBenchIOV(reinterpret_cast<uintptr_t>(sender.data()), sender.size(), 0)}},
                block,
                entries,
                workers);
            ASSERT_EQ(local[0].size(), exchanged.size());
            for (size_t j = 0; j < entries; ++j) {
                const auto &remote = exchanged[j];
                const size_t offset = remote.addr - allocated[worker][0].addr;
                EXPECT_EQ(offset, peer * region + j * block);
                EXPECT_EQ(remote.len, block);
                EXPECT_EQ(remote.devId, 0);
                EXPECT_EQ(remote.metaInfo, "target");
                ASSERT_LE(offset + block, (peer + 1) * region);
                ASSERT_LE(offset + block, buffers[worker].size());
                EXPECT_EQ(local[0][j].addr - reinterpret_cast<uintptr_t>(sender.data()), j * block);
                ASSERT_LE(j * block + block, sender.size());
                std::memcpy(reinterpret_cast<void *>(remote.addr),
                            reinterpret_cast<const void *>(local[0][j].addr),
                            block);
            }
        }
        // Distinct source bytes and untouched tails independently expose aliased peer regions.
        for (size_t offset = 0; offset < buffers[worker].size(); ++offset) {
            const unsigned char expected = offset % region < entries * block ?
                static_cast<unsigned char>(offset / region + 1) :
                0xcc;
            ASSERT_EQ(buffers[worker][offset], expected) << "offset=" << offset;
        }
    }
}

TEST(SgProductionLayoutTest, FourExactFitSenderRegions) {
    EXPECT_EXIT(
        {
            checkSgLayout(1, 64 * 1024, 1, 1, 0, 1, 64 * 1024, false);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(SgProductionLayoutTest, MultipleWorkersBatchesAndUnequalPhases) {
    EXPECT_EXIT(
        {
            checkSgLayout(2, 4096, 2, 3, 1, 7, 32768, false);
            checkSgLayout(3, 4096, 2, 8, 3, 1, 32768, false);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(SgProductionLayoutTest, HugepageAlignedRegionsWithUntouchedTails) {
    EXPECT_EXIT(
        {
            checkSgLayout(2, 4096, 2, 3, 5, 1, HUGEPAGE_SIZE, true);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0),
        "");
}

} // namespace
