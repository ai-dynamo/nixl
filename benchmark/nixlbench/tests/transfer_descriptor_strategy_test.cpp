/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/transfer_descriptor_strategy.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <algorithm>
#include <random>
#include <variant>
#include <vector>

namespace nixlbench {
namespace {

transferDescriptorConfig
makeConfig() {
    transferDescriptorConfig config;
    config.block_size = 10;
    config.batch_size = 2;
    config.num_threads = 1;
    config.num_initiator_dev = 1;
    config.num_target_dev = 4;
    config.total_buffer_size = 400;
    config.scheme = XFERBENCH_SCHEME_ONE_TO_MANY;
    config.mode = XFERBENCH_MODE_SG;
    config.is_initiator = true;
    config.is_target = false;
    return config;
}

std::vector<std::vector<xferBenchIOV>>
makeIovLists() {
    return {{xferBenchIOV(1000, 400, 7, "meta")}};
}

TEST(TransferDescriptorStrategyTest, SequentialGenerationUsesStrideAndBatchOffsets) {
    auto config = makeConfig();
    auto iov_lists = makeIovLists();

    auto result = createTransferDescLists(config, iov_lists, false);

    ASSERT_FALSE(std::holds_alternative<int>(result));
    auto descriptors = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(result));
    ASSERT_EQ(descriptors.size(), 1U);
    ASSERT_EQ(descriptors[0].size(), 8U);

    const std::vector<uintptr_t> expected_addresses{1000, 1010, 1100, 1110,
                                                    1200, 1210, 1300, 1310};
    for (std::size_t i = 0; i < expected_addresses.size(); ++i) {
        EXPECT_EQ(descriptors[0][i].addr, expected_addresses[i]);
        EXPECT_EQ(descriptors[0][i].len, 10U);
        EXPECT_EQ(descriptors[0][i].devId, 7);
        EXPECT_EQ(descriptors[0][i].metaInfo, "meta");
    }
}

TEST(TransferDescriptorStrategyTest, RandomizedGenerationShufflesStrideOffsets) {
    auto config = makeConfig();
    auto iov_lists = makeIovLists();
    std::mt19937 rng(0);

    auto result = createTransferDescLists(config, iov_lists, true, &rng);

    ASSERT_FALSE(std::holds_alternative<int>(result));
    auto descriptors = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(result));
    ASSERT_EQ(descriptors.size(), 1U);
    ASSERT_EQ(descriptors[0].size(), 8U);

    const std::vector<uintptr_t> sequential_addresses{1000, 1010, 1100, 1110,
                                                      1200, 1210, 1300, 1310};
    std::vector<uintptr_t> actual_addresses;
    actual_addresses.reserve(descriptors[0].size());
    for (const auto &descriptor : descriptors[0]) {
        actual_addresses.push_back(descriptor.addr);
    }

    EXPECT_NE(actual_addresses, sequential_addresses);

    std::vector<uintptr_t> sorted_actual = actual_addresses;
    std::vector<uintptr_t> sorted_expected = sequential_addresses;
    std::sort(sorted_actual.begin(), sorted_actual.end());
    std::sort(sorted_expected.begin(), sorted_expected.end());
    EXPECT_EQ(sorted_actual, sorted_expected);
}

TEST(TransferDescriptorStrategyTest, StrategyUsesAllocationLocalIovs) {
    auto config = makeConfig();
    benchmarkAllocation allocation;
    allocation.local_iovs = makeIovLists();
    offsetTransferDescriptorStrategy strategy(config, false);

    auto result = strategy.create(allocation);

    ASSERT_FALSE(std::holds_alternative<int>(result));
    auto descriptors = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(result));
    ASSERT_EQ(descriptors.size(), 1U);
    ASSERT_EQ(descriptors[0].size(), 8U);
    EXPECT_EQ(descriptors[0][0].addr, 1000U);
}

TEST(TransferDescriptorStrategyTest, FixedIterationPolicyEncodesAllocationLifecycle) {
    fixedIterationPolicy allocate_once(2, benchmarkAllocationLifecycle::AllocateOnce);
    EXPECT_TRUE(allocate_once.allocateOnce());
    EXPECT_TRUE(allocate_once.hasNext());
    allocate_once.advance();
    EXPECT_TRUE(allocate_once.hasNext());
    allocate_once.advance();
    EXPECT_FALSE(allocate_once.hasNext());

    fixedIterationPolicy per_iteration(1, benchmarkAllocationLifecycle::AllocatePerIteration);
    EXPECT_FALSE(per_iteration.allocateOnce());
}

} // namespace
} // namespace nixlbench
