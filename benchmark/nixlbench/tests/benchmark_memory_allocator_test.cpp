/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/nixl_storage_allocator.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace nixlbench {
namespace {

TEST(BenchmarkMemoryAllocatorTest, DramLocalStrategyAllocatesAndCleansUpPerThreadIovs) {
    dramLocalIovStrategy strategy;

    auto result = strategy.create(2, 4096);
    ASSERT_FALSE(std::holds_alternative<int>(result));

    auto iov_lists = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(result));
    ASSERT_EQ(iov_lists.size(), 2U);
    for (const auto &iov_list : iov_lists) {
        ASSERT_EQ(iov_list.size(), 1U);
        EXPECT_NE(iov_list[0].addr, 0U);
        EXPECT_EQ(iov_list[0].len, 4096U);
        EXPECT_EQ(iov_list[0].devId, 0);
    }

    strategy.cleanup(iov_lists);
    EXPECT_TRUE(iov_lists.empty());
}

TEST(BenchmarkMemoryAllocatorTest, FileRemoteStrategyCreatesFileAndTransferIovs) {
    storageConfig config;
    config.filepath = std::filesystem::temp_directory_path().string();
    config.num_files = 1;

    fileRemoteIovStrategy strategy(config, "POSIX", XFERBENCH_OP_WRITE);
    auto create_result = strategy.create(2, 4096);
    ASSERT_FALSE(std::holds_alternative<int>(create_result));
    auto storage_iovs = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(create_result));

    ASSERT_EQ(storage_iovs.size(), 2U);
    ASSERT_EQ(storage_iovs[0].size(), 1U);
    ASSERT_EQ(storage_iovs[1].size(), 1U);
    EXPECT_EQ(storage_iovs[0][0].addr, 0U);
    EXPECT_EQ(storage_iovs[1][0].addr, 4096U);
    EXPECT_EQ(storage_iovs[0][0].devId, storage_iovs[1][0].devId);

    std::vector<std::vector<xferBenchIOV>> local_iovs{
        {xferBenchIOV(1000, 8192, 0)},
        {xferBenchIOV(2000, 8192, 0)},
    };
    auto transfer_result = strategy.createTransferIovs(local_iovs, 1024);
    ASSERT_FALSE(std::holds_alternative<int>(transfer_result));
    auto transfer_iovs = std::get<std::vector<std::vector<xferBenchIOV>>>(
        std::move(transfer_result));

    ASSERT_EQ(transfer_iovs.size(), 2U);
    EXPECT_EQ(transfer_iovs[0][0].addr, 0U);
    EXPECT_EQ(transfer_iovs[0][0].len, 1024U);
    EXPECT_EQ(transfer_iovs[1][0].addr, 0U);
    EXPECT_EQ(transfer_iovs[1][0].len, 1024U);
    EXPECT_EQ(transfer_iovs[0][0].devId, storage_iovs[0][0].devId);

    strategy.cleanup(storage_iovs);
    EXPECT_TRUE(storage_iovs.empty());
}

TEST(BenchmarkMemoryAllocatorTest, ObjectRemoteStrategyPreservesObjectMetadataForTransfers) {
    objectStorageConfig config;
    config.name_prefix = "test_obj";
    config.num_devices = 2;
    objectRemoteIovStrategy strategy(config);

    auto create_result = strategy.create(1, 2048);
    ASSERT_FALSE(std::holds_alternative<int>(create_result));
    auto storage_iovs = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(create_result));
    ASSERT_EQ(storage_iovs.size(), 1U);
    ASSERT_EQ(storage_iovs[0].size(), 2U);
    EXPECT_EQ(storage_iovs[0][0].addr, 0U);
    EXPECT_EQ(storage_iovs[0][0].len, 2048U);
    EXPECT_FALSE(storage_iovs[0][0].metaInfo.empty());

    auto transfer_result = strategy.createTransferIovs(storage_iovs, 512);
    ASSERT_FALSE(std::holds_alternative<int>(transfer_result));
    auto transfer_iovs = std::get<std::vector<std::vector<xferBenchIOV>>>(
        std::move(transfer_result));

    ASSERT_EQ(transfer_iovs.size(), 1U);
    ASSERT_EQ(transfer_iovs[0].size(), 2U);
    EXPECT_EQ(transfer_iovs[0][0].addr, 0U);
    EXPECT_EQ(transfer_iovs[0][0].len, 512U);
    EXPECT_EQ(transfer_iovs[0][0].metaInfo, storage_iovs[0][0].metaInfo);
}

TEST(BenchmarkMemoryAllocatorTest, BlockRemoteStrategyUsesConfiguredDeviceOffsets) {
    std::vector<blockStorageDevice> devices{{7, 128}, {8, 256}};
    blockRemoteIovStrategy strategy(devices);

    auto create_result = strategy.create(1, 4096);
    ASSERT_FALSE(std::holds_alternative<int>(create_result));
    auto storage_iovs = std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(create_result));

    ASSERT_EQ(storage_iovs.size(), 1U);
    ASSERT_EQ(storage_iovs[0].size(), 2U);
    EXPECT_EQ(storage_iovs[0][0].addr, 128U);
    EXPECT_EQ(storage_iovs[0][0].devId, 7);
    EXPECT_EQ(storage_iovs[0][1].addr, 256U);
    EXPECT_EQ(storage_iovs[0][1].devId, 8);
}

TEST(BenchmarkMemoryAllocatorTest, LocalStrategyFactoryRejectsUnknownSegment) {
    EXPECT_NE(makeLocalIovStrategy(XFERBENCH_SEG_TYPE_DRAM), nullptr);
    EXPECT_EQ(makeLocalIovStrategy("UNKNOWN"), nullptr);
}

} // namespace
} // namespace nixlbench
