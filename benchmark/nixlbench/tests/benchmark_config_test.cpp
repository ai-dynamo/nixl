/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_config.h"

#include <gtest/gtest.h>

namespace nixlbench {
namespace {

TEST(BenchmarkConfigTest, DefaultsMatchLegacyFlagDefaults) {
    const benchmarkConfig config;

    EXPECT_EQ(config.common.benchmark_group, "default");
    EXPECT_FALSE(config.common.check_consistency);
    EXPECT_FALSE(config.common.recreate_xfer);
    EXPECT_EQ(config.common.num_iter, 1000);
    EXPECT_EQ(config.common.large_blk_iter_ftr, 16);
    EXPECT_EQ(config.common.warmup_iter, 100);

    EXPECT_EQ(config.runtime.type, "ETCD");
    EXPECT_TRUE(config.runtime.etcd_endpoints.empty());

    EXPECT_EQ(config.worker.type, "nixl");
    EXPECT_EQ(config.worker.num_initiator_dev, 1);
    EXPECT_EQ(config.worker.num_target_dev, 1);
    EXPECT_FALSE(config.worker.enable_pt);
    EXPECT_EQ(config.worker.progress_threads, 0U);
    EXPECT_EQ(config.worker.device_list, "all");
    EXPECT_FALSE(config.worker.enable_vmm);

    EXPECT_EQ(config.transfer.initiator_seg_type, "DRAM");
    EXPECT_EQ(config.transfer.target_seg_type, "DRAM");
    EXPECT_EQ(config.transfer.scheme, "pairwise");
    EXPECT_EQ(config.transfer.mode, "SG");
    EXPECT_EQ(config.transfer.op_type, "WRITE");
    EXPECT_EQ(config.transfer.total_buffer_size, 8ULL * 1024ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(config.transfer.start_block_size, 4ULL * 1024ULL);
    EXPECT_EQ(config.transfer.max_block_size, 64ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(config.transfer.start_batch_size, 1U);
    EXPECT_EQ(config.transfer.max_batch_size, 1U);
    EXPECT_EQ(config.transfer.num_threads, 1);

    EXPECT_EQ(config.backend.name, "UCX");
    EXPECT_FALSE(config.backend.capabilities.canUseAsStorage);
    EXPECT_FALSE(config.backend.capabilities.canUseAsNetworkDestination);
    EXPECT_FALSE(config.backend.capabilities.canReadWriteFiles);
    EXPECT_TRUE(config.backend.options.empty());

    EXPECT_TRUE(config.storage.filepath.empty());
    EXPECT_TRUE(config.storage.filenames.empty());
    EXPECT_EQ(config.storage.num_files, 1);
    EXPECT_FALSE(config.storage.enable_direct);
}

TEST(BenchmarkConfigTest, IdentifiesStorageBackendsByCapabilities) {
    benchmarkConfig config;
    config.backend.name = "CUSTOM_STORAGE";
    config.backend.capabilities.canUseAsStorage = true;

    EXPECT_TRUE(isStorageBackend(config.backend));
}

TEST(BenchmarkConfigTest, NonStorageBackendIsNotStorage) {
    const benchmarkConfig config;

    EXPECT_FALSE(isStorageBackend(config.backend));
}

} // namespace
} // namespace nixlbench
