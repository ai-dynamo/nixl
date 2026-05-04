/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "worker/worker.h"

#include "utils/utils.h"

#include <gtest/gtest.h>

namespace nixlbench {
namespace {

TEST(WorkerConfigTest, SingleGroupRuntimeUsesAllInitiatorAndTargetRanks) {
    benchmarkConfig config;
    config.transfer.mode = XFERBENCH_MODE_SG;
    config.worker.num_initiator_dev = 2;
    config.worker.num_target_dev = 3;

    EXPECT_EQ(runtimeWorldSize(config), 5);
    EXPECT_EQ(rankRoleName(config, 0), "initiator");
    EXPECT_EQ(rankRoleName(config, 1), "initiator");
    EXPECT_EQ(rankRoleName(config, 2), "target");
    EXPECT_EQ(rankRoleName(config, 4), "target");
}

TEST(WorkerConfigTest, MultiGroupRuntimeUsesTwoRanksWithRankZeroInitiating) {
    benchmarkConfig config;
    config.transfer.mode = XFERBENCH_MODE_MG;
    config.worker.num_initiator_dev = 4;
    config.worker.num_target_dev = 5;

    EXPECT_EQ(runtimeWorldSize(config), 2);
    EXPECT_EQ(rankRoleName(config, 0), "initiator");
    EXPECT_EQ(rankRoleName(config, 1), "target");
}

TEST(WorkerConfigTest, StorageBackendWithoutEtcdUsesNullRuntimeInitiatorRole) {
    benchmarkConfig config;
    config.backend.name = XFERBENCH_BACKEND_POSIX;
    config.backend.capabilities.canUseAsStorage = true;
    config.runtime.etcd_endpoints = "";
    config.transfer.mode = XFERBENCH_MODE_SG;
    config.worker.num_initiator_dev = 3;
    config.worker.num_target_dev = 7;

    EXPECT_TRUE(usesNullRuntime(config));
    EXPECT_EQ(runtimeWorldSize(config), 1);
    EXPECT_EQ(rankRoleName(config, 0), "initiator");
}

} // namespace
} // namespace nixlbench
