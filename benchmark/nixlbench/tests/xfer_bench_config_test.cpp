/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/utils.h"

#include <gtest/gtest.h>

namespace {

TEST(XferBenchConfigTest, InstancesHoldIndependentValues) {
    xferBenchConfig first;
    xferBenchConfig second;

    first.backend = XFERBENCH_BACKEND_UCX;
    first.device_list = "mlx5_0";
    first.num_initiator_dev = 1;
    first.num_target_dev = 1;
    first.total_buffer_size = 1024;

    second.backend = XFERBENCH_BACKEND_POSIX;
    second.device_list = "mlx5_1,mlx5_2";
    second.num_initiator_dev = 2;
    second.num_target_dev = 2;
    second.total_buffer_size = 2048;

    EXPECT_EQ(first.backend, XFERBENCH_BACKEND_UCX);
    EXPECT_EQ(first.device_list, "mlx5_0");
    EXPECT_EQ(first.num_initiator_dev, 1);
    EXPECT_EQ(first.num_target_dev, 1);
    EXPECT_EQ(first.total_buffer_size, 1024U);
    EXPECT_FALSE(first.isStorageBackend());

    EXPECT_EQ(second.backend, XFERBENCH_BACKEND_POSIX);
    EXPECT_EQ(second.device_list, "mlx5_1,mlx5_2");
    EXPECT_EQ(second.num_initiator_dev, 2);
    EXPECT_EQ(second.num_target_dev, 2);
    EXPECT_EQ(second.total_buffer_size, 2048U);
    EXPECT_TRUE(second.isStorageBackend());
}

TEST(XferBenchConfigTest, DeviceListParsingUsesInstanceValues) {
    xferBenchConfig single;
    single.device_list = "mlx5_0";
    single.num_initiator_dev = 1;
    single.num_target_dev = 1;

    xferBenchConfig pair;
    pair.device_list = "mlx5_1,mlx5_2";
    pair.num_initiator_dev = 2;
    pair.num_target_dev = 2;

    EXPECT_EQ(single.parseDeviceList(), std::vector<std::string>({"mlx5_0"}));
    EXPECT_EQ(pair.parseDeviceList(), std::vector<std::string>({"mlx5_1", "mlx5_2"}));
}

TEST(XferBenchConfigTest, CliHelpRequestStateIsInstanceLocal) {
    xferBenchConfig first;
    xferBenchConfig second;

    first.cli_help_requested = true;
    second.cli_help_requested = false;

    EXPECT_TRUE(first.cliHelpRequested());
    EXPECT_FALSE(second.cliHelpRequested());
}

} // namespace
