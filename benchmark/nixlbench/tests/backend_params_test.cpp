/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/utils.h"

#include <gtest/gtest.h>

#include <string>

namespace {

TEST(BackendParamsTest, EmptyGivesNoParameters) {
    nixl_b_params_t params = {{"stale", "value"}};
    std::string error;
    ASSERT_TRUE(xferBenchConfig::parseBackendParams("", params, error));
    EXPECT_TRUE(params.empty());
}

TEST(BackendParamsTest, ValuesAreKeptAsWritten) {
    nixl_b_params_t params;
    std::string error;
    ASSERT_TRUE(xferBenchConfig::parseBackendParams(
        "rdma_nics=mlx5_1,mlx5_2;split_size=4194304;;url=http://h:1/a=b;empty=", params, error));
    const nixl_b_params_t expected = {{"rdma_nics", "mlx5_1,mlx5_2"},
                                      {"split_size", "4194304"},
                                      {"url", "http://h:1/a=b"},
                                      {"empty", ""}};
    EXPECT_EQ(params, expected);
}

TEST(BackendParamsTest, MalformedEntriesAreRejected) {
    nixl_b_params_t params;
    std::string error;
    EXPECT_FALSE(xferBenchConfig::parseBackendParams("split_size", params, error));
    EXPECT_EQ(error, "'split_size' is not key=value");
    EXPECT_FALSE(xferBenchConfig::parseBackendParams("=4096", params, error));
    EXPECT_FALSE(xferBenchConfig::parseBackendParams("a=1;a=2", params, error));
    EXPECT_EQ(error, "'a' is given twice");
}

} // namespace
