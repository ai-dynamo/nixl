/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_runner_config.h"

#include "utils/utils.h"

#include <gtest/gtest.h>

namespace nixlbench {
namespace {

benchmarkConfig
makeStrideConfig(const std::string &scheme, int initiator_devices, int target_devices) {
    benchmarkConfig config;
    config.transfer.scheme = scheme;
    config.transfer.total_buffer_size = 3600;
    config.worker.num_initiator_dev = initiator_devices;
    config.worker.num_target_dev = target_devices;
    return config;
}

TEST(BenchmarkRunnerConfigTest, PairwiseUsesOneStrideForEitherRole) {
    const benchmarkConfig config = makeStrideConfig(XFERBENCH_SCHEME_PAIRWISE, 2, 3);

    EXPECT_EQ(getStrideScheme(config, true, false, 2),
              (std::make_pair<size_t, size_t>(1, 900)));
    EXPECT_EQ(getStrideScheme(config, false, true, 2),
              (std::make_pair<size_t, size_t>(1, 900)));
}

TEST(BenchmarkRunnerConfigTest, OneToManyInitiatorStridesAcrossTargets) {
    const benchmarkConfig config = makeStrideConfig(XFERBENCH_SCHEME_ONE_TO_MANY, 2, 3);

    EXPECT_EQ(getStrideScheme(config, true, false, 2),
              (std::make_pair<size_t, size_t>(3, 300)));
    EXPECT_EQ(getStrideScheme(config, false, true, 2),
              (std::make_pair<size_t, size_t>(1, 900)));
}

TEST(BenchmarkRunnerConfigTest, ManyToOneTargetStridesAcrossInitiators) {
    const benchmarkConfig config = makeStrideConfig(XFERBENCH_SCHEME_MANY_TO_ONE, 3, 2);

    EXPECT_EQ(getStrideScheme(config, true, false, 2),
              (std::make_pair<size_t, size_t>(1, 600)));
    EXPECT_EQ(getStrideScheme(config, false, true, 2),
              (std::make_pair<size_t, size_t>(3, 200)));
}

TEST(BenchmarkRunnerConfigTest, TpInitiatorStridesWhenTargetsOutnumberInitiators) {
    const benchmarkConfig config = makeStrideConfig(XFERBENCH_SCHEME_TP, 2, 6);

    EXPECT_EQ(getStrideScheme(config, true, false, 2),
              (std::make_pair<size_t, size_t>(3, 300)));
    EXPECT_EQ(getStrideScheme(config, false, true, 2),
              (std::make_pair<size_t, size_t>(1, 900)));
}

TEST(BenchmarkRunnerConfigTest, TpTargetStridesWhenInitiatorsOutnumberTargets) {
    const benchmarkConfig config = makeStrideConfig(XFERBENCH_SCHEME_TP, 6, 2);

    EXPECT_EQ(getStrideScheme(config, true, false, 2),
              (std::make_pair<size_t, size_t>(1, 300)));
    EXPECT_EQ(getStrideScheme(config, false, true, 2),
              (std::make_pair<size_t, size_t>(3, 100)));
}

} // namespace
} // namespace nixlbench
