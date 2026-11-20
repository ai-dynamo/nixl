/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/utils.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

namespace {

class CapacityConfig : public xferBenchConfig {
public:
    using xferBenchConfig::loadParams;
};

int
load(std::initializer_list<std::pair<const char *, std::string>> overrides = {}) {
    gflags::FlagSaver flags;
    for (const auto &[name, value] : {std::pair{"config_file", ""},
                                      {"worker_type", "nixl"},
                                      {"backend", "UCX"},
                                      {"mode", "SG"},
                                      {"scheme", "manytoone"},
                                      {"runtime_type", "ASIO"},
                                      {"initiator_seg_type", "DRAM"},
                                      {"target_seg_type", "DRAM"},
                                      {"total_buffer_size", "33554432"},
                                      {"num_initiator_dev", "2"},
                                      {"num_target_dev", "1"},
                                      {"num_threads", "1"},
                                      {"num_iter", "16"},
                                      {"warmup_iter", "0"},
                                      {"pipeline_depth", "16"},
                                      {"large_blk_iter_ftr", "16"},
                                      {"start_block_size", "1048576"},
                                      {"max_block_size", "16777216"},
                                      {"start_batch_size", "1"},
                                      {"max_batch_size", "1"},
                                      {"use_device_api", "false"},
                                      {"use_hugepages", "false"}}) {
        if (gflags::SetCommandLineOption(name, value).empty()) {
            std::abort();
        }
    }
    for (const auto &[name, value] : overrides) {
        if (gflags::SetCommandLineOption(name, value.c_str()).empty()) {
            std::abort();
        }
    }
    return CapacityConfig::loadParams();
}

// Run configuration loading in a child so its static state cannot affect other tests.
TEST(CapacityTest, AcceptsMixedDepthSweepAtExactCapacity) {
    EXPECT_EXIT(std::_Exit(load() == 0 ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST(CapacityTest, RejectsBufferBelowExactCapacity) {
    EXPECT_EXIT(std::_Exit(load({{"total_buffer_size", "33554428"}}) != 0 ? 0 : 1),
                ::testing::ExitedWithCode(0),
                "SG manytoone requires.*block_size=1048576, descriptor_depth=16");
    // The same rejection must hold when the final block is reduced.
    EXPECT_EXIT(
        std::_Exit(
            load({{"total_buffer_size", "33554428"}, {"max_block_size", "2097152"}}) != 0 ? 0 : 1),
        ::testing::ExitedWithCode(0),
        "block_size=1048576, descriptor_depth=16");
}

TEST(CapacityTest, ChecksIntermediateSizesInSweep) {
    // The final 2 MiB cell needs one slot; the preceding 1 MiB cell needs sixteen.
    EXPECT_EXIT(
        std::_Exit(
            load({{"total_buffer_size", "4194304"}, {"max_block_size", "2097152"}}) != 0 ? 0 : 1),
        ::testing::ExitedWithCode(0),
        "block_size=1048576, descriptor_depth=16");
}

TEST(CapacityTest, UsesNormalizedIterations) {
    EXPECT_EXIT(
        {
            const int result = load({{"num_iter", "17"},
                                     {"start_block_size", "2097152"},
                                     {"max_block_size", "2097152"},
                                     {"total_buffer_size", "8388608"}});
            std::_Exit(result == 0 && xferBenchConfig::num_iter == 32 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
    EXPECT_EXIT(std::_Exit(load({{"num_iter", "17"},
                                 {"start_block_size", "2097152"},
                                 {"max_block_size", "2097152"},
                                 {"total_buffer_size", "4194304"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "descriptor_depth=2");
}

TEST(CapacityTest, ReservesSlotsForLongerWarmupPhase) {
    EXPECT_EXIT(std::_Exit(load({{"warmup_iter", "64"},
                                 {"start_block_size", "2097152"},
                                 {"max_block_size", "2097152"},
                                 {"total_buffer_size", "16777216"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
    EXPECT_EXIT(std::_Exit(load({{"warmup_iter", "64"},
                                 {"start_block_size", "2097152"},
                                 {"max_block_size", "2097152"},
                                 {"total_buffer_size", "8388608"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "descriptor_depth=4");
}

TEST(CapacityTest, PreservesWorkerAndSenderPartitions) {
    EXPECT_EXIT(std::_Exit(load({{"num_threads", "2"},
                                 {"num_iter", "32"},
                                 {"start_block_size", "2097152"},
                                 {"max_block_size", "4194304"},
                                 {"total_buffer_size", "16777216"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
    EXPECT_EXIT(std::_Exit(load({{"num_threads", "2"},
                                 {"num_iter", "32"},
                                 {"start_block_size", "2097152"},
                                 {"max_block_size", "4194304"},
                                 {"total_buffer_size", "16777208"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "block_size=4194304, descriptor_depth=1");
}

TEST(CapacityTest, IncludesMaximumBatchWidth) {
    EXPECT_EXIT(std::_Exit(load({{"start_block_size", "2097152"},
                                 {"max_block_size", "8388608"},
                                 {"max_batch_size", "2"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
    EXPECT_EXIT(std::_Exit(load({{"start_block_size", "2097152"},
                                 {"max_block_size", "16777216"},
                                 {"max_batch_size", "2"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "Incorrect buffer size configuration");
}

TEST(CapacityTest, UsesEmittedBatchLimit) {
    EXPECT_EXIT(std::_Exit(load({{"num_threads", "2"},
                                 {"num_iter", "2"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "4096"},
                                 {"max_batch_size", "3"},
                                 {"total_buffer_size", "32768"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
    EXPECT_EXIT(std::_Exit(load({{"num_threads", "2"},
                                 {"num_iter", "2"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "4096"},
                                 {"max_batch_size", "3"},
                                 {"total_buffer_size", "32764"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "SG manytoone requires.*descriptor_depth=1");
}

TEST(CapacityTest, UsesEmittedBlockLimit) {
    EXPECT_EXIT(std::_Exit(load({{"num_threads", "2"},
                                 {"num_iter", "2"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "6000"},
                                 {"total_buffer_size", "16384"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
}

TEST(CapacityTest, UsesEmittedLimitsBeforeGenericCapacityChecks) {
    EXPECT_EXIT(std::_Exit(load({{"num_iter", "1"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "7000"},
                                 {"max_batch_size", "3"},
                                 {"total_buffer_size", "16384"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
}

TEST(CapacityTest, DoublesFromConfiguredBatchStart) {
    EXPECT_EXIT(std::_Exit(load({{"num_iter", "1"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "4096"},
                                 {"start_batch_size", "3"},
                                 {"max_batch_size", "8"},
                                 {"total_buffer_size", "49152"}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
    EXPECT_EXIT(std::_Exit(load({{"num_iter", "1"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "4096"},
                                 {"max_block_size", "4096"},
                                 {"start_batch_size", "3"},
                                 {"max_batch_size", "8"},
                                 {"total_buffer_size", "49150"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "SG manytoone requires.*descriptor_depth=1");
}

TEST(CapacityTest, StopsBatchValidationBeforeDoublingWraps) {
    const auto large = std::to_string(std::numeric_limits<size_t>::max() / 2 + 1);
    EXPECT_EXIT(std::_Exit(load({{"num_initiator_dev", "1"},
                                 {"num_iter", "1"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", "1"},
                                 {"max_block_size", "1"},
                                 {"start_batch_size", large},
                                 {"max_batch_size", large},
                                 {"total_buffer_size", large}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
}

TEST(CapacityTest, RejectsBatchDepthOverflow) {
    const auto large = std::to_string(std::numeric_limits<size_t>::max() / 2 + 1);
    EXPECT_EXIT(std::_Exit(load({{"num_initiator_dev", "1"},
                                 {"num_iter", "2"},
                                 {"large_blk_iter_ftr", "1"},
                                 {"pipeline_depth", "2"},
                                 {"start_block_size", "2"},
                                 {"max_block_size", "2"},
                                 {"max_batch_size", large}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "Batch size and descriptor depth are too large");
}

TEST(CapacityTest, RejectsBlockBatchProductOverflow) {
    const auto large = std::to_string(std::numeric_limits<size_t>::max() / 2 + 1);
    EXPECT_EXIT(std::_Exit(load({{"num_initiator_dev", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", large},
                                 {"max_block_size", large},
                                 {"max_batch_size", "4"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "SG manytoone requires.*descriptor_depth=1");
}

TEST(CapacityTest, StopsValidationSweepBeforeDoublingWraps) {
    const auto large = std::to_string(std::numeric_limits<size_t>::max() / 2 + 1);
    EXPECT_EXIT(std::_Exit(load({{"num_initiator_dev", "1"},
                                 {"pipeline_depth", "1"},
                                 {"start_block_size", large},
                                 {"max_block_size", large},
                                 {"total_buffer_size", large}}) == 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "");
}

TEST(CapacityTest, RejectsEmptyOrZeroRanges) {
    for (const auto &[name, value] : {std::pair{"start_block_size", "0"},
                                      {"start_batch_size", "0"},
                                      {"max_block_size", "524288"},
                                      {"max_batch_size", "0"}}) {
        EXPECT_EXIT(std::_Exit(load({{name, value}}) != 0 ? 0 : 1),
                    ::testing::ExitedWithCode(0),
                    "nonzero, ordered block and batch size ranges");
    }
}

TEST(CapacityTest, KeepsNvshmemManyToOneUnsupported) {
    EXPECT_EXIT(std::_Exit(load({{"worker_type", "nvshmem"},
                                 {"initiator_seg_type", "VRAM"},
                                 {"target_seg_type", "VRAM"}}) != 0 ?
                               0 :
                               1),
                ::testing::ExitedWithCode(0),
                "Unsupported configuration for NVSHMEM worker");
}

} // namespace
