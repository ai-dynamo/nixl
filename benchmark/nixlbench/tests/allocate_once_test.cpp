/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/allocate_once.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <set>
#include <unistd.h>

namespace nixlbench {
namespace {

    TEST(AllocateOnceRegionsTest, AssignsThreadsRoundRobinToDisjointFilePartitions) {
        allocateOnceRequest request;
        request.fileSize = 10 * 4096;
        request.common.blockSize = 4096;
        request.common.batchSize = 2;
        request.common.threads = 4;
        request.files = {"/tmp/file-0", "/tmp/file-1"};

        std::string error;
        const auto regions = allocateOnceThreadRegions(request, error);
        ASSERT_TRUE(regions) << error;
        ASSERT_EQ(regions->size(), 4U);
        EXPECT_EQ((*regions)[0].fileIndex, 0U);
        EXPECT_EQ((*regions)[1].fileIndex, 1U);
        EXPECT_EQ((*regions)[2].fileIndex, 0U);
        EXPECT_EQ((*regions)[3].fileIndex, 1U);
        EXPECT_EQ((*regions)[0].firstSlot, 0U);
        EXPECT_EQ((*regions)[0].slotCount, 5U);
        EXPECT_EQ((*regions)[2].firstSlot, 5U);
        EXPECT_EQ((*regions)[2].slotCount, 5U);
    }

    TEST(AllocateOnceRegionsTest, DoesNotOverflowWhilePartitioningLargeFiles) {
        allocateOnceRequest request;
        request.fileSize = std::numeric_limits<size_t>::max();
        request.common.blockSize = 1;
        request.common.batchSize = 1;
        request.common.threads = 3;
        request.files = {"file-0"};

        std::string error;
        const auto regions = allocateOnceThreadRegions(request, error);
        ASSERT_TRUE(regions) << error;
        ASSERT_EQ(regions->size(), 3U);
        EXPECT_EQ((*regions)[0].firstSlot, 0U);
        EXPECT_EQ((*regions)[1].firstSlot, (*regions)[0].slotCount);
        EXPECT_EQ((*regions)[2].firstSlot, (*regions)[1].firstSlot + (*regions)[1].slotCount);
        EXPECT_EQ((*regions)[2].firstSlot + (*regions)[2].slotCount,
                  std::numeric_limits<size_t>::max());
    }

    TEST(AllocateOnceRegionsTest, SupportsUnevenThreadCountsAcrossFiles) {
        allocateOnceRequest request;
        request.fileSize = 8 * 4096;
        request.common.blockSize = 4096;
        request.common.batchSize = 2;
        request.common.threads = 3;
        request.files = {"file-0", "file-1"};

        std::string error;
        const auto regions = allocateOnceThreadRegions(request, error);
        ASSERT_TRUE(regions) << error;
        ASSERT_EQ(regions->size(), 3U);
        EXPECT_EQ((*regions)[0].fileIndex, 0U);
        EXPECT_EQ((*regions)[0].slotCount, 4U);
        EXPECT_EQ((*regions)[1].fileIndex, 1U);
        EXPECT_EQ((*regions)[1].slotCount, 8U);
        EXPECT_EQ((*regions)[2].fileIndex, 0U);
        EXPECT_EQ((*regions)[2].firstSlot, 4U);
        EXPECT_EQ((*regions)[2].slotCount, 4U);
    }

    TEST(OffsetSequenceTest, RandomBatchesAreSeededUniqueAndRemainInTheThreadRegion) {
        const threadFileRegion region{0, 10, 8};
        offsetSequence first(region, 4, offset_mode_t::RANDOM, 1234);
        offsetSequence second(region, 4, offset_mode_t::RANDOM, 1234);

        const auto first_batch = first.next();
        const auto second_batch = second.next();
        EXPECT_EQ(first_batch, second_batch);
        EXPECT_EQ(std::set<uint64_t>(first_batch.begin(), first_batch.end()).size(),
                  first_batch.size());
        EXPECT_TRUE(std::all_of(first_batch.begin(), first_batch.end(), [](uint64_t slot) {
            return slot >= 10 && slot < 18;
        }));
    }

    TEST(OffsetSequenceTest, SequentialBatchesWrapInsideTheThreadRegion) {
        offsetSequence offsets({0, 7, 5}, 3, offset_mode_t::SEQUENTIAL, 0);
        EXPECT_EQ(offsets.next(), (std::vector<uint64_t>{7, 8, 9}));
        EXPECT_EQ(offsets.next(), (std::vector<uint64_t>{10, 11, 7}));
    }

    TEST(OffsetSequenceTest, ZeroSeedResolvesToANonzeroRandomSeed) {
        EXPECT_EQ(resolveOffsetSeed(1234), 1234U);
        EXPECT_NE(resolveOffsetSeed(0), 0U);
    }

    TEST(AllocateOnceFileNamesTest, ManagedNamesAreScenarioOwnedAndDeterministic) {
        fileOptions file;
        file.path = "/tmp/scenario";
        file.numFiles = 2;

        std::string error;
        const auto names = allocateOnceFileNames(file, error);
        ASSERT_TRUE(names) << error;
        ASSERT_EQ(names->size(), 2U);
        EXPECT_EQ((*names)[0], "/tmp/scenario/nixlbench_allocate_once_0.dat");
        EXPECT_EQ((*names)[1], "/tmp/scenario/nixlbench_allocate_once_1.dat");
    }

    TEST(AllocateOnceFileNamesTest, PreservesExplicitFileNames) {
        fileOptions file;
        file.filenames = "/tmp/name one,/tmp/name-two";
        file.numFiles = 2;

        std::string error;
        const auto names = allocateOnceFileNames(file, error);
        ASSERT_TRUE(names) << error;
        ASSERT_EQ(names->size(), 2U);
        EXPECT_EQ((*names)[0], "/tmp/name one");
        EXPECT_EQ((*names)[1], "/tmp/name-two");
    }

    TEST(AllocateOnceFileNamesTest, ReportsAnUnavailableCurrentDirectory) {
        EXPECT_EXIT(
            {
                char directory_template[] = "/tmp/nixlbench-missing-cwd-XXXXXX";
                const char *directory = mkdtemp(directory_template);
                if (directory == nullptr || chdir(directory) != 0 || rmdir(directory) != 0) {
                    _exit(EXIT_FAILURE);
                }

                fileOptions file;
                std::string error;
                const auto names = allocateOnceFileNames(file, error);
                const bool rejected = !names &&
                    error.find("could not determine the current working directory") !=
                        std::string::npos;
                _exit(rejected ? EXIT_SUCCESS : EXIT_FAILURE);
            },
            ::testing::ExitedWithCode(EXIT_SUCCESS),
            "");
    }

} // namespace
} // namespace nixlbench
