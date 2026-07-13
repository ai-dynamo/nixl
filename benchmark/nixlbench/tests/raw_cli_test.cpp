/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/raw_cli.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>

namespace nixlbench {
namespace {

    PluginMetadata
    posixMetadata(bool include_queue_sizes = true) {
        PluginMetadata metadata{"POSIX", {DRAM_SEG, FILE_SEG}, {{"use_aio", "true"}}};
        if (include_queue_sizes) {
            metadata.parameters["ios_pool_size"] = "4096";
            metadata.parameters["kernel_queue_size"] = "128";
        }
        return metadata;
    }

    struct Arguments {
        explicit Arguments(std::initializer_list<const char *> values) {
            for (const auto *value : values) {
                storage.emplace_back(value);
            }
            for (auto &value : storage) {
                pointers.push_back(value.data());
            }
        }

        int
        argc() const {
            return static_cast<int>(pointers.size());
        }

        char **
        argv() {
            return pointers.data();
        }

        std::vector<std::string> storage;
        std::vector<char *> pointers;
    };

    TEST(RawCliDispatchTest, OnlyExplicitRawSelectsTheNewParser) {
        Arguments raw{"nixlbench", "raw", "posix"};
        EXPECT_TRUE(isRawCommand(raw.argc(), raw.argv()));

        Arguments legacy{"nixlbench", "--backend=POSIX"};
        EXPECT_FALSE(isRawCommand(legacy.argc(), legacy.argv()));
    }

    TEST(PluginMetadataDiscoveryTest, InstalledPosixAdvertisesUsableOptionsAndMemoryTypes) {
        std::string error;
        const auto metadata = discoverPluginMetadata("POSIX", error);
        ASSERT_TRUE(metadata) << error;
        EXPECT_NE(std::find(metadata->memory_types.begin(), metadata->memory_types.end(), DRAM_SEG),
                  metadata->memory_types.end());
        EXPECT_NE(std::find(metadata->memory_types.begin(), metadata->memory_types.end(), FILE_SEG),
                  metadata->memory_types.end());
        EXPECT_TRUE(metadata->parameters.contains("use_aio") ||
                    metadata->parameters.contains("use_uring") ||
                    metadata->parameters.contains("use_posix_aio"));
        EXPECT_EQ(metadata->parameters.at("ios_pool_size"), "65536");
        EXPECT_EQ(metadata->parameters.at("kernel_queue_size"), "256");
    }

    TEST(HumanSizeTest, ParsesBinarySizesAndCompatibilityAliases) {
        std::string error;
        EXPECT_EQ(parseHumanSize("4KiB", error), 4096U);
        EXPECT_EQ(parseHumanSize("2 MB", error), 2U * 1024 * 1024);
        EXPECT_EQ(parseHumanSize("1g", error), 1ULL * 1024 * 1024 * 1024);
    }

    TEST(HumanSizeTest, RejectsInvalidAndOverflowingValues) {
        std::string error;
        EXPECT_FALSE(parseHumanSize("0", error));
        EXPECT_FALSE(parseHumanSize("4XB", error));
        EXPECT_FALSE(parseHumanSize("999999999999999999999TiB", error));
    }

    TEST(RawPosixParserTest, DefaultsComeFromPluginMetadata) {
        Arguments args{"nixlbench", "raw", "posix", "--dry-run"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parseRawPosixCommand(
                      args.argc(), args.argv(), posixMetadata(), request, help, out, err),
                  0)
            << err.str();
        EXPECT_FALSE(help);
        EXPECT_EQ(request.api, "AIO");
        EXPECT_EQ(request.io_pool_size, 4096);
        EXPECT_EQ(request.kernel_queue_size, 128);
        EXPECT_TRUE(request.dry_run);
    }

    TEST(RawPosixParserTest, ParsesExplicitTypedValues) {
        PluginMetadata metadata = posixMetadata();
        metadata.parameters["use_uring"] = "false";
        Arguments args{"nixlbench", "raw",
                       "posix",     "--operation",
                       "read",      "--total-buffer-size",
                       "8MiB",      "--start-block-size",
                       "4KiB",      "--max-block-size",
                       "1MiB",      "--iterations",
                       "32",        "--warmup-iterations",
                       "0",         "--threads",
                       "2",         "--num-files",
                       "2",         "--api",
                       "uring",     "--io-pool-size",
                       "2048",      "--check-consistency"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parseRawPosixCommand(args.argc(), args.argv(), metadata, request, help, out, err),
                  0)
            << err.str();
        EXPECT_EQ(request.operation, "READ");
        EXPECT_EQ(request.total_buffer_size, 8U * 1024 * 1024);
        EXPECT_EQ(request.max_block_size, 1U * 1024 * 1024);
        EXPECT_EQ(request.api, "URING");
        EXPECT_EQ(request.io_pool_size, 2048);
        EXPECT_TRUE(request.check_consistency);
    }

    TEST(RawPosixParserTest, RejectsUnknownAndUnadvertisedOptions) {
        Arguments unknown{"nixlbench", "raw", "posix", "--gds-batch-limit", "8"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(parseRawPosixCommand(
                      unknown.argc(), unknown.argv(), posixMetadata(), request, help, out, err),
                  0);

        Arguments unadvertised{"nixlbench", "raw", "posix", "--io-pool-size", "8"};
        out.str("");
        err.str("");
        EXPECT_NE(parseRawPosixCommand(unadvertised.argc(),
                                       unadvertised.argv(),
                                       posixMetadata(false),
                                       request,
                                       help,
                                       out,
                                       err),
                  0);
    }

    TEST(RawPosixParserTest, RejectsMalformedPluginMetadataDefaults) {
        auto metadata = posixMetadata();
        metadata.parameters["ios_pool_size"] = "not-a-number";
        Arguments args{"nixlbench", "raw", "posix", "--dry-run"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        EXPECT_NE(parseRawPosixCommand(args.argc(), args.argv(), metadata, request, help, out, err),
                  0);
        EXPECT_NE(err.str().find("ios_pool_size"), std::string::npos);
    }

    TEST(RawPosixParserTest, RejectsQueueSizesOutsideBackendLimits) {
        Arguments small_pool{"nixlbench", "raw", "posix", "--io-pool-size", "1"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(
            parseRawPosixCommand(
                small_pool.argc(), small_pool.argv(), posixMetadata(), request, help, out, err),
            0);

        Arguments large_kernel{"nixlbench", "raw", "posix", "--kernel-queue-size", "2048"};
        out.str("");
        err.str("");
        EXPECT_NE(
            parseRawPosixCommand(
                large_kernel.argc(), large_kernel.argv(), posixMetadata(), request, help, out, err),
            0);
    }

    TEST(RawPosixParserTest, RejectsInvalidFileAndSweepConfiguration) {
        Arguments args{"nixlbench",
                       "raw",
                       "posix",
                       "--threads",
                       "1",
                       "--num-files",
                       "2",
                       "--start-block-size",
                       "8KiB",
                       "--max-block-size",
                       "4KiB"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(parseRawPosixCommand(
                      args.argc(), args.argv(), posixMetadata(), request, help, out, err),
                  0);
        EXPECT_NE(err.str().find("--num-files"), std::string::npos);
    }

    TEST(RawPosixParserTest, ScopedHelpContainsOnlyAdvertisedPosixOptions) {
        Arguments args{"nixlbench", "raw", "posix", "--help"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(parseRawPosixCommand(
                      args.argc(), args.argv(), posixMetadata(false), request, help, out, err),
                  0);
        EXPECT_TRUE(help);
        EXPECT_NE(out.str().find("--api"), std::string::npos);
        EXPECT_EQ(out.str().find("--io-pool-size"), std::string::npos);
        EXPECT_EQ(out.str().find("--backend"), std::string::npos);
    }

    TEST(RawRequestConversionTest, ProducesEquivalentLegacyConfigurationArguments) {
        RawPosixRequest request;
        request.operation = "READ";
        request.path = "/tmp/nixlbench";
        request.filenames = "";
        request.num_files = 2;
        request.total_buffer_size = 65536;
        request.start_block_size = 4096;
        request.max_block_size = 32768;
        request.start_batch_size = 2;
        request.max_batch_size = 8;
        request.iterations = 16;
        request.warmup_iterations = 0;
        request.threads = 4;
        request.pipeline_depth = 3;
        request.check_consistency = true;
        request.direct = true;
        request.api = "AIO";
        request.io_pool_size = 4096;
        request.kernel_queue_size = 128;

        const auto args = legacyArguments(request, "nixlbench");
        const std::vector<std::string> expected = {
            "nixlbench",
            "--worker_type=nixl",
            "--backend=POSIX",
            "--initiator_seg_type=DRAM",
            "--target_seg_type=DRAM",
            "--op_type=READ",
            "--check_consistency=true",
            "--total_buffer_size=65536",
            "--start_block_size=4096",
            "--max_block_size=32768",
            "--start_batch_size=2",
            "--max_batch_size=8",
            "--num_iter=16",
            "--warmup_iter=0",
            "--num_threads=4",
            "--filepath=/tmp/nixlbench",
            "--filenames=",
            "--num_files=2",
            "--storage_enable_direct=true",
            "--pipeline_depth=3",
            "--posix_api_type=AIO",
            "--posix_ios_pool_size=4096",
            "--posix_kernel_queue_size=128",
        };
        EXPECT_EQ(args.size(), expected.size());
        for (const auto &argument : expected) {
            EXPECT_NE(std::find(args.begin(), args.end(), argument), args.end()) << argument;
        }
    }

} // namespace
} // namespace nixlbench
