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
    posixMetadata() {
        return {
            "POSIX",
            {DRAM_SEG, FILE_SEG},
            {{"future_parameter", "default"}, {"ios_pool_size", "4096"}, {"use_uring", "false"}}};
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

    int
    parse(Arguments &arguments,
          const PluginMetadata &metadata,
          RawPosixRequest &request,
          std::ostringstream &out,
          std::ostringstream &err,
          bool &help) {
        return parseRawPosixCommand(
            arguments.argc(), arguments.argv(), metadata, request, help, out, err);
    }

    TEST(RawCliDispatchTest, OnlyExplicitRawSelectsTheNewParser) {
        Arguments raw{"nixlbench", "raw", "posix"};
        EXPECT_TRUE(isRawCommand(raw.argc(), raw.argv()));

        Arguments legacy{"nixlbench", "--backend=POSIX"};
        EXPECT_FALSE(isRawCommand(legacy.argc(), legacy.argv()));
    }

    TEST(PluginMetadataDiscoveryTest, InstalledPosixPublishesDefaults) {
        std::string error;
        const auto metadata = discoverPluginMetadata("POSIX", error);
        ASSERT_TRUE(metadata) << error;
        EXPECT_NE(std::find(metadata->memory_types.begin(), metadata->memory_types.end(), DRAM_SEG),
                  metadata->memory_types.end());
        EXPECT_NE(std::find(metadata->memory_types.begin(), metadata->memory_types.end(), FILE_SEG),
                  metadata->memory_types.end());
        for (const char *key : {"ios_pool_size", "kernel_queue_size"}) {
            const auto parameter = metadata->parameters.find(key);
            ASSERT_NE(parameter, metadata->parameters.end()) << key;
            EXPECT_FALSE(parameter->second.empty()) << key;
        }

        bool found_selection_toggle = false;
        for (const char *key : {"use_aio", "use_uring", "use_posix_aio"}) {
            const auto parameter = metadata->parameters.find(key);
            if (parameter != metadata->parameters.end()) {
                found_selection_toggle = true;
                EXPECT_EQ(parameter->second, "false") << key;
            }
        }
        EXPECT_TRUE(found_selection_toggle);
    }

    TEST(HumanSizeTest, ParsesAndRejectsHumanReadableSizes) {
        std::string error;
        EXPECT_EQ(parseHumanSize("4KiB", error), 4096U);
        EXPECT_EQ(parseHumanSize("2 MB", error), 2U * 1024 * 1024);
        EXPECT_EQ(parseHumanSize("1g", error), 1ULL * 1024 * 1024 * 1024);
        EXPECT_FALSE(parseHumanSize("0", error));
        EXPECT_FALSE(parseHumanSize("4XB", error));
        EXPECT_FALSE(parseHumanSize("999999999999999999999TiB", error));
    }

    TEST(RawPosixParserTest, PreservesAdvertisedPluginDefaultsOverridesAndValues) {
        Arguments defaults{"nixlbench", "raw", "posix", "--dry-run"};
        auto metadata = posixMetadata();
        metadata.parameters.emplace("path,alias", "default-value");
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parse(defaults, metadata, request, out, err, help), 0) << err.str();
        EXPECT_FALSE(help);
        EXPECT_EQ(request.plugin_parameters, metadata.parameters);
        EXPECT_TRUE(request.raw.dry_run);

        Arguments overrides{"nixlbench",
                            "raw",
                            "posix",
                            "--plugin-param",
                            "future_parameter",
                            "override",
                            "--plugin-param",
                            "ios_pool_size",
                            "not-a-number",
                            "--plugin-param",
                            "path,alias",
                            "Exact Value"};
        request = {};
        out.str("");
        err.str("");
        ASSERT_EQ(parse(overrides, metadata, request, out, err, help), 0) << err.str();
        EXPECT_EQ(request.plugin_parameters.at("future_parameter"), "override");
        EXPECT_EQ(request.plugin_parameters.at("ios_pool_size"), "not-a-number");
        EXPECT_EQ(request.plugin_parameters.at("path,alias"), "Exact Value");
    }

    TEST(RawPosixParserTest, ParsesRawAndFileOptionsWithoutMixingPluginParameters) {
        Arguments arguments{"nixlbench",
                            "raw",
                            "--operation",
                            "read",
                            "--total-buffer-size",
                            "8MiB",
                            "--iterations",
                            "32",
                            "--threads",
                            "2",
                            "posix",
                            "--path",
                            "/tmp/nixlbench",
                            "--num-files",
                            "2",
                            "--direct",
                            "--plugin-param",
                            "future_parameter",
                            "Exact-Value"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parse(arguments, posixMetadata(), request, out, err, help), 0) << err.str();
        EXPECT_EQ(request.raw.operation, "READ");
        EXPECT_EQ(request.raw.total_buffer_size, 8U * 1024 * 1024);
        EXPECT_EQ(request.raw.iterations, 32);
        EXPECT_TRUE(request.has_file_options);
        EXPECT_EQ(request.file.path, "/tmp/nixlbench");
        EXPECT_EQ(request.file.num_files, 2);
        EXPECT_TRUE(request.file.direct);
        EXPECT_EQ(request.plugin_parameters.at("future_parameter"), "Exact-Value");
    }

    TEST(RawPosixParserTest, RejectsUnknownAndUnadvertisedOptions) {
        PluginMetadata metadata = posixMetadata();
        metadata.parameters.erase("future_parameter");
        Arguments unadvertised{
            "nixlbench", "raw", "posix", "--plugin-param", "future_parameter", "override"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(parse(unadvertised, metadata, request, out, err, help), 0);

        Arguments unknown{"nixlbench", "raw", "posix", "--gds-batch-limit", "8"};
        out.str("");
        err.str("");
        EXPECT_NE(parse(unknown, metadata, request, out, err, help), 0);
    }

    TEST(RawPosixParserTest, ValidatesRawBenchmarkOptions) {
        Arguments threads{"nixlbench", "raw", "posix", "--threads", "0"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(parse(threads, posixMetadata(), request, out, err, help), 0);
        EXPECT_NE(err.str().find("threads"), std::string::npos);

        Arguments sweep{
            "nixlbench", "raw", "posix", "--start-block-size", "8KiB", "--max-block-size", "4KiB"};
        request = {};
        out.str("");
        err.str("");
        EXPECT_NE(parse(sweep, posixMetadata(), request, out, err, help), 0);
        EXPECT_NE(err.str().find("block sizes"), std::string::npos);
    }

    TEST(RawPosixParserTest, ValidatesFileResourceOptions) {
        Arguments files{"nixlbench", "raw", "posix", "--threads", "1", "--num-files", "2"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_NE(parse(files, posixMetadata(), request, out, err, help), 0);
        EXPECT_NE(err.str().find("--num-files"), std::string::npos);

        Arguments names{"nixlbench",
                        "raw",
                        "posix",
                        "--threads",
                        "2",
                        "--num-files",
                        "2",
                        "--filenames",
                        "one"};
        request = {};
        out.str("");
        err.str("");
        EXPECT_NE(parse(names, posixMetadata(), request, out, err, help), 0);
        EXPECT_NE(err.str().find("exactly --num-files"), std::string::npos);

        const auto expect_empty_name_rejected = [](const char *filenames, const char *num_files) {
            Arguments empty_name{"nixlbench",
                                 "raw",
                                 "posix",
                                 "--threads",
                                 num_files,
                                 "--num-files",
                                 num_files,
                                 "--filenames",
                                 filenames};
            RawPosixRequest invalid_request;
            bool invalid_help = false;
            std::ostringstream invalid_out;
            std::ostringstream invalid_err;
            EXPECT_NE(parse(empty_name,
                            posixMetadata(),
                            invalid_request,
                            invalid_out,
                            invalid_err,
                            invalid_help),
                      0);
            EXPECT_NE(invalid_err.str().find("empty entries"), std::string::npos);
        };
        expect_empty_name_rejected(",one", "2");
        expect_empty_name_rejected("one,", "2");
        expect_empty_name_rejected("one,,three", "3");
    }

    TEST(RawPosixParserTest, HelpShowsSortedAdvertisedPluginParametersAndDefaults) {
        auto metadata = posixMetadata();
        metadata.parameters.emplace("zeta_parameter", "z");
        metadata.parameters.emplace("alpha_parameter", "a");
        metadata.parameters.emplace("middle_parameter", "m");
        metadata.parameters.emplace("path,alias", "default-value");
        Arguments posix_help{"nixlbench", "raw", "posix", "--help"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;
        EXPECT_EQ(parse(posix_help, metadata, request, out, err, help), 0);
        EXPECT_TRUE(help);
        EXPECT_NE(out.str().find("FILE_SEG resource options"), std::string::npos);
        EXPECT_NE(out.str().find("Plugin initialization parameters"), std::string::npos);
        EXPECT_NE(out.str().find("--plugin-param"), std::string::npos);
        EXPECT_NE(out.str().find("Advertised parameters and defaults:"), std::string::npos);
        EXPECT_NE(out.str().find("--path"), std::string::npos);
        EXPECT_NE(out.str().find("future_parameter: default"), std::string::npos);
        EXPECT_NE(out.str().find("path,alias: default-value"), std::string::npos);
        EXPECT_EQ(out.str().find("KEY VALUE:{"), std::string::npos);
        EXPECT_EQ(out.str().find("--future_parameter"), std::string::npos);
        EXPECT_EQ(out.str().find("--operation"), std::string::npos);
        EXPECT_EQ(out.str().find("--api"), std::string::npos);
        EXPECT_EQ(out.str().find("--io-pool-size"), std::string::npos);
        EXPECT_EQ(out.str().find("--backend"), std::string::npos);

        const auto alpha = out.str().find("alpha_parameter: a");
        const auto middle = out.str().find("middle_parameter: m");
        const auto zeta = out.str().find("zeta_parameter: z");
        ASSERT_NE(alpha, std::string::npos);
        ASSERT_NE(middle, std::string::npos);
        ASSERT_NE(zeta, std::string::npos);
        EXPECT_LT(alpha, middle);
        EXPECT_LT(middle, zeta);

        Arguments raw_help{"nixlbench", "raw", "--help"};
        request = {};
        help = false;
        out.str("");
        err.str("");
        EXPECT_EQ(parse(raw_help, metadata, request, out, err, help), 0);
        EXPECT_TRUE(help);
        EXPECT_NE(out.str().find("Raw benchmark options"), std::string::npos);
        EXPECT_NE(out.str().find("--operation"), std::string::npos);
        EXPECT_NE(out.str().find("8 GiB (8589934592 bytes)"), std::string::npos);
        EXPECT_NE(out.str().find("4 KiB (4096 bytes)"), std::string::npos);
        EXPECT_NE(out.str().find("64 MiB (67108864 bytes)"), std::string::npos);
        EXPECT_EQ(out.str().find("--path"), std::string::npos);
        EXPECT_EQ(out.str().find("--plugin-param"), std::string::npos);
    }

    TEST(RawPosixParserTest, GatesAllFileSurfacesOnAdvertisedFileSeg) {
        auto metadata = posixMetadata();
        metadata.name = "OTHER";
        metadata.memory_types = {DRAM_SEG};
        Arguments help_arguments{"nixlbench", "raw", "posix", "--help"};
        RawPosixRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parse(help_arguments, metadata, request, out, err, help), 0) << err.str();
        ASSERT_TRUE(help);
        EXPECT_FALSE(request.has_file_options);
        EXPECT_EQ(out.str().find("FILE_SEG resource options"), std::string::npos);
        for (const char *option : {"--path", "--filenames", "--num-files", "--direct"}) {
            EXPECT_EQ(out.str().find(option), std::string::npos) << option;
        }

        std::ostringstream plan;
        printRawPosixPlan(
            request, metadata, request.raw.iterations, request.raw.warmup_iterations, plan);
        EXPECT_EQ(plan.str().find("file-resource options:"), std::string::npos);
        const auto arguments = benchmarkFileArguments(request, "nixlbench");
        for (const char *prefix :
             {"--filepath=", "--filenames=", "--num_files=", "--storage_enable_direct="}) {
            EXPECT_TRUE(std::none_of(
                arguments.begin(),
                arguments.end(),
                [prefix](const auto &argument) { return argument.rfind(prefix, 0) == 0; }))
                << prefix;
        }
    }

    TEST(RawPlanTest, PrintsSeparatedSectionsAndSortedExactPluginParameters) {
        RawPosixRequest request;
        request.raw.operation = "READ";
        request.has_file_options = true;
        request.file.path = "/tmp/nixlbench";
        request.plugin_parameters = {{"zeta_parameter", "Value-Z"}, {"alpha_parameter", "Value-A"}};
        const PluginMetadata metadata{"POSIX", {FILE_SEG, DRAM_SEG}, request.plugin_parameters};
        std::ostringstream out;

        printRawPosixPlan(
            request, metadata, request.raw.iterations, request.raw.warmup_iterations, out);

        const auto benchmark = out.str().find("benchmark options:");
        const auto file = out.str().find("file-resource options:");
        const auto plugin = out.str().find("plugin parameters:");
        const auto alpha = out.str().find("alpha_parameter: Value-A");
        const auto zeta = out.str().find("zeta_parameter: Value-Z");
        ASSERT_NE(benchmark, std::string::npos);
        ASSERT_NE(file, std::string::npos);
        ASSERT_NE(plugin, std::string::npos);
        ASSERT_NE(alpha, std::string::npos);
        ASSERT_NE(zeta, std::string::npos);
        EXPECT_NE(out.str().find("memory types: DRAM_SEG, FILE_SEG"), std::string::npos);
        EXPECT_LT(benchmark, file);
        EXPECT_LT(file, plugin);
        EXPECT_LT(alpha, zeta);
    }

    TEST(RawRequestConversionTest, BridgeContainsOnlyBenchmarkAndFileResourceSettings) {
        RawPosixRequest request;
        request.raw.operation = "READ";
        request.raw.total_buffer_size = 65536;
        request.raw.start_block_size = 4096;
        request.raw.max_block_size = 32768;
        request.raw.start_batch_size = 2;
        request.raw.max_batch_size = 8;
        request.raw.iterations = 16;
        request.raw.warmup_iterations = 0;
        request.raw.threads = 4;
        request.raw.pipeline_depth = 3;
        request.raw.check_consistency = true;
        request.has_file_options = true;
        request.file.path = "/tmp/nixlbench";
        request.file.num_files = 2;
        request.file.direct = true;
        request.plugin_parameters = {{"future_parameter", "override"}};

        const auto arguments = benchmarkFileArguments(request, "nixlbench");
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
            "--pipeline_depth=3",
            "--filepath=/tmp/nixlbench",
            "--filenames=",
            "--num_files=2",
            "--storage_enable_direct=true",
        };
        EXPECT_EQ(arguments, expected);
        for (const auto &argument : arguments) {
            EXPECT_EQ(argument.find("--posix_api_type="), std::string::npos);
            EXPECT_EQ(argument.find("--posix_ios_pool_size="), std::string::npos);
            EXPECT_EQ(argument.find("--posix_kernel_queue_size="), std::string::npos);
            EXPECT_EQ(argument.find("future_parameter"), std::string::npos);
        }
    }

} // namespace
} // namespace nixlbench
