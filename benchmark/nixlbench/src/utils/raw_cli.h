/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_RAW_CLI_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_RAW_CLI_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include <nixl_types.h>

namespace nixlbench {

struct PluginMetadata {
    std::string name;
    nixl_mem_list_t memory_types;
    nixl_b_params_t parameters;
};

struct RawOptions {
    std::string operation = "WRITE";
    size_t total_buffer_size = 8ULL * 1024 * 1024 * 1024;
    size_t start_block_size = 4ULL * 1024;
    size_t max_block_size = 64ULL * 1024 * 1024;
    size_t start_batch_size = 1;
    size_t max_batch_size = 1;
    int iterations = 1000;
    int warmup_iterations = 100;
    int threads = 1;
    int pipeline_depth = 1;
    bool check_consistency = false;
    bool dry_run = false;
};

struct FileOptions {
    std::string path;
    std::string filenames;
    int num_files = 1;
    bool direct = false;
};

struct RawPosixRequest {
    RawOptions raw;
    FileOptions file;
    bool has_file_options = false;
    nixl_b_params_t plugin_parameters;
};

struct RawCommandResult {
    int status = 0;
    bool execute = false;
    std::optional<nixl_b_params_t> plugin_parameters;
};

bool
isRawCommand(int argc, char *argv[]);

std::optional<size_t>
parseHumanSize(const std::string &value, std::string &error);

std::optional<PluginMetadata>
discoverPluginMetadata(const std::string &name, std::string &error);

int
parseRawPosixCommand(int argc,
                     char *argv[],
                     const PluginMetadata &metadata,
                     RawPosixRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err);

std::vector<std::string>
benchmarkFileArguments(const RawPosixRequest &request, const std::string &program_name);

void
printRawPosixPlan(const RawPosixRequest &request,
                  const PluginMetadata &metadata,
                  std::ostream &out);

RawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err);

} // namespace nixlbench

#endif
