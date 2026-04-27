/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_REQUESTS_H
#define NIXLBENCH_BENCHMARK_REQUESTS_H

#include <string>

namespace nixlbench {

enum class ScenarioType {
    None,
    G3,
    G4,
};

enum class PluginType {
    None,
    Posix,
    Obj,
};

enum class CommandPath {
    None,
    Help,
    Raw,
    Scenario,
};

struct G3ScenarioRequest {
    std::string file_size;
    int parallel_threads = 1;
    std::string batch_size;
};

struct G4ScenarioRequest {
    std::string file_size;
    int num_kvs = 0;
    int parallel_threads = 1;
    std::string batch_size;
};

struct RawRequest {
    int num_iterations = 1000;
};

struct PosixPluginRequest {
    std::string storage_path;
    bool should_split_dir_per_thread = false;
    std::string mode = "aio";
    std::string api_type = "AIO";
    bool enable_direct = false;
};

struct ObjPluginRequest {
    std::string endpoint_url;
    std::string bucket_name;
};

struct ParsedBenchmarkCommand {
    CommandPath path = CommandPath::None;
    ScenarioType scenario = ScenarioType::None;
    PluginType plugin = PluginType::None;
    G3ScenarioRequest g3;
    G4ScenarioRequest g4;
    RawRequest raw;
    PosixPluginRequest posix;
    ObjPluginRequest obj;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_REQUESTS_H
