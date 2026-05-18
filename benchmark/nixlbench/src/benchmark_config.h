/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_CONFIG_H
#define NIXLBENCH_BENCHMARK_CONFIG_H

#include "nixl_types.h"
#include "utils/cli/benchmark_requests.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <algorithm>

class xferBenchConfig;

namespace nixlbench {

struct benchmarkCommonConfig {
    std::string benchmark_group = "default";
    bool check_consistency = false;
    bool recreate_xfer = false;
    int num_iter = 1000;
    int large_blk_iter_ftr = 16;
    int warmup_iter = 100;
};

struct runtimeConfig {
    std::string type = "ETCD";
    std::string etcd_endpoints;
};

struct transferConfig {
    std::string initiator_seg_type = "DRAM";
    std::string target_seg_type = "DRAM";
    std::string scheme = "pairwise";
    std::string mode = "SG";
    std::string op_type = "WRITE";
    std::size_t total_buffer_size = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t start_block_size = 4ULL * 1024ULL;
    std::size_t max_block_size = 64ULL * 1024ULL * 1024ULL;
    std::size_t start_batch_size = 1;
    std::size_t max_batch_size = 1;
    int num_threads = 1;
};

struct workerConfig {
    std::string type = "nixl";
    int num_initiator_dev = 1;
    int num_target_dev = 1;
    bool enable_progress_thread = false;
    std::size_t progress_threads = 0;
    std::string device_list = "all";
    bool enable_vmm = false;
};

struct backendConfig {
    std::string name = "";
    nixlBackendPluginCapabilities capabilities{};
    metadata_plugin_option_map_t options;
    nixl_mem_list_t memory_types;
};

struct storageConfig {
    std::string filepath;
    std::string filenames;
    int num_files = 1;
    bool enable_direct = false;
};

struct benchmarkConfig {
    benchmarkCommonConfig common;
    runtimeConfig runtime;
    transferConfig transfer;
    workerConfig worker;
    backendConfig backend;
    storageConfig storage;
};

inline bool
isStorageBackend(const backendConfig &backend) {
    // check whether the backend supports FILE_SEG, OBJ_SEG, or BLK_SEG, any of those are considered storage backends

    return std::find(backend.memory_types.begin(), backend.memory_types.end(), FILE_SEG) != backend.memory_types.end() ||
           std::find(backend.memory_types.begin(), backend.memory_types.end(), OBJ_SEG) != backend.memory_types.end() ||
           std::find(backend.memory_types.begin(), backend.memory_types.end(), BLK_SEG) != backend.memory_types.end();
}

inline bool
isObjStorageBackend(const backendConfig &backend) {
    return std::find(backend.memory_types.begin(), backend.memory_types.end(), OBJ_SEG) != backend.memory_types.end();
}

benchmarkConfig
makeBenchmarkConfigFromLegacy(const xferBenchConfig &legacy_config);

benchmarkConfig
makeBenchmarkConfigFromRawRequest(const rawRequest &request);

// Temporary Phase 4 bridge while runner and workers still consume xferBenchConfig.
xferBenchConfig
makeLegacyConfigFromBenchmarkConfig(const benchmarkConfig &config);

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_CONFIG_H
