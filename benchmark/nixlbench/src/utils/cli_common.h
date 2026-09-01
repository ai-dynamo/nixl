/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_CLI_COMMON_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_CLI_COMMON_H

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nixl_types.h>

namespace nixlbench {

/** @brief Plugin capabilities and parameters discovered from the NIXL API. */
struct pluginMetadata {
    std::string name;
    nixl_mem_list_t memoryTypes;
    nixl_b_params_t parameters;
};

/** @brief Shared file-resource values parsed from the verb-based CLI. */
struct fileOptions {
    std::string path;
    std::string filenames;
    int numFiles = 1;
    bool direct = false;
};

/** @brief Parse a positive byte size with an optional binary unit suffix. */
std::optional<size_t>
parseHumanSize(const std::string &value, std::string &error);

/** @brief Return whether plugin metadata advertises one memory type. */
bool
hasMemoryType(const pluginMetadata &metadata, nixl_mem_t memory_type);

/** @brief Validate shared file-resource option relationships. */
bool
validateFileOptions(const fileOptions &file, std::string &error);

/** @brief Split a validated comma-separated filename list. */
std::vector<std::string>
splitFileNames(const std::string &value);

/** @brief Format a byte count using a compact binary unit. */
std::string
formatSize(size_t bytes);

/** @brief Discover metadata for all installed NIXL plugins. */
std::optional<std::vector<pluginMetadata>>
discoverPluginMetadata(std::string &error);

/** @brief Return plugin parameter keys in deterministic sorted order. */
std::vector<std::string>
sortedParameterKeys(const nixl_b_params_t &parameters);

/** @brief Build CLI help text for metadata-advertised plugin parameters. */
std::string
pluginParameterDescription(const nixl_b_params_t &parameters);

} // namespace nixlbench

#endif
