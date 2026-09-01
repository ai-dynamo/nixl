/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli_common.h"

#include <CLI/CLI.hpp>
#include <nixl.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>

namespace nixlbench {
namespace {

    bool
    listAvailablePlugins(nixlAgent &agent,
                         std::vector<nixl_backend_t> &plugins,
                         std::string &error) {
        const auto status = agent.getAvailPlugins(plugins);
        if (status == NIXL_SUCCESS) {
            return true;
        }
        error = "failed to discover NIXL plugins: " + nixlEnumStrings::statusStr(status);
        return false;
    }

    std::optional<pluginMetadata>
    queryPluginMetadata(nixlAgent &agent, const std::string &name, std::string &error) {
        pluginMetadata metadata;
        metadata.name = name;
        const auto status = agent.getPluginParams(name, metadata.memoryTypes, metadata.parameters);
        if (status != NIXL_SUCCESS) {
            error = "failed to query " + name +
                " plugin metadata: " + nixlEnumStrings::statusStr(status);
            return std::nullopt;
        }
        return metadata;
    }

} // namespace

CLI::Validator
binarySizeTransform() {
    const std::map<std::string, uint64_t> units = {
        {"B", 1},
        {"K", 1024ULL},
        {"KB", 1024ULL},
        {"M", 1024ULL * 1024},
        {"MB", 1024ULL * 1024},
        {"G", 1024ULL * 1024 * 1024},
        {"GB", 1024ULL * 1024 * 1024},
        {"T", 1024ULL * 1024 * 1024 * 1024},
        {"TB", 1024ULL * 1024 * 1024 * 1024},
    };
    return CLI::AsNumberWithUnit(units);
}

void
addFileOptions(CLI::App &command, fileOptions &options) {
    command.add_option("--path", options.path, "Directory for automatically named files")
        ->group("FILE_SEG resource options");
    command.add_option("--filenames", options.filenames, "Comma-separated explicit file names")
        ->group("FILE_SEG resource options");
    command.add_option("--num-files", options.numFiles, "Number of backing files")
        ->group("FILE_SEG resource options");
    command.add_flag("--direct", options.direct, "Use direct file opening")
        ->group("FILE_SEG resource options");
}

void
addPluginOptions(CLI::App &command,
                 const nixl_b_params_t &parameters,
                 std::vector<std::pair<std::string, std::string>> &overrides) {
    if (parameters.empty()) {
        return;
    }
    command.add_option("--plugin-param", overrides, pluginParameterDescription(parameters))
        ->check(CLI::IsMember(sortedParameterKeys(parameters)).description("").application_index(0))
        ->type_name("KEY VALUE")
        ->group("Plugin initialization parameters");
}

bool
hasMemoryType(const pluginMetadata &metadata, nixl_mem_t memory_type) {
    return std::find(metadata.memoryTypes.begin(), metadata.memoryTypes.end(), memory_type) !=
        metadata.memoryTypes.end();
}

bool
validateFileOptions(const fileOptions &file, std::string &error) {
    if (!file.path.empty() && !file.filenames.empty()) {
        error = "--path and --filenames are mutually exclusive";
        return false;
    }
    if (file.numFiles < 1) {
        error = "--num-files must be at least 1";
        return false;
    }
    if (!file.filenames.empty() &&
        (file.filenames.front() == ',' || file.filenames.back() == ',' ||
         file.filenames.find(",,") != std::string::npos)) {
        error = "--filenames must not contain empty entries";
        return false;
    }
    if (!file.filenames.empty() &&
        splitFileNames(file.filenames).size() != static_cast<size_t>(file.numFiles)) {
        error = "--filenames must contain exactly --num-files entries";
        return false;
    }
    return true;
}

std::vector<std::string>
splitFileNames(const std::string &value) {
    std::vector<std::string> names;
    std::stringstream input(value);
    std::string name;
    while (std::getline(input, name, ',')) {
        names.push_back(name);
    }
    return names;
}

std::string
formatSize(size_t bytes) {
    static constexpr const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream output;
    output << std::fixed << std::setprecision(value == static_cast<size_t>(value) ? 0 : 2) << value
           << ' ' << units[unit] << " (" << bytes << " bytes)";
    return output.str();
}

std::optional<pluginMetadata>
discoverPluginMetadata(const std::string &name, std::string &error) {
    nixlAgent agent("nixlbench-cli", nixlAgentConfig{});
    std::vector<nixl_backend_t> plugins;
    if (!listAvailablePlugins(agent, plugins, error)) {
        return std::nullopt;
    }
    if (std::find(plugins.begin(), plugins.end(), name) == plugins.end()) {
        error = name + " plugin is not installed or not visible in the NIXL plugin path";
        return std::nullopt;
    }
    return queryPluginMetadata(agent, name, error);
}

std::optional<std::vector<pluginMetadata>>
discoverPluginMetadata(std::string &error) {
    nixlAgent agent("nixlbench-cli", nixlAgentConfig{});
    std::vector<nixl_backend_t> plugins;
    if (!listAvailablePlugins(agent, plugins, error)) {
        return std::nullopt;
    }

    std::sort(plugins.begin(), plugins.end());
    std::vector<pluginMetadata> metadata;
    metadata.reserve(plugins.size());
    for (const auto &plugin : plugins) {
        std::string plugin_error;
        auto entry = queryPluginMetadata(agent, plugin, plugin_error);
        if (!entry) {
            error = std::move(plugin_error);
            return std::nullopt;
        }
        metadata.push_back(std::move(*entry));
    }
    return metadata;
}

std::vector<std::string>
sortedParameterKeys(const nixl_b_params_t &parameters) {
    std::vector<std::string> keys;
    keys.reserve(parameters.size());
    for (const auto &[key, value] : parameters) {
        (void)value;
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::string
pluginParameterDescription(const nixl_b_params_t &parameters) {
    std::ostringstream description;
    description << "Override an advertised plugin parameter"
                << "\nAdvertised parameters and defaults:";
    for (const auto &key : sortedParameterKeys(parameters)) {
        description << "\n  " << key << ": " << parameters.at(key);
    }
    return description.str();
}

} // namespace nixlbench
