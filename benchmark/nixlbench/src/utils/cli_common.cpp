/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli_common.h"

#include <nixl.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>

namespace nixlbench {
namespace {

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

std::optional<size_t>
parseHumanSize(const std::string &input, std::string &error) {
    std::string value;
    value.reserve(input.size());
    for (unsigned char ch : input) {
        if (!std::isspace(ch)) {
            value.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    if (value.empty()) {
        error = "size cannot be empty";
        return std::nullopt;
    }

    size_t digit_count = 0;
    while (digit_count < value.size() &&
           std::isdigit(static_cast<unsigned char>(value[digit_count]))) {
        ++digit_count;
    }
    if (digit_count == 0) {
        error = "size must begin with a positive integer";
        return std::nullopt;
    }

    uint64_t number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + digit_count, number);
    if (parsed.ec != std::errc() || number == 0) {
        error = "size must be a positive integer";
        return std::nullopt;
    }

    const std::string suffix = value.substr(digit_count);
    uint64_t multiplier = 0;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1;
    } else if (suffix == "K" || suffix == "KB") {
        multiplier = 1024ULL;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1024ULL * 1024;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1024ULL * 1024 * 1024;
    } else if (suffix == "T" || suffix == "TB") {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
    } else {
        error = "unsupported size suffix '" + suffix +
            "' (use B, KB, MB, GB, or TB; units are binary multiples)";
        return std::nullopt;
    }
    if (number > std::numeric_limits<size_t>::max() / multiplier) {
        error = "size is too large for this platform";
        return std::nullopt;
    }
    return static_cast<size_t>(number * multiplier);
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

std::optional<std::vector<pluginMetadata>>
discoverPluginMetadata(std::string &error) {
    nixlAgent agent("nixlbench-cli", nixlAgentConfig{});
    std::vector<nixl_backend_t> plugins;
    const auto list_status = agent.getAvailPlugins(plugins);
    if (list_status != NIXL_SUCCESS) {
        error = "failed to discover NIXL plugins: " + nixlEnumStrings::statusStr(list_status);
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
