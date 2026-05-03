/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/plugin_registry.h"

#include "utils/cli/metadata_plugin_command.h"

#include <algorithm>

namespace nixlbench {
// namespace {

//     struct pluginDefinition {
//         plugin_type_t type;
//         std::string commandName;
//         std::string backendName;
//         std::string description;
//     };

//     const std::vector<pluginDefinition> &
//     pluginDefinitions() {
//         static const std::vector<pluginDefinition> definitions{
//             {plugin_type_t::POSIX, "posix", "POSIX", "Use the POSIX storage backend"},
//             {plugin_type_t::OBJ, "obj", "OBJ", "Use the object storage backend"},
//             {plugin_type_t::GDS, "gds", "GDS", "Use the GDS storage backend"},
//             {plugin_type_t::GDS_MT, "gds_mt", "GDS_MT", "Use the GDS-MT storage backend"},
//             {plugin_type_t::GPUNETIO, "gpunetio", "GPUNETIO", "Use the GPUNetIO backend"},
//             {plugin_type_t::AZURE_BLOB, "azure_blob", "AZURE_BLOB", "Use the Azure Blob
//             backend"}, {plugin_type_t::HF3FS, "hf3fs", "HF3FS", "Use the HF3FS storage backend"},
//             {plugin_type_t::GUSLI, "gusli", "GUSLI", "Use the GUSLI storage backend"},
//         };
//         return definitions;
//     }

// } // namespace

southboundPluginRegistry &
southboundPluginRegistry::instance() {
    static southboundPluginRegistry registry;
    return registry;
}

bool
southboundPluginRegistry::registerPlugin(Factory factory) {
    factories_.push_back(std::move(factory));
    return true;
}

std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>>
southboundPluginRegistry::createAll() const {
    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> plugins;
    plugins.reserve(factories_.size());
    for (const auto &factory : factories_) {
        plugins.push_back(factory());
    }
    return plugins;
}

std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>>
southboundPluginRegistry::createForAvailableMetadataPlugins() const {
    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> plugins;
    nixlAgent agent("nixlbench-cli", nixlAgentConfig{});
    std::vector<nixl_backend_t> available_plugins;
    agent.getAvailPlugins(available_plugins);

    for (const auto &plugin_name : available_plugins) {
        nixl_backend_option_list_t option_specs;
        nixlBackendPluginCapabilities capabilities;
        if (agent.getPluginOptionSpecs(plugin_name, option_specs) != NIXL_SUCCESS) {
            continue;
        }
        if (agent.getPluginCapabilities(plugin_name, capabilities) != NIXL_SUCCESS) {
            capabilities = {};
        }
        plugins.push_back(
            std::make_unique<metadataPluginCommand>(plugin_name, capabilities, option_specs));
    }


    return plugins;
}

} // namespace nixlbench
