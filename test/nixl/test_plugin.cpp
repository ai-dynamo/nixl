/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <set>
#include <string>
#include "nixl.h"
#include "plugin_manager.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [plugin_directory]" << std::endl;
    std::cout << "Environment variables:" << std::endl;
    std::cout << "  NIXL_PLUGIN_DIR   - Single directory containing plugins" << std::endl;
}

void printParams(const nixl_b_params_t& params) {
    if (params.empty()) {
        std::cout << "Parameters: (empty)" << std::endl;
        return;
    }

    std::cout << "Parameters:" << std::endl;
    for (const auto& pair : params) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }
}

int verify_plugin(std::string name, nixlPluginManager& plugin_manager)
{
    // Discover available plugins
    std::cout << "\nLoading " << name << " plugin..." << std::endl;

    // Load the plugin
    auto plugin_ = plugin_manager.loadPlugin(name);
    if (!plugin_) {
        std::cerr << "Failed to load " << name << " plugin" << std::endl;
        return -1;
    }

    // Display plugin information
    std::cout << "Plugin name: " << plugin_->getName() << std::endl;
    std::cout << "Plugin version: " << plugin_->getVersion() << std::endl;

    // Get backend options
    printParams(plugin_->getBackendOptions());

    return 0;
}

int main(int argc, char** argv) {
    char *plugindir = NULL;
    std::set<nixl_backend_t> staticPlugs;
    std::set<std::string> plugins =
        {"UCX", "GDS", "POSIX", "UCX_MO", "MOCK_BASIC", "MOCK_DRAM"};

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    // Initialize the plugin manager (will read env vars)
    auto& plugin_manager = nixlPluginManager::getInstance();

    // If a directory is provided as command line argument, set it as NIXL_PLUGIN_DIR
    if (argc > 1) {
        std::cout << "Adding NIXL_PLUGIN_DIR to list: " << argv[1] << std::endl;
        plugindir = argv[1];

        // Add custom plugin directory
        plugin_manager.addPluginDirectory(plugindir);
    }

    // Print list of static plugins available
    std::cout << "Available static plugins:" << std::endl;
    for (const auto& plugin : plugin_manager.getStaticPlugins()) {
        std::cout << " - " << plugin.name << std::endl;
        staticPlugs.insert(plugin.name);
    }

    // First make sure tested plugins are not already loaded
    for (const auto& plugin : plugins) {
        plugin_manager.unloadPlugin(plugin);
    }

    for (const auto& plugin : plugins) {
        verify_plugin(plugin, plugin_manager);
    }

    // List all loaded plugins
    std::cout << "\nLoaded plugins:" << std::endl;
    for (const auto& name : plugin_manager.getLoadedPluginNames()) {
        std::cout << " - " << name << std::endl;
    }

    for (const auto& plugin : plugins) {
        plugin_manager.unloadPlugin(plugin);
    }

    // List all loaded plugins and make sure static plugins are present
    std::cout << "Loaded plugins after unload:" << std::endl;
    for (const auto& name : plugin_manager.getLoadedPluginNames()) {
        std::cout << " - " << name << std::endl;
    }

    // Plugins loaded should only be the static plugins
    if (plugin_manager.getLoadedPluginNames().size() !=
        staticPlugs.size()) {
        std::cerr << "TEST FAILED: Dynamic Plugins are still loaded." << std::endl;
        return -1;
    }

    std::cout << std::endl << "TEST PASSED" << std::endl;

    return 0;
}
