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

#ifndef __PLUGIN_MANAGER_H
#define __PLUGIN_MANAGER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/backend_plugin.h"

// Forward declarations
class nixlBackendEngine;
struct nixlBackendInitParams;

class nixlPluginHandle {
private:
    void* handle_;               // Handle to the dynamically loaded library
    nixlBackendPlugin* plugin_;  // Plugin interface

public:
    nixlPluginHandle(void* handle, nixlBackendPlugin* plugin);
    ~nixlPluginHandle();

    nixlBackendEngine* createEngine(const nixlBackendInitParams* init_params);
    void destroyEngine(nixlBackendEngine* engine);
    const char* getName();
    const char* getVersion();
    nixl_b_params_t getBackendOptions();
    nixl_mem_list_t getBackendMems();
};

// Creator Function for static plugins
typedef nixlBackendPlugin* (*nixlStaticPluginCreatorFunc)();

// Structure to hold static plugin info
struct nixlStaticPluginInfo {
    const char* name;
    nixlStaticPluginCreatorFunc createFunc;
};

class nixlPluginManager {
private:
    std::map<nixl_backend_t, std::shared_ptr<nixlPluginHandle>> loaded_plugins_;
    std::vector<std::string> plugin_dirs_;

    // Static Plugins
    static std::vector<nixlStaticPluginInfo> static_plugins_;

    void registerBuiltinPlugins();

    // Private constructor for singleton pattern
    nixlPluginManager();

public:
    // Singleton instance accessor
    static nixlPluginManager& getInstance();

    // Delete copy constructor and assignment operator
    nixlPluginManager(const nixlPluginManager&) = delete;
    nixlPluginManager& operator=(const nixlPluginManager&) = delete;

    std::shared_ptr<nixlPluginHandle> loadPluginFromPath(const std::string& plugin_path);

    void loadPluginsFromList(const std::string& filename);

    // Load a specific plugin
    std::shared_ptr<nixlPluginHandle> loadPlugin(const nixl_backend_t& plugin_name);

    // Search a directory for plugins
    void discoverPluginsFromDir(const std::string& dirpath);

    // Unload a plugin
    void unloadPlugin(const nixl_backend_t& plugin_name);

    // Get a plugin handle
    std::shared_ptr<nixlPluginHandle> getPlugin(const nixl_backend_t& plugin_name);

    // Get all loaded plugin names
    std::vector<nixl_backend_t> getLoadedPluginNames();

    // Get backend options
    nixl_b_params_t getBackendOptions(const nixl_backend_t& type);

    // Add a plugin directory
    void addPluginDirectory(const std::string& directory);

    // Static Plugin Helpers
    void registerStaticPlugin(const char* name, nixlStaticPluginCreatorFunc creator);
    static std::vector<nixlStaticPluginInfo>& getStaticPlugins();
};

#endif  // __PLUGIN_MANAGER_H
