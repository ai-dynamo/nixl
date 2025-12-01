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

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include "backend/backend_plugin.h"
#include "base_plugin_manager.h"

// Forward declarations
class nixlBackendEngine;
struct nixlBackendInitParams;

/**
 * This class represents a NIXL plugin and is used to create plugin instances. nixlPluginHandle
 * attributes are modified only in the constructor and destructor and remain unchanged during normal
 * operation, e.g., query operations and plugin instance creation. This allows using it in
 * multi-threading environments without lock protection.
 */
class nixlPluginHandle : public basePluginHandle {
private:
    nixlBackendPlugin *plugin_;

public:
    nixlPluginHandle(std::unique_ptr<void, dlHandleDeleter> handle, nixlBackendPlugin *plugin);
    ~nixlPluginHandle() override = default;

    nixlBackendEngine* createEngine(const nixlBackendInitParams* init_params) const;
    void destroyEngine(nixlBackendEngine* engine) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;
    nixl_b_params_t getBackendOptions() const;
    nixl_mem_list_t getBackendMems() const;
};

// Creator Function for static plugins
typedef nixlBackendPlugin* (*nixlStaticPluginCreatorFunc)();

// Structure to hold static plugin info
struct nixlStaticPluginInfo {
    const char* name;
    nixlStaticPluginCreatorFunc createFunc;
};

class nixlPluginManager : public basePluginManager {
private:
    std::vector<nixlStaticPluginInfo> staticPlugins_;

    void registerBuiltinPlugins();
    void registerStaticPlugin(const char* name, nixlStaticPluginCreatorFunc creator);

    // Private constructor for singleton pattern
    nixlPluginManager();

protected:
    bool
    checkApiVersion(void *plugin_interface) const override;

    std::shared_ptr<basePluginHandle>
    createPluginHandle(std::unique_ptr<void, dlHandleDeleter> dl_handle,
                       void *plugin_interface) override;

    bool
    canUnloadPlugin(const std::string &plugin_name) const override;

public:
    // Singleton instance accessor
    static nixlPluginManager& getInstance();

    // Delete copy constructor and assignment operator
    nixlPluginManager(const nixlPluginManager&) = delete;
    nixlPluginManager& operator=(const nixlPluginManager&) = delete;

    // Backend-specific plugin loading
    void
    loadPluginsFromList(const std::string &filename);

    // Get backend options
    nixl_b_params_t
    getBackendOptions(const nixl_backend_t &type);

    // Static Plugin Helpers
    const std::vector<nixlStaticPluginInfo> &
    getStaticPlugins() const noexcept;
};

#endif // __PLUGIN_MANAGER_H
