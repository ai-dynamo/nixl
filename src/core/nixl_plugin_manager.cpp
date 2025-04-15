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

#include "plugin_manager.h"
#include "nixl.h"
#include <dlfcn.h>
#include <iostream>
#include <filesystem>
#include <dirent.h>
#include <unistd.h>  // For access() and F_OK
#include <cstdlib>  // For getenv
#include <fstream>
#include <iostream>
#include <string>
#include <map>

// pluginHandle implementation
nixlPluginHandle::nixlPluginHandle(void* handle, nixlBackendPlugin* plugin)
    : handle_(handle), plugin_(plugin) {
}

nixlPluginHandle::~nixlPluginHandle() {
    if (handle_) {
        // Call the plugin's cleanup function
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t) dlsym(handle_, "nixl_plugin_fini");
        if (fini) {
            fini();
        }

        // Close the dynamic library
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

nixlBackendEngine* nixlPluginHandle::createEngine(const nixlBackendInitParams* init_params) {
    if (plugin_ && plugin_->create_engine) {
        return plugin_->create_engine(init_params);
    }
    return nullptr;
}

void nixlPluginHandle::destroyEngine(nixlBackendEngine* engine) {
    if (plugin_ && plugin_->destroy_engine && engine) {
        plugin_->destroy_engine(engine);
    }
}

const char* nixlPluginHandle::getName() {
    if (plugin_ && plugin_->get_plugin_name) {
        return plugin_->get_plugin_name();
    }
    return "unknown";
}

const char* nixlPluginHandle::getVersion() {
    if (plugin_ && plugin_->get_plugin_version) {
        return plugin_->get_plugin_version();
    }
    return "unknown";
}

std::map<nixl_backend_t, std::string> loadPluginList(const std::string& filename) {
    std::map<nixl_backend_t, std::string> plugins;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open plugin list file: " << filename << std::endl;
        return plugins;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find the equals sign
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string name = line.substr(0, pos);
            std::string path = line.substr(pos + 1);

            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
            };
            trim(name);
            trim(path);

            // Add to map
            plugins[name] = path;
        }
    }

    return plugins;
}

std::shared_ptr<nixlPluginHandle> nixlPluginManager::loadPluginFromPath(const std::string& plugin_path) {
    // Open the plugin file
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "Failed to load plugin from " << plugin_path <<
                     ": " << dlerror() << std::endl;
        return nullptr;
    }

    // Get the initialization function
    typedef nixlBackendPlugin* (*init_func_t)();
    init_func_t init = (init_func_t) dlsym(handle, "nixl_plugin_init");
    if (!init) {
        std::cerr << "Failed to find nixl_plugin_init in " << plugin_path
                    << ": " << dlerror() << std::endl;
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlBackendPlugin* plugin = init();
    if (!plugin) {
        std::cerr << "Plugin initialization failed for " << plugin_path << std::endl;
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
        std::cerr << "Plugin API version mismatch for " << plugin_path
                    << ": expected " << NIXL_PLUGIN_API_VERSION
                    << ", got " << plugin->api_version << std::endl;
        dlclose(handle);
        return nullptr;
    }

    // Create and store the plugin handle
    auto plugin_handle = std::make_shared<nixlPluginHandle>(handle, plugin);

    return plugin_handle;
}

void nixlPluginManager::loadPluginsFromList(const std::string& filename) {
    auto plugins = loadPluginList(filename);

    for (const auto& pair : plugins) {
        const std::string& name = pair.first;
        const std::string& path = pair.second;

        auto plugin_handle = loadPluginFromPath(path);
        if (plugin_handle) {
            loaded_plugins_[name] = plugin_handle;
        }
    }
}

// PluginManager implementation
nixlPluginManager::nixlPluginManager() {
#ifdef NIXL_USE_PLUGIN_FILE
    std::string plugin_file = NIXL_USE_PLUGIN_FILE;
    if (access(plugin_file.c_str(), F_OK) == 0) {
        loadPluginsFromList(plugin_file);
    }
#endif

    // Check for NIXL_PLUGIN_DIR environment variable
    const char* plugin_dir = getenv("NIXL_PLUGIN_DIR");
    if (plugin_dir) {
        plugin_dirs_.insert(plugin_dirs_.begin(), plugin_dir);  // Insert at the beginning for priority
        discoverPluginsFromDir(plugin_dir);
    }
}

nixlPluginManager& nixlPluginManager::getInstance() {
    static nixlPluginManager instance;

    // Only register built-in plugins once
    static bool registered = false;
    if (!registered) {
        instance.registerBuiltinPlugins();
        registered = true;
    }

    return instance;
}

void nixlPluginManager::addPluginDirectory(const std::string& directory) {
    if (directory.empty()) {
        std::cerr << "Cannot add empty plugin directory" << std::endl;
        return;
    }

    // Check if directory exists
    if (access(directory.c_str(), F_OK | R_OK) != 0) {
        std::cerr << "Plugin directory does not exist or is not readable: " << directory << std::endl;
        return;
    }

    // Check if directory is already in the list
    for (const auto& dir : plugin_dirs_) {
        if (dir == directory) {
            std::cout << "WARNING: Plugin directory already registered: " << directory << std::endl;
            return;
        }
    }

    // Prioritize the new directory by inserting it at the beginning
    plugin_dirs_.insert(plugin_dirs_.begin(), directory);
    discoverPluginsFromDir(directory);
}

std::shared_ptr<nixlPluginHandle> nixlPluginManager::loadPlugin(const std::string& plugin_name) {
    // Check if the plugin is already loaded
    // Static Plugins are preloaded so return handle
    auto it = loaded_plugins_.find(plugin_name);
    if (it != loaded_plugins_.end()) {
        return it->second;
    }

    // Try to load the plugin from all registered directories
    for (const auto& dir : plugin_dirs_) {
        // Handle path joining correctly with or without trailing slash
        std::string plugin_path;
        if (dir.empty()) {
            continue;
        } else if (dir.back() == '/') {
            plugin_path = dir + "libplugin_" + plugin_name + ".so";
        } else {
            plugin_path = dir + "/libplugin_" + plugin_name + ".so";
        }

        // Check if the plugin file exists before attempting to load i
        if (access(plugin_path.c_str(), F_OK) != 0) {
            std::cerr << "Plugin file does not exist: " << plugin_path << std::endl;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path);
        if (plugin_handle) {
            loaded_plugins_[plugin_name] = plugin_handle;
            return plugin_handle;
        }
    }

    // Failed to load the plugin
    std::cerr << "Failed to load plugin '" << plugin_name << "' from any directory" << std::endl;
    return nullptr;
}

void nixlPluginManager::discoverPluginsFromDir(const std::string& dirpath) {
    DIR* dp = opendir(dirpath.c_str());
    if (!dp) {
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dp))) {
        std::string filename = entry->d_name;

        if(filename.size() < 11) continue;
        // Check if this is a plugin file
        if (filename.substr(0, 10) == "libplugin_" &&
            filename.substr(filename.size() - 3) == ".so") {

            // Extract plugin name
            std::string plugin_name = filename.substr(10, filename.size() - 13);

            // Try to load the plugin
            auto plugin = loadPlugin(plugin_name);
            // if (plugin) { // To be used in logging infra
            //    std::cout << "Loaded plugin " << plugin_name << std::endl;
            // }
        }
    }

    closedir(dp);
}

void nixlPluginManager::unloadPlugin(const nixl_backend_t& plugin_name) {
    // Do no unload static plugins
    for (const auto& splugin : getStaticPlugins()) {
        if (splugin.name == plugin_name) {
            return;
        }
    }
    loaded_plugins_.erase(plugin_name);
}

std::shared_ptr<nixlPluginHandle> nixlPluginManager::getPlugin(const nixl_backend_t& plugin_name) {
    auto it = loaded_plugins_.find(plugin_name);
    if (it != loaded_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

nixl_b_params_t nixlPluginHandle::getBackendOptions() {
    nixl_b_params_t params;
    if (plugin_ && plugin_->get_backend_options) {
        return plugin_->get_backend_options();
    }
    return params; // Return empty params if not implemented
}

nixl_mem_list_t nixlPluginHandle::getBackendMems() {
    nixl_mem_list_t mems;
    if (plugin_ && plugin_->get_backend_mems) {
        return plugin_->get_backend_mems();
    }
    return mems; // Return empty mems if not implemented
}

std::vector<nixl_backend_t> nixlPluginManager::getLoadedPluginNames() {
    std::vector<nixl_backend_t> names;
    for (const auto& pair : loaded_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

// Static Plugin Helpers
std::vector<nixlStaticPluginInfo> nixlPluginManager::static_plugins_;

void nixlPluginManager::registerStaticPlugin(const char* name, nixlStaticPluginCreatorFunc creator) {
    nixlStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    static_plugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<nixlPluginHandle>(nullptr, plugin);
        loaded_plugins_[name] = plugin_handle;
    }
}

std::vector<nixlStaticPluginInfo>& nixlPluginManager::getStaticPlugins() {
    return static_plugins_;
}

void nixlPluginManager::registerBuiltinPlugins() {
    #ifdef STATIC_PLUGIN_UCX
        extern nixlBackendPlugin* createStaticUcxPlugin();
        registerStaticPlugin("UCX", createStaticUcxPlugin);
    #endif

    #ifdef STATIC_PLUGIN_UCX_MO
        extern nixlBackendPlugin* createStaticUcxMoPlugin();
        registerStaticPlugin("UCX_MO", createStaticUcxMoPlugin);
    #endif

    #ifdef STATIC_PLUGIN_GDS
    #ifndef DISABLE_GDS_BACKEND
        extern nixlBackendPlugin* createStaticGdsPlugin();
        registerStaticPlugin("GDS", createStaticGdsPlugin);
    #endif
    #endif
}
