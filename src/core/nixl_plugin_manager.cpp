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
#include "common/nixl_log.h"
#include <dlfcn.h>
#include <filesystem>
#include <dirent.h>
#include <unistd.h>  // For access() and F_OK
#include <cstdlib>  // For getenv
#include <fstream>
#include <string>
#include <map>
#include <dlfcn.h>

using lock_guard = const std::lock_guard<std::mutex>;

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

nixlBackendEngine* nixlPluginHandle::createEngine(const nixlBackendInitParams* init_params) const {
    if (plugin_ && plugin_->create_engine) {
        return plugin_->create_engine(init_params);
    }
    return nullptr;
}

void nixlPluginHandle::destroyEngine(nixlBackendEngine* engine) const {
    if (plugin_ && plugin_->destroy_engine && engine) {
        plugin_->destroy_engine(engine);
    }
}

const char* nixlPluginHandle::getName() const {
    if (plugin_ && plugin_->get_plugin_name) {
        return plugin_->get_plugin_name();
    }
    return "unknown";
}

const char* nixlPluginHandle::getVersion() const {
    if (plugin_ && plugin_->get_plugin_version) {
        return plugin_->get_plugin_version();
    }
    return "unknown";
}

std::map<nixl_backend_t, std::string> loadPluginList(const std::string& filename) {
    std::map<nixl_backend_t, std::string> plugins;
    std::ifstream file(filename);

    if (!file.is_open()) {
        NIXL_ERROR << "Failed to open plugin list file: " << filename;
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

std::shared_ptr<const nixlPluginHandle> nixlPluginManager::loadPluginFromPath(const std::string& plugin_path) {
    // Open the plugin file
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        NIXL_ERROR << "Failed to load plugin from " << plugin_path << ": " << dlerror();
        return nullptr;
    }

    // Get the initialization function
    typedef nixlBackendPlugin* (*init_func_t)();
    init_func_t init = (init_func_t) dlsym(handle, "nixl_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_plugin_init in " << plugin_path << ": " << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlBackendPlugin* plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path
                   << ": expected " << NIXL_PLUGIN_API_VERSION
                   << ", got " << plugin->api_version;
        dlclose(handle);
        return nullptr;
    }

    // Create and store the plugin handle
    auto plugin_handle = std::make_shared<const nixlPluginHandle>(handle, plugin);

    return plugin_handle;
}

void nixlPluginManager::loadPluginsFromList(const std::string& filename) {
    auto plugins = loadPluginList(filename);

    lock_guard lg(lock);

    for (const auto& pair : plugins) {
        const std::string& name = pair.first;
        const std::string& path = pair.second;

        auto plugin_handle = loadPluginFromPath(path);
        if (plugin_handle) {
            loaded_plugins_[name] = plugin_handle;
        }
    }
}

namespace {
static std::string
getPluginDir() {
    // Environment variable takes precedence
    const char *plugin_dir = getenv("NIXL_PLUGIN_DIR");
    if (plugin_dir) {
        return plugin_dir;
    }
    // By default, use the plugin directory relative to the binary
    Dl_info info;
    int ok = dladdr(reinterpret_cast<void *>(&getPluginDir), &info);
    if (!ok) {
        NIXL_ERROR << "Failed to get plugin directory from dladdr";
        return "";
    }
    return (std::filesystem::path(info.dli_fname).parent_path() / "plugins").string();
}
} // namespace

// PluginManager implementation
nixlPluginManager::nixlPluginManager() {
    // Force levels right before logging
#ifdef NIXL_USE_PLUGIN_FILE
    NIXL_DEBUG << "Loading plugins from file: " << NIXL_USE_PLUGIN_FILE;
    std::string plugin_file = NIXL_USE_PLUGIN_FILE;
    if (std::filesystem::exists(plugin_file)) {
        loadPluginsFromList(plugin_file);
    }
#endif

    std::string plugin_dir = getPluginDir();
    if (!plugin_dir.empty()) {
        NIXL_DEBUG << "Loading plugins from: " << plugin_dir;
        plugin_dirs_.insert(plugin_dirs_.begin(), plugin_dir);
        discoverPluginsFromDir(plugin_dir);
    }

    registerBuiltinPlugins();
}

nixlPluginManager& nixlPluginManager::getInstance() {
    // Meyers singleton initialization is safe in multi-threaded environment.
    // Consult standard [stmt.dcl] chapter for details.
    static nixlPluginManager instance;

    return instance;
}

void nixlPluginManager::addPluginDirectory(const std::string& directory) {
    if (directory.empty()) {
        NIXL_ERROR << "Cannot add empty plugin directory";
        return;
    }

    // Check if directory exists
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        NIXL_ERROR << "Plugin directory does not exist or is not readable: " << directory;
        return;
    }

    {
        lock_guard lg(lock);

        // Check if directory is already in the list
        for (const auto& dir : plugin_dirs_) {
            if (dir == directory) {
                NIXL_WARN << "Plugin directory already registered: " << directory;
                return;
            }
        }

        // Prioritize the new directory by inserting it at the beginning
        plugin_dirs_.insert(plugin_dirs_.begin(), directory);
    }

    discoverPluginsFromDir(directory);
}

std::shared_ptr<const nixlPluginHandle> nixlPluginManager::loadPlugin(const std::string& plugin_name) {
    lock_guard lg(lock);

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
        if (!std::filesystem::exists(plugin_path)) {
            NIXL_WARN << "Plugin file does not exist: " << plugin_path;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path);
        if (plugin_handle) {
            loaded_plugins_[plugin_name] = plugin_handle;
            return plugin_handle;
        }
    }

    // Failed to load the plugin
    NIXL_ERROR << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

void nixlPluginManager::discoverPluginsFromDir(const std::string& dirpath) {
    std::filesystem::path dir_path(dirpath);
    std::error_code ec;
    std::filesystem::directory_iterator dir_iter(dir_path, ec);
    if (ec) {
        NIXL_ERROR << "Error accessing directory(" << dir_path << "): "
                   << ec.message();
        return;
    }

    for (const auto& entry : dir_iter) {
        std::string filename = entry.path().filename().string();

        if(filename.size() < 11) continue;
        // Check if this is a plugin file
        if (filename.substr(0, 10) == "libplugin_" &&
            filename.substr(filename.size() - 3) == ".so") {

            // Extract plugin name
            std::string plugin_name = filename.substr(10, filename.size() - 13);

            // Try to load the plugin
            auto plugin = loadPlugin(plugin_name);
            if (plugin) {
                NIXL_INFO << "Discovered and loaded plugin: " << plugin_name;
            }
        }
    }
}

void nixlPluginManager::unloadPlugin(const nixl_backend_t& plugin_name) {
    // Do no unload static plugins
    for (const auto& splugin : getStaticPlugins()) {
        if (splugin.name == plugin_name) {
            return;
        }
    }

    lock_guard lg(lock);

    loaded_plugins_.erase(plugin_name);
}

std::shared_ptr<const nixlPluginHandle> nixlPluginManager::getPlugin(const nixl_backend_t& plugin_name) {
    lock_guard lg(lock);

    auto it = loaded_plugins_.find(plugin_name);
    if (it != loaded_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

nixl_b_params_t nixlPluginHandle::getBackendOptions() const {
    nixl_b_params_t params;
    if (plugin_ && plugin_->get_backend_options) {
        return plugin_->get_backend_options();
    }
    return params; // Return empty params if not implemented
}

nixl_mem_list_t nixlPluginHandle::getBackendMems() const {
    nixl_mem_list_t mems;
    if (plugin_ && plugin_->get_backend_mems) {
        return plugin_->get_backend_mems();
    }
    return mems; // Return empty mems if not implemented
}

std::vector<nixl_backend_t> nixlPluginManager::getLoadedPluginNames() {
    lock_guard lg(lock);

    std::vector<nixl_backend_t> names;
    for (const auto& pair : loaded_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

void nixlPluginManager::registerStaticPlugin(const char* name, nixlStaticPluginCreatorFunc creator) {
    lock_guard lg(lock);

    nixlStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    static_plugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlPluginHandle>(nullptr, plugin);
        loaded_plugins_[name] = plugin_handle;
    }
}

const std::vector<nixlStaticPluginInfo>& nixlPluginManager::getStaticPlugins() {
    return static_plugins_;
}

#define NIXL_REGISTER_STATIC_PLUGIN(name)                   \
    extern nixlBackendPlugin *createStatic##name##Plugin(); \
    registerStaticPlugin(#name, createStatic##name##Plugin);

void nixlPluginManager::registerBuiltinPlugins() {
#ifdef STATIC_PLUGIN_LIBFABRIC
    NIXL_REGISTER_STATIC_PLUGIN(LIBFABRIC)
#endif

#ifdef STATIC_PLUGIN_UCX
    NIXL_REGISTER_STATIC_PLUGIN(UCX)
#endif

#ifdef STATIC_PLUGIN_GDS
#ifndef DISABLE_GDS_BACKEND
    NIXL_REGISTER_STATIC_PLUGIN(GDS)
#endif
#endif

#ifdef STATIC_PLUGIN_GDS_MT
    NIXL_REGISTER_STATIC_PLUGIN(GDS_MT)
#endif

#ifdef STATIC_PLUGIN_POSIX
    NIXL_REGISTER_STATIC_PLUGIN(POSIX)
#endif

#ifdef STATIC_PLUGIN_GPUNETIO
    NIXL_REGISTER_STATIC_PLUGIN(GPUNETIO)
#endif

#ifdef STATIC_PLUGIN_OBJ
    NIXL_REGISTER_STATIC_PLUGIN(OBJ)
#endif

#ifdef STATIC_PLUGIN_MOONCAKE
    NIXL_REGISTER_STATIC_PLUGIN(MOONCAKE)
#endif

#ifdef STATIC_PLUGIN_HF3FS
    NIXL_REGISTER_STATIC_PLUGIN(HF3FS)
#endif
}
