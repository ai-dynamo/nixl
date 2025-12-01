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
nixlPluginHandle::nixlPluginHandle(std::unique_ptr<void, dlHandleDeleter> handle,
                                   nixlBackendPlugin *plugin)
    : basePluginHandle(std::move(handle), plugin),
      plugin_(plugin) {}

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

bool
nixlPluginManager::checkApiVersion(void *plugin_interface) const {
    if (!plugin_interface) {
        return false;
    }

    nixlBackendPlugin *plugin = static_cast<nixlBackendPlugin *>(plugin_interface);
    return plugin->api_version == NIXL_PLUGIN_API_VERSION;
}

std::shared_ptr<basePluginHandle>
nixlPluginManager::createPluginHandle(std::unique_ptr<void, dlHandleDeleter> dl_handle,
                                      void *plugin_interface) {
    auto *plugin = static_cast<nixlBackendPlugin *>(plugin_interface);
    return std::make_shared<nixlPluginHandle>(std::move(dl_handle), plugin);
}

bool
nixlPluginManager::canUnloadPlugin(const std::string &plugin_name) const {
    // Do not unload static plugins
    for (const auto &splugin : staticPlugins_) {
        if (splugin.name == plugin_name) {
            return false;
        }
    }
    return true;
}

void
nixlPluginManager::loadPluginsFromList(const std::string &filename) {
    auto plugins = loadPluginList(filename);

    for (const auto& pair : plugins) {
        const std::string& name = pair.first;
        const std::filesystem::path path = pair.second;

        // Load using base class - it will handle storage
        basePluginManager::loadPluginFromPath<nixlPluginHandle>(path, name);
    }
}

namespace {
static std::filesystem::path
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
    return std::filesystem::path(info.dli_fname).parent_path() / "plugins";
}
} // namespace

// PluginManager implementation
nixlPluginManager::nixlPluginManager()
    : basePluginManager(pluginConfig{.initFuncName = "nixl_plugin_init",
                                     .finiFuncName = "nixl_plugin_fini",
                                     .filenamePrefix = "libplugin_",
                                     .filenameSuffix = ".so",
                                     .expectedApiVersion = NIXL_PLUGIN_API_VERSION}) {
    // Force levels right before logging
#ifdef NIXL_USE_PLUGIN_FILE
    NIXL_DEBUG << "Loading plugins from file: " << NIXL_USE_PLUGIN_FILE;
    std::string plugin_file = NIXL_USE_PLUGIN_FILE;
    if (std::filesystem::exists(plugin_file)) {
        loadPluginsFromList(plugin_file);
    }
#endif

    auto plugin_dir = getPluginDir();
    if (!plugin_dir.empty()) {
        NIXL_DEBUG << "Loading backend plugins from: " << plugin_dir;
        addPluginDirectory(plugin_dir);
    }

    registerBuiltinPlugins();
}

nixlPluginManager& nixlPluginManager::getInstance() {
    // Meyers singleton initialization is safe in multi-threaded environment.
    // Consult standard [stmt.dcl] chapter for details.
    static nixlPluginManager instance;

    return instance;
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

void
nixlPluginManager::registerStaticPlugin(const char *name, nixlStaticPluginCreatorFunc creator) {
    nixlStaticPluginInfo info;
    info.name = name;
    info.createFunc = creator;
    staticPlugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin (nullptr handle for static plugins)
        dlHandleDeleter deleter(""); // No cleanup for static plugins
        std::unique_ptr<void, dlHandleDeleter> handle(nullptr, deleter);
        auto plugin_handle = createPluginHandle(std::move(handle), plugin);

        // Store in base class map
        lock_guard lg(lock_);
        loadedPlugins_.emplace(name, plugin_handle);
    }
}

const std::vector<nixlStaticPluginInfo> &
nixlPluginManager::getStaticPlugins() const noexcept {
    return staticPlugins_;
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
