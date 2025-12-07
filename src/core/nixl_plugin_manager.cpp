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

constexpr const char *backendPluginPrefix = "libplugin_";
constexpr const char *telemetryPluginPrefix = "libtelemetry_exporter_";
constexpr const std::string_view kPluginSuffix = ".so";

// pluginHandle implementation
nixlBackendPluginHandle::nixlBackendPluginHandle(void *handle, nixlBackendPlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlBackendPluginHandle::~nixlBackendPluginHandle() {
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

nixlBackendEngine *
nixlBackendPluginHandle::createEngine(const nixlBackendInitParams *init_params) const {
    if (plugin_ && plugin_->create_engine) {
        return plugin_->create_engine(init_params);
    }
    return nullptr;
}

void
nixlBackendPluginHandle::destroyEngine(nixlBackendEngine *engine) const {
    if (plugin_ && plugin_->destroy_engine && engine) {
        plugin_->destroy_engine(engine);
    }
}

const char *
nixlBackendPluginHandle::getName() const {
    if (plugin_ && plugin_->get_plugin_name) {
        return plugin_->get_plugin_name();
    }
    return "unknown";
}

const char *
nixlBackendPluginHandle::getVersion() const {
    if (plugin_ && plugin_->get_plugin_version) {
        return plugin_->get_plugin_version();
    }
    return "unknown";
}

namespace {
// Backend plugin loader
std::shared_ptr<const nixlPluginHandle>
backendLoader(void *handle, const std::string_view &plugin_path) {
    // Get the initialization function
    typedef nixlBackendPlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_plugin_init in " << plugin_path.data() << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlBackendPlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path.data();
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path.data() << ": expected "
                   << NIXL_PLUGIN_API_VERSION << ", got " << plugin->api_version;
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlBackendPluginHandle>(handle, plugin);
}
} // namespace

nixlTelemetryPluginHandle::nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlTelemetryPluginHandle::~nixlTelemetryPluginHandle() {
    if (handle_) {
        // Call the plugin's cleanup function
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_telemetry_plugin_fini");
        if (fini) {
            fini();
        }

        // Close the dynamic library
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

std::unique_ptr<nixlTelemetryExporter>
nixlTelemetryPluginHandle::createExporter(
    const nixlTelemetryExporterInitParams &init_params) const {
    if (plugin_ && plugin_->create_exporter) {
        return plugin_->create_exporter(init_params);
    }
    return nullptr;
}

const char *
nixlTelemetryPluginHandle::getName() const {
    if (plugin_) {
        return plugin_->getName().data();
    }
    return "unknown";
}

const char *
nixlTelemetryPluginHandle::getVersion() const {
    if (plugin_) {
        return plugin_->getVersion().data();
    }
    return "unknown";
}

namespace {
// Telemetry plugin loader
std::shared_ptr<const nixlPluginHandle>
telemetryLoader(void *handle, const std::string_view &plugin_path) {
    // Get the initialization function
    typedef nixlTelemetryPlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_telemetry_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_telemetry_plugin_init in " << plugin_path.data() << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    // Call the initialization function
    nixlTelemetryPlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path.data();
        dlclose(handle);
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != nixlTelemetryPluginApiVersionV1) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path.data() << ": expected "
                   << static_cast<int>(nixlTelemetryPluginApiVersionV1) << ", got "
                   << static_cast<int>(plugin->api_version);
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlTelemetryPluginHandle>(handle, plugin);
}
} // namespace

std::map<nixl_backend_t, std::string>
loadPluginList(const std::string &filename) {
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

std::shared_ptr<const nixlPluginHandle>
nixlPluginManager::loadPluginFromPath(const std::string &plugin_path, nixlPluginLoaderFunc loader) {
    // Open the plugin file
    void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        NIXL_INFO << "Failed to load plugin from " << plugin_path << ": " << dlerror();
        return nullptr;
    }

    return loader(handle, plugin_path);
}

void
nixlPluginManager::loadPluginsFromList(const std::string &filename) {
    auto plugins = loadPluginList(filename);

    lock_guard lg(lock);

    for (const auto& pair : plugins) {
        const std::string& name = pair.first;
        const std::string& path = pair.second;

        auto plugin_handle = loadPluginFromPath(path, backendLoader);
        if (plugin_handle) {
            auto backend_plugin =
                std::dynamic_pointer_cast<const nixlBackendPluginHandle>(plugin_handle);
            loaded_backend_plugins_[name] = backend_plugin;
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

void
nixlPluginManager::addPluginDirectory(const std::string &directory) {
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

std::string
nixlPluginManager::composePluginPath(const std::string &dir,
                                     const std::string &plugin_prefix,
                                     const std::string &plugin_name) {
    std::string plugin_path;
    if (dir.empty()) {
        return "";
    } else if (dir.back() == '/') {
        plugin_path = dir + plugin_prefix + plugin_name + ".so";
    } else {
        plugin_path = dir + "/" + plugin_prefix + plugin_name + ".so";
    }
    return plugin_path;
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::loadBackendPlugin(const std::string &plugin_name) {
    lock_guard lg(lock);

    // Check if the plugin is already loaded
    // Static Plugins are preloaded so return handle
    auto it = loaded_backend_plugins_.find(plugin_name);
    if (it != loaded_backend_plugins_.end()) {
        return it->second;
    }

    // Try to load the plugin from all registered directories
    for (const auto& dir : plugin_dirs_) {
        std::string plugin_path = composePluginPath(dir, backendPluginPrefix, plugin_name);
        if (plugin_path.empty()) {
            continue;
        }

        if (!std::filesystem::exists(plugin_path)) {
            NIXL_WARN << "Plugin file does not exist: " << plugin_path;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path, backendLoader);
        if (plugin_handle) {
            auto backend_plugin =
                std::dynamic_pointer_cast<const nixlBackendPluginHandle>(plugin_handle);
            loaded_backend_plugins_[plugin_name] = backend_plugin;
            return backend_plugin;
        }
    }

    // Failed to load the plugin
    NIXL_INFO << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::loadTelemetryPlugin(const std::string &plugin_name) {
    lock_guard lg(lock);

    // Check if the plugin is already loaded
    auto it = loaded_telemetry_plugins_.find(plugin_name);
    if (it != loaded_telemetry_plugins_.end()) {
        return it->second;
    }

    // Try to load the plugin from all registered directories
    for (const auto &dir : plugin_dirs_) {
        std::string plugin_path = composePluginPath(dir, telemetryPluginPrefix, plugin_name);
        if (plugin_path.empty()) {
            continue;
        }

        if (!std::filesystem::exists(plugin_path)) {
            NIXL_WARN << "Plugin file does not exist: " << plugin_path;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path, telemetryLoader);
        if (plugin_handle) {
            auto telemetry_plugin =
                std::dynamic_pointer_cast<const nixlTelemetryPluginHandle>(plugin_handle);
            loaded_telemetry_plugins_[plugin_name] = telemetry_plugin;
            return telemetry_plugin;
        }
    }

    NIXL_INFO << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

namespace {
static bool
startsWith(const std::string &str, const std::string &prefix) {
    return str.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), str.begin());
}

static bool
endsWith(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

static std::string
extractPluginName(const std::string &filename, const std::string &prefix) {
    return filename.substr(prefix.size(), filename.size() - prefix.size() - kPluginSuffix.size());
}
} // namespace

void
nixlPluginManager::discoverBackendPlugin(const std::string &filename) {
    if (startsWith(filename, backendPluginPrefix) && endsWith(filename, kPluginSuffix.data())) {
        std::string plugin_name = extractPluginName(filename, backendPluginPrefix);
        auto plugin = loadBackendPlugin(plugin_name);
        if (plugin) {
            NIXL_INFO << "Discovered and loaded backend plugin: " << plugin_name;
        }
    }
}

void
nixlPluginManager::discoverTelemetryPlugin(const std::string &filename) {
    if (startsWith(filename, telemetryPluginPrefix) && endsWith(filename, kPluginSuffix.data())) {
        std::string plugin_name = extractPluginName(filename, telemetryPluginPrefix);

        auto plugin = loadTelemetryPlugin(plugin_name);
        if (plugin) {
            NIXL_INFO << "Discovered and loaded telemetry plugin: " << plugin_name;
        }
    }
}

void
nixlPluginManager::discoverPluginsFromDir(const std::string_view &dirpath) {
    std::filesystem::path dir_path(dirpath.data());
    std::error_code ec;
    std::filesystem::directory_iterator dir_iter(dir_path, ec);
    if (ec) {
        NIXL_ERROR << "Error accessing directory(" << dir_path << "): "
                   << ec.message();
        return;
    }

    for (const auto& entry : dir_iter) {
        std::string filename = entry.path().filename().string();
        discoverBackendPlugin(filename);
        discoverTelemetryPlugin(filename);
    }
}

void
nixlPluginManager::unloadBackendPlugin(const nixl_backend_t &plugin_name) {
    // Do not unload static plugins
    for (const auto &splugin : getBackendStaticPlugins()) {
        if (splugin.name == plugin_name) {
            return;
        }
    }

    lock_guard lg(lock);

    loaded_backend_plugins_.erase(plugin_name);
}

void
nixlPluginManager::unloadTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name) {
    // Do not unload static plugins
    for (const auto &splugin : getTelemetryStaticPlugins()) {
        if (splugin.name == plugin_name) {
            return;
        }
    }

    lock_guard lg(lock);

    loaded_telemetry_plugins_.erase(plugin_name);
}

std::shared_ptr<const nixlBackendPluginHandle>
nixlPluginManager::getBackendPlugin(const nixl_backend_t &plugin_name) {
    lock_guard lg(lock);

    auto it = loaded_backend_plugins_.find(plugin_name);
    if (it != loaded_backend_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlPluginManager::getTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name) {
    lock_guard lg(lock);
    auto it = loaded_telemetry_plugins_.find(plugin_name);
    if (it != loaded_telemetry_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

nixl_b_params_t
nixlBackendPluginHandle::getBackendOptions() const {
    nixl_b_params_t params;
    if (plugin_ && plugin_->get_backend_options) {
        return plugin_->get_backend_options();
    }
    return params; // Return empty params if not implemented
}

nixl_mem_list_t
nixlBackendPluginHandle::getBackendMems() const {
    nixl_mem_list_t mems;
    if (plugin_ && plugin_->get_backend_mems) {
        return plugin_->get_backend_mems();
    }
    return mems; // Return empty mems if not implemented
}

std::vector<nixl_backend_t>
nixlPluginManager::getLoadedBackendPluginNames() {
    lock_guard lg(lock);

    std::vector<nixl_backend_t> names;
    for (const auto &pair : loaded_backend_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

std::vector<nixl_telemetry_plugin_t>
nixlPluginManager::getLoadedTelemetryPluginNames() {
    lock_guard lg(lock);

    std::vector<nixl_telemetry_plugin_t> names;
    for (const auto &pair : loaded_telemetry_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}

void
nixlPluginManager::registerBackendStaticPlugin(const std::string_view &name,
                                               nixlStaticPluginCreatorFunc creator) {
    lock_guard lg(lock);

    nixlBackendStaticPluginInfo info;
    info.name = name.data();
    info.createFunc = creator;
    backend_static_plugins_.push_back(info);

    //Static Plugins are considered pre-loaded
    nixlBackendPlugin* plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlBackendPluginHandle>(nullptr, plugin);
        loaded_backend_plugins_[name.data()] = plugin_handle;
    }
}

void
nixlPluginManager::registerTelemetryStaticPlugin(const std::string_view &name,
                                                 nixlTelemetryStaticPluginCreatorFunc creator) {
    lock_guard lg(lock);

    nixlTelemetryStaticPluginInfo info;
    info.name = name.data();
    info.createFunc = creator;
    telemetry_static_plugins_.push_back(info);

    // Static Plugins are considered pre-loaded
    nixlTelemetryPlugin *plugin = info.createFunc();
    NIXL_INFO << "Loading static plugin: " << name;
    if (plugin) {
        // Register the loaded plugin
        auto plugin_handle = std::make_shared<const nixlTelemetryPluginHandle>(nullptr, plugin);
        loaded_telemetry_plugins_[name.data()] = plugin_handle;
    }
}

const std::vector<nixlBackendStaticPluginInfo> &
nixlPluginManager::getBackendStaticPlugins() {
    return backend_static_plugins_;
}

const std::vector<nixlTelemetryStaticPluginInfo> &
nixlPluginManager::getTelemetryStaticPlugins() {
    return telemetry_static_plugins_;
}

#define NIXL_REGISTER_STATIC_PLUGIN(plugin_type, name)              \
    extern nixl##plugin_type##Plugin *createStatic##name##Plugin(); \
    register##plugin_type##StaticPlugin(#name, createStatic##name##Plugin);

void nixlPluginManager::registerBuiltinPlugins() {
#ifdef STATIC_PLUGIN_LIBFABRIC
    NIXL_REGISTER_STATIC_PLUGIN(Backend, LIBFABRIC)
#endif

#ifdef STATIC_PLUGIN_UCX
    NIXL_REGISTER_STATIC_PLUGIN(Backend, UCX)
#endif

#ifdef STATIC_PLUGIN_GDS
#ifndef DISABLE_GDS_BACKEND
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GDS)
#endif
#endif

#ifdef STATIC_PLUGIN_GDS_MT
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GDS_MT)
#endif

#ifdef STATIC_PLUGIN_POSIX
    NIXL_REGISTER_STATIC_PLUGIN(Backend, POSIX)
#endif

#ifdef STATIC_PLUGIN_GPUNETIO
    NIXL_REGISTER_STATIC_PLUGIN(Backend, GPUNETIO)
#endif

#ifdef STATIC_PLUGIN_OBJ
    NIXL_REGISTER_STATIC_PLUGIN(Backend, OBJ)
#endif

#ifdef STATIC_PLUGIN_MOONCAKE
    NIXL_REGISTER_STATIC_PLUGIN(Backend, MOONCAKE)
#endif

#ifdef STATIC_PLUGIN_HF3FS
    NIXL_REGISTER_STATIC_PLUGIN(Backend, HF3FS)
#endif
}
