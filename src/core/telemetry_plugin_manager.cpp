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

#include "telemetry_plugin_manager.h"
#include "common/nixl_log.h"
#include <dlfcn.h>
#include <filesystem>
#include <cstdlib>
#include <cassert>

using lock_guard = const std::lock_guard<std::mutex>;

void
nixlTelemetryPluginHandle::DlHandleDeleter::operator()(void *handle) const noexcept {
    if (handle) {
        using fini_func_t = void (*)();
        fini_func_t fini =
            reinterpret_cast<fini_func_t>(dlsym(handle, "nixl_telemetry_plugin_fini"));
        if (fini) {
            try {
                fini();
            }
            catch (const std::exception &e) {
                NIXL_ERROR << "Exception in plugin cleanup: " << e.what();
            }
            catch (...) {
                NIXL_ERROR << "Unknown exception in plugin cleanup";
            }
        }

        dlclose(handle);
    }
}

nixlTelemetryPluginHandle::nixlTelemetryPluginHandle(std::unique_ptr<void, DlHandleDeleter> handle,
                                                     nixlTelemetryPlugin *plugin)
    : handle_(std::move(handle)),
      plugin_(plugin) {
    assert(handle_ && "Plugin handle must not be null");
    assert(plugin_ && "Plugin interface must not be null");
}

nixlTelemetryPluginHandle::~nixlTelemetryPluginHandle() = default;

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
        auto name = plugin_->getName();
        return !name.empty() ? name.data() : "unknown";
    }
    return "unknown";
}

const char *
nixlTelemetryPluginHandle::getVersion() const {
    if (plugin_) {
        auto version = plugin_->getVersion();
        return !version.empty() ? version.data() : "unknown";
    }
    return "unknown";
}

// Helper function to get plugin directory
namespace {
static std::filesystem::path
getTelemetryPluginDir() {
    // Environment variable takes precedence
    const char *plugin_dir = getenv(TELEMETRY_EXPORTER_PLUGIN_DIR_VAR);
    if (plugin_dir) {
        return plugin_dir;
    }

    // By default, use the telemetry_exporters subdirectory relative to the library
    Dl_info info;
    int ok = dladdr(reinterpret_cast<void *>(&getTelemetryPluginDir), &info);
    if (!ok) {
        NIXL_ERROR << "Failed to get telemetry plugin directory from dladdr";
        return "";
    }
    return std::filesystem::path(info.dli_fname).parent_path() / "telemetry_exporters";
}
} // namespace

// Plugin Manager implementation
nixlTelemetryPluginManager::nixlTelemetryPluginManager() {
    std::filesystem::path plugin_dir = getTelemetryPluginDir();
    if (!plugin_dir.empty()) {
        NIXL_DEBUG << "Loading telemetry exporter plugins from: " << plugin_dir;
        plugin_dirs_.push_back(plugin_dir);

        // Auto-discover plugins if directory exists
        if (std::filesystem::exists(plugin_dir) && std::filesystem::is_directory(plugin_dir)) {
            discoverPluginsFromDir(plugin_dir);
        }
    }
}

nixlTelemetryPluginManager &
nixlTelemetryPluginManager::getInstance() {
    // Meyers singleton - thread-safe in C++11+
    static nixlTelemetryPluginManager instance;
    return instance;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlTelemetryPluginManager::loadPluginFromPath(const std::filesystem::path &plugin_path) {
    std::unique_ptr<void, nixlTelemetryPluginHandle::DlHandleDeleter> handle(
        dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL));

    if (!handle) {
        NIXL_ERROR << "Failed to load telemetry exporter plugin from " << plugin_path << ": "
                   << dlerror();
        return nullptr;
    }

    // Get the initialization function
    using init_func_t = nixlTelemetryPlugin *(*)();
    init_func_t init =
        reinterpret_cast<init_func_t>(dlsym(handle.get(), "nixl_telemetry_plugin_init"));
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_telemetry_plugin_init in " << plugin_path << ": "
                   << dlerror();
        return nullptr;
    }

    // Call the initialization function
    nixlTelemetryPlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Telemetry exporter plugin initialization failed for " << plugin_path;
        return nullptr;
    }

    // Check API version
    if (plugin->api_version != NIXL_TELEMETRY_PLUGIN_API_VERSION) {
        NIXL_ERROR << "Telemetry exporter plugin API version mismatch for " << plugin_path
                   << ": expected " << static_cast<unsigned int>(NIXL_TELEMETRY_PLUGIN_API_VERSION)
                   << ", got " << static_cast<unsigned int>(plugin->api_version);
        return nullptr;
    }

    // Create and store the plugin handle - transfer ownership from unique_ptr
    auto plugin_handle =
        std::make_shared<const nixlTelemetryPluginHandle>(std::move(handle), plugin);

    return plugin_handle;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlTelemetryPluginManager::loadPlugin(const std::string &plugin_name) {
    lock_guard lg(lock_);

    // Check if the plugin is already loaded
    auto it = loaded_plugins_.find(plugin_name);
    if (it != loaded_plugins_.end()) {
        return it->second;
    }

    // Track all directories tried for error reporting
    std::vector<std::filesystem::path> tried_paths;

    // Try to load the plugin from all registered directories
    for (const auto &dir : plugin_dirs_) {
        if (dir.empty()) {
            continue;
        }

        // Construct plugin path: libtelemetry_exporter_<name>.so
        std::filesystem::path plugin_path = dir / ("libtelemetry_exporter_" + plugin_name + ".so");

        tried_paths.push_back(plugin_path);

        // Check if the plugin file exists
        if (!std::filesystem::exists(plugin_path)) {
            NIXL_DEBUG << "Telemetry exporter plugin file does not exist: " << plugin_path;
            continue;
        }

        auto plugin_handle = loadPluginFromPath(plugin_path);
        if (plugin_handle) {
            loaded_plugins_.try_emplace(plugin_name, plugin_handle);
            NIXL_INFO << "Loaded telemetry exporter plugin: " << plugin_name << " (version "
                      << plugin_handle->getVersion() << ")";
            return plugin_handle;
        }
    }

    // Failed to load the plugin - show all directories tried
    NIXL_ERROR << "Failed to load telemetry exporter plugin '" << plugin_name
               << "' from any directory. Tried " << tried_paths.size() << " location(s):";
    for (const auto &path : tried_paths) {
        NIXL_ERROR << "  - " << path;
    }

    return nullptr;
}

std::shared_ptr<const nixlTelemetryPluginHandle>
nixlTelemetryPluginManager::getPlugin(std::string_view plugin_name) {
    lock_guard lg(lock_);

    auto it = loaded_plugins_.find(plugin_name);
    if (it != loaded_plugins_.end()) {
        return it->second;
    }
    return nullptr;
}

std::unique_ptr<nixlTelemetryExporter>
nixlTelemetryPluginManager::createExporter(std::string_view plugin_name,
                                           const nixlTelemetryExporterInitParams &init_params) {

    // Load plugin if not already loaded
    auto plugin_handle = getPlugin(plugin_name);
    if (!plugin_handle) {
        plugin_handle = loadPlugin(std::string(plugin_name));
    }

    if (!plugin_handle) {
        NIXL_ERROR << "Cannot create exporter: plugin '" << plugin_name << "' not found";
        return nullptr;
    }

    // Create the exporter instance (smart pointer handles cleanup automatically)
    auto exporter = plugin_handle->createExporter(init_params);
    if (!exporter) {
        NIXL_ERROR << "Failed to create exporter instance from plugin '" << plugin_name << "'";
        return nullptr;
    }

    NIXL_INFO << "Created telemetry exporter from plugin: " << plugin_name;
    return exporter;
}

void
nixlTelemetryPluginManager::discoverPluginsFromDir(const std::filesystem::path &dirpath) {
    std::error_code ec;
    std::filesystem::directory_iterator dir_iter(dirpath, ec);
    if (ec) {
        NIXL_ERROR << "Error accessing telemetry exporter plugin directory(" << dirpath
                   << "): " << ec.message();
        return;
    }

    for (const auto &entry : dir_iter) {
        std::string filename = entry.path().filename().string();
        NIXL_INFO << "Entry:" << filename;

        if (filename.size() < 27) continue; // "libtelemetry_exporter_X.so" min length

        // Check if this is a telemetry exporter plugin file
        if (filename.substr(0, 24) == "libtelemetry_exporter_" &&
            filename.substr(filename.size() - 3) == ".so") {

            // Extract plugin name: libtelemetry_exporter_<name>.so
            std::string plugin_name = filename.substr(24, filename.size() - 27);

            // Try to load the plugin
            auto plugin = loadPlugin(plugin_name);
            if (plugin) {
                NIXL_INFO << "Discovered and loaded telemetry exporter plugin: " << plugin_name;
            }
        }
    }
}

void
nixlTelemetryPluginManager::addPluginDirectory(const std::filesystem::path &directory) {
    if (directory.empty()) {
        NIXL_ERROR << "Cannot add empty telemetry exporter plugin directory";
        return;
    }

    // Check if directory exists
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        NIXL_ERROR << "Telemetry exporter plugin directory does not exist or is not readable: "
                   << directory;
        return;
    }

    {
        lock_guard lg(lock_);

        // Check if directory is already in the list
        for (const auto &dir : plugin_dirs_) {
            if (dir == directory) {
                NIXL_WARN << "Telemetry exporter plugin directory already registered: "
                          << directory;
                return;
            }
        }

        // Prioritize the new directory by inserting it at the beginning
        plugin_dirs_.insert(plugin_dirs_.begin(), directory);
    }

    // Discover plugins in the new directory
    discoverPluginsFromDir(directory);
}

void
nixlTelemetryPluginManager::unloadPlugin(std::string_view plugin_name) {
    lock_guard lg(lock_);
    loaded_plugins_.erase(std::string(plugin_name));
}

std::vector<std::string>
nixlTelemetryPluginManager::getLoadedPluginNames() const {
    lock_guard lg(lock_);

    std::vector<std::string> names;
    names.reserve(loaded_plugins_.size());
    for (const auto &pair : loaded_plugins_) {
        names.push_back(pair.first);
    }
    return names;
}
