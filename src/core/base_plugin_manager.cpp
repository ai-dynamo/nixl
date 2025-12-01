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

#include "base_plugin_manager.h"
#include "common/nixl_log.h"
#include <dlfcn.h>
#include <algorithm>

using lock_guard = const std::lock_guard<std::mutex>;

void
dlHandleDeleter::operator()(void *handle) const noexcept {
    if (handle) {
        // Call cleanup function if specified
        if (!fini_func_name.empty()) {
            using fini_func_t = void (*)();
            fini_func_t fini = reinterpret_cast<fini_func_t>(dlsym(handle, fini_func_name.c_str()));
            if (fini) {
                try {
                    fini();
                }
                catch (const std::exception &e) {
                    NIXL_WARN << "Exception in plugin cleanup (" << fini_func_name
                              << "): " << e.what();
                }
                catch (...) {
                    NIXL_WARN << "Unknown exception in plugin cleanup (" << fini_func_name << ")";
                }
            }
        }

        dlclose(handle);
    }
}

basePluginHandle::basePluginHandle(std::unique_ptr<void, dlHandleDeleter> handle,
                                   const void *plugin_interface)
    : dlHandle_(std::move(handle)),
      pluginInterface_(plugin_interface) {
    assert(dlHandle_ && "DlHandleDeleter must not be null");
    assert(pluginInterface_ && "Plugin interface must not be null");
}

basePluginManager::basePluginManager(pluginConfig config) : config_(std::move(config)) {}

plugin_load_result<void>
basePluginManager::loadPluginFromPathInternal(const std::filesystem::path &plugin_path) {
    plugin_load_result<void> result;

    dlHandleDeleter deleter(config_.finiFuncName);
    std::unique_ptr<void, dlHandleDeleter> handle(
        dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL), deleter);

    if (!handle) {
        result = failure_result{std::string("Failed to dlopen: ") + dlerror()};
        NIXL_ERROR << "Failed to load plugin from " << plugin_path << ": "
                   << std::get<failure_result>(result).message;
        return result;
    }

    using init_func_t = void *(*)();
    init_func_t init =
        reinterpret_cast<init_func_t>(dlsym(handle.get(), config_.initFuncName.c_str()));

    if (!init) {
        result = failure_result{std::string("Failed to find ") + config_.initFuncName + ": " +
                                dlerror()};
        NIXL_ERROR << "Failed to find " << config_.initFuncName << " in " << plugin_path << ": "
                   << std::get<failure_result>(result).message;
        return result;
    }

    void *plugin_interface = init();
    if (!plugin_interface) {
        result = failure_result{"Plugin initialization returned nullptr"};
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        return result;
    }

    if (!checkApiVersion(plugin_interface)) {
        result = failure_result{"API version mismatch"};
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path;
        return result;
    }

    result = success_result<void>{std::move(handle), plugin_interface};

    return result;
}

std::string
basePluginManager::extractPluginNameFromFilename(const std::string &filename) const {
    const auto &prefix = config_.filenamePrefix;
    const auto &suffix = config_.filenameSuffix;

    const size_t min_length = prefix.size() + suffix.size() + 1; // +1 for at least 1 char name
    if (filename.size() < min_length) {
        return "";
    }

    if (filename.compare(0, prefix.size(), prefix) != 0) {
        return "";
    }

    if (filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return "";
    }

    return filename.substr(prefix.size(), filename.size() - prefix.size() - suffix.size());
}

std::filesystem::path
basePluginManager::constructPluginPath(const std::filesystem::path &directory,
                                       const std::string &plugin_name) const {

    std::string filename = config_.filenamePrefix + plugin_name + config_.filenameSuffix;
    return directory / filename;
}

void
basePluginManager::discoverPluginsFromDir(const std::filesystem::path &dirpath) {
    std::error_code ec;
    // Use recursive iterator to find plugins in subdirectories too (for build directories)
    std::filesystem::recursive_directory_iterator dir_iter(dirpath, ec);
    if (ec) {
        NIXL_ERROR << "Error accessing plugin directory(" << dirpath << "): " << ec.message();
        return;
    }

    for (const auto &entry : dir_iter) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }

        std::string filename = entry.path().filename().string();
        std::string plugin_name = extractPluginNameFromFilename(filename);

        if (!plugin_name.empty()) {
            // Restore old behavior: actually load the plugin during discovery
            auto plugin = loadPlugin(plugin_name);
            if (plugin) {
                NIXL_INFO << "Discovered and loaded plugin: " << plugin_name;
            }
        }
    }
}

void
basePluginManager::addPluginDirectory(const std::filesystem::path &directory) {
    if (directory.empty()) {
        NIXL_ERROR << "Cannot add empty plugin directory";
        return;
    }

    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        NIXL_ERROR << "Plugin directory does not exist or is not readable: " << directory;
        return;
    }

    {
        lock_guard lg(lock_);

        if (std::find(pluginDirs_.begin(), pluginDirs_.end(), directory) != pluginDirs_.end()) {
            NIXL_WARN << "Plugin directory already registered: " << directory;
            return;
        }

        pluginDirs_.insert(pluginDirs_.begin(), directory);
    }

    NIXL_INFO << "Added plugin directory: " << directory;

    discoverPluginsFromDir(directory);
}

std::vector<std::filesystem::path>
basePluginManager::getPluginDirectories() const {
    lock_guard lg(lock_);
    return pluginDirs_;
}

std::shared_ptr<basePluginHandle>
basePluginManager::loadPluginInternal(const std::string &plugin_name) {
    lock_guard lg(lock_);

    // Check if the plugin is already loaded
    auto it = loadedPlugins_.find(plugin_name);
    if (it != loadedPlugins_.end()) {
        return it->second;
    }

    // Try to load the plugin from all registered directories
    for (const auto &dir : pluginDirs_) {
        if (dir.empty()) {
            continue;
        }

        // Construct expected plugin path in this directory
        auto plugin_path = constructPluginPath(dir, plugin_name);

        // Skip if plugin file doesn't exist in this directory
        if (!std::filesystem::exists(plugin_path)) {
            NIXL_DEBUG << "Plugin not found at: " << plugin_path;
            continue;
        }

        // Load the plugin
        auto result = loadPluginFromPathInternal(plugin_path);
        if (std::holds_alternative<success_result<void>>(result)) {
            auto &success = std::get<success_result<void>>(result);
            auto plugin_handle = createPluginHandle(std::move(success.handle), success.interface);

            if (plugin_handle) {
                loadedPlugins_.emplace(plugin_name, plugin_handle);
                NIXL_INFO << "Loaded plugin: " << plugin_name << " (version "
                          << plugin_handle->getVersion() << ")";
                onPluginLoaded(plugin_name, success.interface);
                return plugin_handle;
            }
        }
    }

    // Failed to load the plugin
    NIXL_ERROR << "Failed to load plugin '" << plugin_name << "' from any directory";
    return nullptr;
}

std::shared_ptr<basePluginHandle>
basePluginManager::getPluginInternal(const std::string &plugin_name) const {
    lock_guard lg(lock_);

    auto it = loadedPlugins_.find(plugin_name);
    if (it != loadedPlugins_.end()) {
        return it->second;
    }
    return nullptr;
}

void
basePluginManager::unloadPlugin(const std::string &plugin_name) {
    // Check if plugin can be unloaded (e.g., not a static plugin)
    if (!canUnloadPlugin(plugin_name)) {
        NIXL_DEBUG << "Plugin '" << plugin_name << "' cannot be unloaded";
        return;
    }

    lock_guard lg(lock_);
    loadedPlugins_.erase(plugin_name);
}

std::vector<std::string>
basePluginManager::getLoadedPluginNames() const {
    lock_guard lg(lock_);

    std::vector<std::string> names;
    names.reserve(loadedPlugins_.size());
    for (const auto &pair : loadedPlugins_) {
        names.push_back(pair.first);
    }
    return names;
}
