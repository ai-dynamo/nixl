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

#ifndef __BASE_PLUGIN_MANAGER_H
#define __BASE_PLUGIN_MANAGER_H

#include "common/nixl_log.h"

#include <string>
#include <vector>
#include <variant>
#include <mutex>
#include <filesystem>
#include <memory>
#include <functional>
#include <map>

/**
 * Custom deleter for dlopen handles
 */
struct dlHandleDeleter {
    std::string fini_func_name;

    explicit dlHandleDeleter(std::string fini_name = "") : fini_func_name(std::move(fini_name)) {}

    void
    operator()(void *handle) const noexcept;
};

struct pluginConfig {
    std::string initFuncName; // e.g., "nixl_plugin_init"
    std::string finiFuncName; // e.g., "nixl_plugin_fini"
    std::string filenamePrefix; // e.g., "libplugin_"
    std::string filenameSuffix; // e.g., ".so"
    int expectedApiVersion; // Expected API version for validation
};

template<typename pluginInterface> struct success_result {
    std::unique_ptr<void, dlHandleDeleter> handle;
    pluginInterface *interface;
};

struct failure_result {
    std::string message;
};

template<typename pluginInterface>
using plugin_load_result = std::variant<failure_result, success_result<pluginInterface>>;

/**
 * Base class for all plugin handles
 * Provides common functionality for managing dynamic library handles
 */
class basePluginHandle {
public:
    virtual ~basePluginHandle() = default;

    basePluginHandle(const basePluginHandle &) = delete;
    basePluginHandle &
    operator=(const basePluginHandle &) = delete;
    basePluginHandle(basePluginHandle &&) = delete;
    basePluginHandle &
    operator=(basePluginHandle &&) = delete;

    virtual const char *
    getName() const = 0;
    virtual const char *
    getVersion() const = 0;

    const void *
    getPluginInterface() const {
        return pluginInterface_;
    }

protected:
    std::unique_ptr<void, dlHandleDeleter> dlHandle_;
    const void *pluginInterface_;

    basePluginHandle(std::unique_ptr<void, dlHandleDeleter> handle, const void *pluginInterface);
};

/**
 * Base plugin manager providing common functionality for all plugin types
 *
 * This class implements the common plugin loading, discovery, and management
 * logic that is shared across different plugin types (backend, telemetry, etc.)
 */
class basePluginManager {
public:
    virtual ~basePluginManager() = default;

    basePluginManager(const basePluginManager &) = delete;
    basePluginManager &
    operator=(const basePluginManager &) = delete;

    /**
     * Discover and load all plugins from a directory
     * Searches for files matching the configured pattern
     */
    void
    discoverPluginsFromDir(const std::filesystem::path &dirpath);

    /**
     * Add a directory to search for plugins
     * New directory is prioritized over existing ones
     */
    void
    addPluginDirectory(const std::filesystem::path &directory);

    /**
     * Get all registered plugin directories
     */
    std::vector<std::filesystem::path>
    getPluginDirectories() const;

    /**
     * Get plugin configuration
     */
    const pluginConfig &
    getConfig() const {
        return config_;
    }

    /**
     * Load a specific plugin by name with automatic type casting
     * Searches registered directories and loads the plugin if found
     *
     * @tparam handleType The specific plugin handle type (defaults to basePluginHandle)
     * @param plugin_name Name of the plugin to load
     * @return Typed plugin handle or nullptr if not found or cast fails
     */
    template<typename handleType = basePluginHandle>
    std::shared_ptr<handleType>
    loadPlugin(const std::string &plugin_name) {
        auto base_handle = loadPluginInternal(plugin_name);
        auto typed_handle = std::dynamic_pointer_cast<handleType>(base_handle);
        if (!typed_handle && base_handle) {
            NIXL_ERROR << "Failed to cast plugin '" << plugin_name << "' to requested handle type";
        }

        return typed_handle;
    }

    /**
     * Get an already loaded plugin handle with automatic type casting
     * Returns nullptr if plugin is not loaded
     *
     * @tparam handleType The specific plugin handle type (defaults to basePluginHandle)
     * @param plugin_name Name of the plugin
     * @return Typed plugin handle or nullptr if not loaded or cast fails
     */
    template<typename handleType = basePluginHandle>
    [[nodiscard]] std::shared_ptr<handleType>
    getPlugin(const std::string &plugin_name) const {
        auto base_handle = getPluginInternal(plugin_name);
        return std::dynamic_pointer_cast<handleType>(base_handle);
    }

    /**
     * Load a plugin from a specific file path
     * Returns typed plugin handle or nullptr if not loaded or cast fails
     */
    template<typename handleType = basePluginHandle>
    std::shared_ptr<handleType>
    loadPluginFromPath(const std::filesystem::path &plugin_path, const std::string &plugin_name) {
        auto result = loadPluginFromPathInternal(plugin_path);
        if (std::holds_alternative<success_result<void>>(result)) {
            auto &success = std::get<success_result<void>>(result);
            auto base_handle = createPluginHandle(std::move(success.handle), success.interface);
            loadedPlugins_.emplace(plugin_name, base_handle);
            onPluginLoaded(plugin_name, success.interface);

            return std::dynamic_pointer_cast<handleType>(base_handle);
        }
        return nullptr;
    }

    /**
     * Unload a plugin by name
     * Does nothing if plugin is not loaded or cannot be unloaded
     */
    void
    unloadPlugin(const std::string &plugin_name);

    /**
     * Get all loaded plugin names
     */
    std::vector<std::string>
    getLoadedPluginNames() const;

protected:
    mutable std::mutex lock_;
    std::map<std::string, std::shared_ptr<basePluginHandle>, std::less<>> loadedPlugins_;
    explicit basePluginManager(pluginConfig config);

    /**
     * Internal non-template implementation of loadPluginFromPath
     */
    plugin_load_result<void>
    loadPluginFromPathInternal(const std::filesystem::path &plugin_path);

    /**
     * Extract plugin name from filename based on configured prefix/suffix
     * Returns empty string if filename doesn't match pattern
     */
    std::string
    extractPluginNameFromFilename(const std::string &filename) const;

    /**
     * Construct full plugin path from directory and plugin name
     */
    std::filesystem::path
    constructPluginPath(const std::filesystem::path &directory,
                        const std::string &plugin_name) const;

    /**
     * Check if API version matches expected version
     * Derived classes override to provide specific version checking logic
     */
    virtual bool
    checkApiVersion(void *plugin_interface) const = 0;

    /**
     * Factory method for creating typed plugin handles
     * Derived classes override to create their specific handle type
     */
    virtual std::shared_ptr<basePluginHandle>
    createPluginHandle(std::unique_ptr<void, dlHandleDeleter> dl_handle,
                       void *plugin_interface) = 0;

    /**
     * Check if a plugin can be unloaded
     * Derived classes can override to prevent unloading (e.g., static plugins)
     */
    virtual bool
    canUnloadPlugin(const std::string &plugin_name) const {
        return true; // Default: allow unload
    }

    /**
     * Called after successful plugin load - derived classes can do additional setup
     */
    virtual void
    onPluginLoaded(const std::string &plugin_name, void *plugin_interface) {
        // Default: do nothing
        (void)plugin_name;
        (void)plugin_interface;
    }

    /**
     * Internal non-template implementation of loadPlugin
     */
    std::shared_ptr<basePluginHandle>
    loadPluginInternal(const std::string &plugin_name);

    /**
     * Internal non-template implementation of getPlugin
     */
    std::shared_ptr<basePluginHandle>
    getPluginInternal(const std::string &plugin_name) const;

private:
    std::vector<std::filesystem::path> pluginDirs_;
    pluginConfig config_;
};

#endif // __BASE_PLUGIN_MANAGER_H
