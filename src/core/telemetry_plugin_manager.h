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
#ifndef _TELEMETRY_PLUGIN_MANAGER_H
#define _TELEMETRY_PLUGIN_MANAGER_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include "telemetry/telemetry_plugin.h"

/**
 * This class represents a telemetry exporter plugin handle used to create exporter instances.
 * Plugin handle attributes are modified only in the constructor and destructor and remain
 * unchanged during normal operation. This allows using it in multi-threading environments
 * without lock protection.
 */
class nixlTelemetryPluginHandle {
private:
    void *handle_; // Handle to the dynamically loaded library
    nixlTelemetryPlugin *plugin_; // Plugin interface

public:
    nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin);
    ~nixlTelemetryPluginHandle();

    nixlTelemetryExporter *
    createExporter(const nixlTelemetryExporterInitParams *init_params) const;
    void
    destroyExporter(nixlTelemetryExporter *exporter) const;
    const char *
    getName() const;
    const char *
    getVersion() const;
};

/**
 * Telemetry Exporter Plugin Manager
 *
 * Manages dynamic loading of telemetry exporter plugins. Unlike the backend plugin manager,
 * this only supports dynamic plugins (no static plugins) for simplicity.
 */
class nixlTelemetryPluginManager {
private:
    std::map<std::string, std::shared_ptr<const nixlTelemetryPluginHandle>> loaded_plugins_;
    std::vector<std::string> plugin_dirs_;
    std::mutex lock_;

    // Private constructor for singleton pattern
    nixlTelemetryPluginManager();

    std::shared_ptr<const nixlTelemetryPluginHandle>
    loadPluginFromPath(const std::string &plugin_path);

public:
    // Singleton instance accessor
    static nixlTelemetryPluginManager &
    getInstance();

    // Delete copy constructor and assignment operator
    nixlTelemetryPluginManager(const nixlTelemetryPluginManager &) = delete;
    nixlTelemetryPluginManager &
    operator=(const nixlTelemetryPluginManager &) = delete;

    /**
     * Load a specific exporter plugin by name
     * Searches in registered plugin directories for libtelemetry_exporter_<name>.so
     *
     * @param plugin_name Name of the plugin (e.g., "file", "prometheus", "otlp")
     * @return Plugin handle or nullptr if not found
     */
    std::shared_ptr<const nixlTelemetryPluginHandle>
    loadPlugin(const std::string &plugin_name);

    /**
     * Get an already loaded plugin handle
     *
     * @param plugin_name Name of the plugin
     * @return Plugin handle or nullptr if not loaded
     */
    std::shared_ptr<const nixlTelemetryPluginHandle>
    getPlugin(const std::string &plugin_name);

    /**
     * Create an exporter instance from a plugin
     *
     * @param plugin_name Name of the plugin to use
     * @param init_params Initialization parameters for the exporter
     * @return Unique pointer to created exporter or nullptr on failure
     */
    std::unique_ptr<nixlTelemetryExporter>
    createExporter(const std::string &plugin_name,
                   const nixlTelemetryExporterInitParams *init_params);

    /**
     * Discover and load all plugins from a directory
     * Looks for files matching: libtelemetry_exporter_*.so
     *
     * @param dirpath Directory path to search
     */
    void
    discoverPluginsFromDir(const std::string &dirpath);

    /**
     * Add a directory to search for plugins
     * New directory is prioritized over existing ones
     *
     * @param directory Path to plugin directory
     */
    void
    addPluginDirectory(const std::string &directory);

    /**
     * Unload a plugin
     *
     * @param plugin_name Name of the plugin to unload
     */
    void
    unloadPlugin(const std::string &plugin_name);

    /**
     * Get all loaded plugin names
     *
     * @return Vector of loaded plugin names
     */
    std::vector<std::string>
    getLoadedPluginNames();
};

#endif // _TELEMETRY_PLUGIN_MANAGER_H
