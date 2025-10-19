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
#include <string_view>
#include <map>
#include <memory>
#include <vector>
#include <mutex>
#include <filesystem>
#include "telemetry/telemetry_plugin.h"
#include "base_plugin_manager.h"

/**
 * This class represents a telemetry exporter plugin handle used to create exporter instances.
 * Plugin handle attributes are modified only in the constructor and destructor and remain
 * unchanged during normal operation. This allows using it in multi-threading environments
 * without lock protection.
 */
class nixlTelemetryPluginHandle : public basePluginHandle {
private:
    nixlTelemetryPlugin *plugin_; // Plugin interface (cached for type safety)

public:
    nixlTelemetryPluginHandle(std::unique_ptr<void, dlHandleDeleter> handle,
                              nixlTelemetryPlugin *plugin);
    ~nixlTelemetryPluginHandle() override = default;

    [[nodiscard]] std::unique_ptr<nixlTelemetryExporter>
    createExporter(const nixlTelemetryExporterInitParams &init_params) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;
};

/**
 * Telemetry Exporter Plugin Manager
 *
 * Manages dynamic loading of telemetry exporter plugins. Unlike the backend plugin manager,
 * this only supports dynamic plugins (no static plugins) for simplicity.
 */
class nixlTelemetryPluginManager : public basePluginManager {
public:
    // Singleton instance accessor
    [[nodiscard]] static nixlTelemetryPluginManager &
    getInstance();

    // Delete copy constructor and assignment operator
    nixlTelemetryPluginManager(const nixlTelemetryPluginManager &) = delete;
    nixlTelemetryPluginManager &
    operator=(const nixlTelemetryPluginManager &) = delete;

    /**
     * Create an exporter instance from a plugin
     *
     * @param plugin_name Name of the plugin to use
     * @param init_params Initialization parameters for the exporter
     * @return Unique pointer to created exporter or nullptr on failure
     */
    [[nodiscard]] std::unique_ptr<nixlTelemetryExporter>
    createExporter(std::string_view plugin_name,
                   const nixlTelemetryExporterInitParams &init_params);

protected:
    bool
    checkApiVersion(void *plugin_interface) const override;

    std::shared_ptr<basePluginHandle>
    createPluginHandle(std::unique_ptr<void, dlHandleDeleter> dl_handle,
                       void *plugin_interface) override;

private:
    // Private constructor for singleton pattern
    nixlTelemetryPluginManager();
};

#endif // _TELEMETRY_PLUGIN_MANAGER_H
