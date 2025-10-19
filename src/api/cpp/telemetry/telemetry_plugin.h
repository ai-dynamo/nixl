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

#ifndef __TELEMETRY_PLUGIN_H
#define __TELEMETRY_PLUGIN_H

#include "telemetry/telemetry_exporter.h"
//  #include "common/nixl_log.h"

// Forward declarations for special engine types
//  class nixlUcxEngine;

// Define the plugin API version
#define NIXL_TELEMETRY_PLUGIN_API_VERSION 1

// Define the plugin interface class
class nixlTelemetryPlugin {
public:
    int api_version;

    // Function pointer for creating a new exporter instance
    nixlTelemetryExporter *(*create_exporter)(const nixlTelemetryExporterInitParams *init_params);

    // Function pointer for destroying an exporter instance
    void (*destroy_exporter)(nixlTelemetryExporter *exporter);

    // Function to get the plugin name
    const char *(*get_plugin_name)();

    // Function to get the plugin version
    const char *(*get_plugin_version)();
};

// Macro to define exported C functions for the plugin
#define NIXL_TELEMETRY_PLUGIN_EXPORT __attribute__((visibility("default")))

// Template for creating backend plugins with minimal boilerplate
template<typename ExporterType> class nixlTelemetryPluginCreator {
public:
    static nixlTelemetryPlugin *
    create(int api_version, const char *name, const char *version) {

        static const char *plugin_name = name;
        static const char *plugin_version = version;

        static nixlTelemetryPlugin plugin_instance = {api_version,
                                                      createExporter,
                                                      destroyExporter,
                                                      []() { return plugin_name; },
                                                      []() { return plugin_version; }};

        return &plugin_instance;
    }

private:
    [[nodiscard]] static nixlTelemetryExporter *
    createExporter(const nixlTelemetryExporterInitParams *init_params) {
        try {
            return new ExporterType(init_params);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to create exporter: " << e.what();
            return nullptr;
        }
    }

    static void
    destroyExporter(nixlTelemetryExporter *exporter) {
        delete exporter;
    }
};

// Plugin must implement these functions for dynamic loading
// Note: extern "C" is required for dynamic loading to avoid C++ name mangling
extern "C" {
// Initialize the plugin
NIXL_TELEMETRY_PLUGIN_EXPORT
nixlTelemetryPlugin *
nixl_telemetry_plugin_init();

// Cleanup the plugin
NIXL_TELEMETRY_PLUGIN_EXPORT
void
nixl_telemetry_plugin_fini();
}

#endif // __TELEMETRY_PLUGIN_H
