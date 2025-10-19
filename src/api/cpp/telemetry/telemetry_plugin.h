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

#ifndef _NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_PLUGIN_H
#define _NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_PLUGIN_H

#include "telemetry/telemetry_exporter.h"
#include <string_view>
#include <memory>

enum class NixlTelemetryPluginApiVersion : unsigned int { V1 = 1 };

inline constexpr NixlTelemetryPluginApiVersion NIXL_TELEMETRY_PLUGIN_API_VERSION =
    NixlTelemetryPluginApiVersion::V1;

// Type alias for exporter creation function
using ExporterCreatorFn =
    std::unique_ptr<nixlTelemetryExporter> (*)(const nixlTelemetryExporterInitParams &init_params);

class nixlTelemetryPlugin {
private:
    std::string_view name_;
    std::string_view version_;

public:
    NixlTelemetryPluginApiVersion api_version;
    ExporterCreatorFn create_exporter;

    nixlTelemetryPlugin(NixlTelemetryPluginApiVersion version,
                        std::string_view name,
                        std::string_view ver,
                        ExporterCreatorFn create) noexcept
        : name_(name),
          version_(ver),
          api_version(version),
          create_exporter(create) {}

    [[nodiscard]] std::string_view
    getName() const noexcept {
        return name_;
    }

    [[nodiscard]] std::string_view
    getVersion() const noexcept {
        return version_;
    }
};

// Macro to define exported C functions for the plugin
#define NIXL_TELEMETRY_PLUGIN_EXPORT __attribute__((visibility("default")))

// Template for creating backend plugins with minimal boilerplate
template<typename ExporterType> class nixlTelemetryPluginCreator {
public:
    static nixlTelemetryPlugin *
    create(NixlTelemetryPluginApiVersion api_version,
           std::string_view name,
           std::string_view version) {
        static nixlTelemetryPlugin plugin_instance(api_version, name, version, createExporter);

        return &plugin_instance;
    }

private:
    static std::unique_ptr<nixlTelemetryExporter>
    createExporter(const nixlTelemetryExporterInitParams &init_params) {
        try {
            return std::make_unique<ExporterType>(init_params);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to create exporter: " << e.what();
            return nullptr;
        }
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

#endif // _NIXL_SRC_API_CPP_TELEMETRY_TELEMETRY_PLUGIN_H
