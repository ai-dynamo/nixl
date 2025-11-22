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
#include <dlfcn.h>
#include <filesystem>
#include <cstdlib>
#include <cassert>

using lock_guard = const std::lock_guard<std::mutex>;

nixlTelemetryPluginHandle::nixlTelemetryPluginHandle(std::unique_ptr<void, dlHandleDeleter> handle,
                                                     nixlTelemetryPlugin *plugin)
    : basePluginHandle(std::move(handle), plugin),
      plugin_(plugin) {
    assert(plugin_ && "Plugin interface must not be null");
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
    const char *plugin_dir = getenv(telemetryExporterPluginDirVar);
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
    return std::filesystem::path(info.dli_fname).parent_path().parent_path() / "telemetry";
}
} // namespace

// Plugin Manager implementation
nixlTelemetryPluginManager::nixlTelemetryPluginManager()
    : basePluginManager(
          pluginConfig{.initFuncName = "nixl_telemetry_plugin_init",
                       .finiFuncName = "nixl_telemetry_plugin_fini",
                       .filenamePrefix = "libtelemetry_exporter_",
                       .filenameSuffix = ".so",
                       .expectedApiVersion = static_cast<int>(nixlTelemetryPluginApiVersionV1)}) {
    std::filesystem::path plugin_dir = getTelemetryPluginDir();
    if (!plugin_dir.empty()) {
        NIXL_DEBUG << "Loading telemetry exporter plugins from: " << plugin_dir;
        addPluginDirectory(plugin_dir);
    }
}

bool
nixlTelemetryPluginManager::checkApiVersion(void *plugin_interface) const {
    if (!plugin_interface) {
        return false;
    }

    nixlTelemetryPlugin *plugin = static_cast<nixlTelemetryPlugin *>(plugin_interface);
    return plugin->api_version == nixlTelemetryPluginApiVersionV1;
}

std::shared_ptr<basePluginHandle>
nixlTelemetryPluginManager::createPluginHandle(std::unique_ptr<void, dlHandleDeleter> dl_handle,
                                               void *plugin_interface) {

    nixlTelemetryPlugin *plugin = static_cast<nixlTelemetryPlugin *>(plugin_interface);
    return std::make_shared<nixlTelemetryPluginHandle>(std::move(dl_handle), plugin);
}

nixlTelemetryPluginManager &
nixlTelemetryPluginManager::getInstance() {
    // Meyers singleton - thread-safe in C++11+
    static nixlTelemetryPluginManager instance;
    return instance;
}

std::unique_ptr<nixlTelemetryExporter>
nixlTelemetryPluginManager::createExporter(std::string_view plugin_name,
                                           const nixlTelemetryExporterInitParams &init_params) {
    auto plugin_handle = getPlugin<nixlTelemetryPluginHandle>(std::string(plugin_name));
    if (!plugin_handle) {
        plugin_handle = loadPlugin<nixlTelemetryPluginHandle>(std::string(plugin_name));
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
