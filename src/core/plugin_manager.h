/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_CORE_PLUGIN_MANAGER_H
#define NIXL_SRC_CORE_PLUGIN_MANAGER_H

#include <filesystem>
#include <string>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <vector>
#include <mutex>

#include "backend/backend_plugin.h"
#include "nixl_types.h"
#include "telemetry/telemetry_plugin.h"
#include "tracing/trace_plugin.h"

// Forward declarations
class nixlBackendEngine;
struct nixlBackendInitParams;

class nixlPluginHandle {
public:
    nixlPluginHandle(void *handle) : handle_(handle) {}

    virtual ~nixlPluginHandle() = default;

    virtual const char *
    getName() const = 0;
    virtual const char *
    getVersion() const = 0;

protected:
    void *handle_; // Handle to the dynamically loaded library
};

/**
 * This class represents a NIXL plugin and is used to create plugin instances. nixlPluginHandle
 * attributes are modified only in the constructor and destructor and remain unchanged during normal
 * operation, e.g., query operations and plugin instance creation. This allows using it in
 * multi-threading environments without lock protection.
 */
class nixlBackendPluginHandle : public nixlPluginHandle {
public:
    nixlBackendPluginHandle(void *handle, nixlBackendPlugin *plugin);
    ~nixlBackendPluginHandle();

    nixlBackendEngine* createEngine(const nixlBackendInitParams* init_params) const;
    void destroyEngine(nixlBackendEngine* engine) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;
    nixl_b_params_t getBackendOptions() const;
    nixl_mem_list_t getBackendMems() const;

private:
    nixlBackendPlugin *plugin_;
};

// Structure to hold static plugin info
struct nixlBackendStaticPluginInfo {
    std::string name;
    nixlStaticPluginCreatorFunc createFunc;
};

struct nixlTelemetryStaticPluginInfo {
    std::string name;
    nixlTelemetryStaticPluginCreatorFunc createFunc;
};

class nixlTelemetryPluginHandle : public nixlPluginHandle {
public:
    nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin);
    ~nixlTelemetryPluginHandle();

    std::unique_ptr<nixlTelemetryExporter>
    createExporter(const nixlTelemetryExporterInitParams &init_params) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    nixlTelemetryPlugin *plugin_;
};

class nixlTracePluginHandle : public nixlPluginHandle {
public:
    nixlTracePluginHandle(void *handle, nixlTracePlugin *plugin);
    ~nixlTracePluginHandle();

    std::unique_ptr<nixl::trace::TraceBackend>
    createBackend(const nixlTraceBackendInitParams &init_params) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    nixlTracePlugin *plugin_;
};

typedef std::shared_ptr<const nixlPluginHandle> (
    *nixlPluginLoaderFunc)(void *handle, const std::string &plugin_path);

class nixlPluginManager {
public:
    // Singleton instance accessor
    static nixlPluginManager& getInstance();

    // Delete copy constructor and assignment operator
    nixlPluginManager(const nixlPluginManager&) = delete;
    nixlPluginManager& operator=(const nixlPluginManager&) = delete;

    void
    discoverPluginsFromList(const std::string &filename);

    // Load a specific backend plugin. When canonical_name is not null it
    // receives the cache key the plugin was resolved to (the user-provided
    // spelling with whitespace trimmed and matched case-insensitively).
    std::shared_ptr<const nixlBackendPluginHandle>
    loadBackendPlugin(const nixl_backend_t &plugin_name, nixl_backend_t *canonical_name = nullptr);

    // Resolve a user-provided backend name to the canonical plugin name:
    // surrounding whitespace is trimmed and the result is matched
    // case-insensitively against every known plugin name. Falls back to the
    // trimmed input when no plugin matches.
    [[nodiscard]] nixl_backend_t
    canonicalBackendName(const nixl_backend_t &name) const;

    // Load a specific telemetry plugin
    std::shared_ptr<const nixlTelemetryPluginHandle>
    loadTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name);

    // Load a specific trace-backend plugin
    std::shared_ptr<const nixlTracePluginHandle>
    loadTracePlugin(const std::string &plugin_name);

    // Unload backend plugin
    void
    unloadBackendPluginForUnitTest(const nixl_backend_t &plugin_name);

    [[nodiscard]] nixl_status_t
    getBackendParams(const nixl_backend_t &type,
                     nixl_mem_list_t &mems,
                     nixl_b_params_t &params) const;

    // Get a backend plugin handle
    std::shared_ptr<const nixlBackendPluginHandle>
    getBackendPlugin(const nixl_backend_t &plugin_name);

    // Get a telemetry plugin handle
    std::shared_ptr<const nixlTelemetryPluginHandle>
    getTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name);

    // Get all loaded backend plugin names
    std::vector<nixl_backend_t>
    getLoadedBackendPluginNames();

    // Get all available backend plugin names (loaded + discovered on disk)
    std::vector<nixl_backend_t>
    getAvailBackendPluginNames();

    // Get all loaded telemetry plugin names
    std::vector<nixl_telemetry_plugin_t>
    getLoadedTelemetryPluginNames();

    // Add a plugin directory
    void
    addPluginDirectory(const std::string &directory);

    // Get a copy of the registered plugin directories
    [[nodiscard]] std::vector<std::string>
    getPluginDirectories() const;

    // Remove a plugin directory again (unit tests only)
    void
    removePluginDirectoryForUnitTest(const std::string &directory);

    // Report a directory entry as a scan carrying the given registration
    // generation would (unit tests only): the entry is discarded unless the
    // directory's current generation still matches.
    void
    discoverBackendPluginWithGenerationForUnitTest(const std::filesystem::path &path,
                                                   const std::string &dir,
                                                   uint64_t generation);

    // Current registration generation of a plugin directory; zero when the
    // directory is not registered (unit tests only).
    [[nodiscard]] uint64_t
    pluginDirectoryScanGenerationForUnitTest(const std::string &dir) const;

    // Forget a discovered, not-yet-loaded plugin name entirely (unit tests only)
    void
    forgetDiscoveredPluginForUnitTest(const nixl_backend_t &name);

    // Static Plugin Helpers
    const std::vector<nixlBackendStaticPluginInfo> &
    getBackendStaticPlugins();

    const std::vector<nixlTelemetryStaticPluginInfo> &
    getTelemetryStaticPlugins();

private:
    // canonicalBackendName() without locking; callers must hold mutex_.
    [[nodiscard]] nixl_backend_t
    canonicalBackendNameLocked(const nixl_backend_t &name) const;

    std::map<nixl_backend_t, std::shared_ptr<const nixlBackendPluginHandle>>
        loaded_backend_plugins_;
    std::map<nixl_telemetry_plugin_t, std::shared_ptr<const nixlTelemetryPluginHandle>>
        loaded_telemetry_plugins_;
    std::map<std::string, std::shared_ptr<const nixlTracePluginHandle>> loaded_trace_plugins_;

    // Per-plugin discovery bookkeeping: whether the pluginlist file names it
    // and which registered directories provide it. A name stays available as
    // long as it has at least one source; sources are tracked even for
    // plugins that are already loaded, so they survive an unload.
    struct DiscoverySources {
        bool from_plugin_list = false;
        std::set<std::string> dirs;
    };

    std::map<nixl_backend_t, DiscoverySources> discovered_backend_plugins_;
    // Explicit paths from the plugin list file (name -> .so path)
    std::map<nixl_backend_t, std::string> explicit_plugin_paths_;
    // Real paths of plugins discovered behind non-canonical file names,
    // keyed by the directory they were found in (canonical name -> directory
    // -> actual .so path). The loader walks them in plugin_dirs_ order, so
    // priority follows directory registration, never scan completion order;
    // consulted only after the standard file names so pluginlist entries and
    // canonical files win.
    std::map<nixl_backend_t, std::map<std::string, std::string>> dir_override_paths_;

    // Registration generation per directory string. Every registration
    // allocates a fresh number from next_scan_generation_; a scan carries the
    // number it started under and discards its results when the directory's
    // current number differs (deregistered, or deregistered and re-registered).
    // Zero is never a valid registration number.
    std::map<std::string, uint64_t> dir_scan_generations_;
    uint64_t next_scan_generation_ = 1;
    std::vector<std::string> plugin_dirs_;
    std::vector<nixlBackendStaticPluginInfo> backend_static_plugins_;
    std::vector<nixlTelemetryStaticPluginInfo> telemetry_static_plugins_;
    mutable std::mutex mutex_;

    [[nodiscard]] std::shared_ptr<const nixlBackendPluginHandle>
    loadBackendPluginImpl(const nixl_backend_t &plugin_name) const;

    void
    registerBuiltinPlugins();
    void
    registerBackendStaticPlugin(const std::string &name, nixlStaticPluginCreatorFunc creator);
    void
    registerTelemetryStaticPlugin(const std::string &name,
                                  nixlTelemetryStaticPluginCreatorFunc creator);

    // Search a directory for plugins
    void
    discoverPluginsFromDir(const std::string &dir, uint64_t generation);

    // Discover helper functions
    void
    discoverBackendPlugin(const std::filesystem::path &path,
                          const std::string &dir_key,
                          uint64_t generation);

    void
    discoverTelemetryPlugin(const std::string &filename);

    void
    discoverTracePlugin(const std::string &filename);

    [[nodiscard]] static std::shared_ptr<const nixlPluginHandle>
    loadPluginFromPath(const std::string &plugin_path,
                       nixlPluginLoaderFunc loader,
                       bool deepbind = false);

    [[nodiscard]] static std::string
    composePluginPath(const std::string &dir,
                      const std::string &plugin_prefix,
                      const std::string &plugin_name);

    // Private constructor for singleton pattern
    nixlPluginManager();
};

#endif
