/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_PLUGIN_H
#define NIXL_SRC_CORE_TRACING_TRACE_PLUGIN_H

/*
 * Trace-backend plugin interface (internal core API).
 *
 * A tracing backend (NVTX, Chakra, ...) is shipped as a separately built, on-demand
 * loaded `.so` plugin, exactly like NIXL's data backends and telemetry exporters. This
 * header is the contract a trace plugin implements; it is loaded by nixlPluginManager
 * (prefix `libtrace_backend_`, init symbol `nixl_trace_plugin_init`). The facade types
 * (`nixl::trace::TraceBackend`) live in `trace.h`.
 */

#include "tracing/trace.h"
#include "common/nixl_log.h"

#include <memory>
#include <string>
#include <string_view>

enum class nixl_trace_plugin_api_version : unsigned int {
    V1 = 1,
};

// Parameters passed to a trace backend at creation time.
struct nixlTraceBackendInitParams {
    // Agent name; used e.g. as the NVTX domain name so per-agent timelines stay distinct.
    std::string_view agentName;
};

// Factory function type: builds one backend instance for an agent.
using trace_backend_creator_fn_t =
    std::unique_ptr<nixl::trace::TraceBackend> (*)(const nixlTraceBackendInitParams &init_params);

class nixlTracePlugin {
public:
    nixl_trace_plugin_api_version api_version;
    trace_backend_creator_fn_t create_backend;

    nixlTracePlugin(nixl_trace_plugin_api_version version,
                    std::string_view name,
                    std::string_view ver,
                    trace_backend_creator_fn_t create) noexcept
        : api_version(version),
          create_backend(create),
          name_(name),
          version_(ver) {}

    const std::string &
    getName() const noexcept {
        return name_;
    }

    const std::string &
    getVersion() const noexcept {
        return version_;
    }

private:
    std::string name_;
    std::string version_;
};

// Macro to define exported C functions for the plugin
#define NIXL_TRACE_PLUGIN_EXPORT __attribute__((visibility("default")))

// Template for backends whose constructor takes (const nixlTraceBackendInitParams &),
// giving the minimal-boilerplate path (mirrors nixlTelemetryPluginCreator). Backends that
// keep their type private may instead pass a plain factory function to nixlTracePlugin.
template<typename BackendType> class nixlTracePluginCreator {
public:
    static nixlTracePlugin *
    create(nixl_trace_plugin_api_version api_version,
           std::string_view name,
           std::string_view version) {
        static nixlTracePlugin plugin_instance(api_version, name, version, createBackend);
        return &plugin_instance;
    }

private:
    static std::unique_ptr<nixl::trace::TraceBackend>
    createBackend(const nixlTraceBackendInitParams &init_params) {
        try {
            return std::make_unique<BackendType>(init_params);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Failed to create trace backend: " << e.what();
            return nullptr;
        }
    }
};

// Plugin must implement these for dynamic loading (extern "C" avoids name mangling).
extern "C" {
NIXL_TRACE_PLUGIN_EXPORT nixlTracePlugin *
nixl_trace_plugin_init();

NIXL_TRACE_PLUGIN_EXPORT void
nixl_trace_plugin_fini();
}

#endif // NIXL_SRC_CORE_TRACING_TRACE_PLUGIN_H
