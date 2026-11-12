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

#include "tracing/core/trace_plugin_loader.h"

#include <dlfcn.h>
#include <utility>

#include "common/nixl_log.h"
#include "tracing/api/trace_core_plugin.h"
#include "tracing/api/trace_plugin.h"

const std::string tracePluginPrefix = "libtrace_backend_";
const std::string traceCorePluginPrefix = "libtrace_";

nixlTracePluginHandle::nixlTracePluginHandle(void *handle, nixlTracePlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlTracePluginHandle::~nixlTracePluginHandle() {
    if (handle_) {
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_trace_plugin_fini");
        if (fini) {
            fini();
        }
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

std::unique_ptr<nixl::trace::TraceBackend>
nixlTracePluginHandle::createBackend(const nixlTraceBackendInitParams &init_params) const {
    if (plugin_ && plugin_->create_backend) {
        return plugin_->create_backend(init_params);
    }
    return nullptr;
}

const char *
nixlTracePluginHandle::getName() const {
    if (plugin_) {
        return plugin_->getName().c_str();
    }
    return "unknown";
}

const char *
nixlTracePluginHandle::getVersion() const {
    if (plugin_) {
        return plugin_->getVersion().c_str();
    }
    return "unknown";
}

namespace {
// Trace plugin loader
std::shared_ptr<const nixlPluginHandle>
traceLoader(void *handle, const std::string &plugin_path) {
    typedef nixlTracePlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_trace_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_trace_plugin_init in " << plugin_path << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    nixlTracePlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    if (plugin->api_version != nixl_trace_plugin_api_version::V2) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path << ": expected "
                   << static_cast<unsigned int>(nixl_trace_plugin_api_version::V2) << ", got "
                   << static_cast<unsigned int>(plugin->api_version);
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlTracePluginHandle>(handle, plugin);
}
} // namespace

nixlTraceCorePluginHandle::nixlTraceCorePluginHandle(void *handle, nixlTraceCorePlugin *plugin)
    : nixlPluginHandle(handle),
      plugin_(plugin) {}

nixlTraceCorePluginHandle::~nixlTraceCorePluginHandle() {
    if (handle_) {
        typedef void (*fini_func_t)();
        fini_func_t fini = (fini_func_t)dlsym(handle_, "nixl_trace_core_plugin_fini");
        if (fini) {
            fini();
        }
        dlclose(handle_);
        handle_ = nullptr;
        plugin_ = nullptr;
    }
}

std::unique_ptr<nixl::trace::Tracer>
nixlTraceCorePluginHandle::createTracer(nixlTraceCoreConfig config) const {
    if (plugin_ && plugin_->create_tracer) {
        return plugin_->create_tracer(std::move(config));
    }
    return nullptr;
}

const char *
nixlTraceCorePluginHandle::getName() const {
    return "core";
}

const char *
nixlTraceCorePluginHandle::getVersion() const {
    return "1";
}

namespace {
std::shared_ptr<const nixlPluginHandle>
traceCoreLoader(void *handle, const std::string &plugin_path) {
    typedef nixlTraceCorePlugin *(*init_func_t)();
    init_func_t init = (init_func_t)dlsym(handle, "nixl_trace_core_plugin_init");
    if (!init) {
        NIXL_ERROR << "Failed to find nixl_trace_core_plugin_init in " << plugin_path << ": "
                   << dlerror();
        dlclose(handle);
        return nullptr;
    }

    nixlTraceCorePlugin *plugin = init();
    if (!plugin) {
        NIXL_ERROR << "Plugin initialization failed for " << plugin_path;
        dlclose(handle);
        return nullptr;
    }

    if (plugin->api_version != nixl_trace_core_api_version::V1) {
        NIXL_ERROR << "Plugin API version mismatch for " << plugin_path << ": expected "
                   << static_cast<unsigned int>(nixl_trace_core_api_version::V1) << ", got "
                   << static_cast<unsigned int>(plugin->api_version);
        dlclose(handle);
        return nullptr;
    }

    return std::make_shared<const nixlTraceCorePluginHandle>(handle, plugin);
}
} // namespace

namespace nixl::trace {

std::shared_ptr<const nixlTracePluginHandle>
loadTracePlugin(const std::string &plugin_name) {
    auto trace_plugin = std::dynamic_pointer_cast<const nixlTracePluginHandle>(
        nixlPluginManager::getInstance().loadPlugin(tracePluginPrefix, plugin_name, traceLoader));
    if (!trace_plugin) {
        NIXL_INFO << "Failed to load trace plugin '" << plugin_name << "' from any directory";
    }
    return trace_plugin;
}

std::shared_ptr<const nixlTraceCorePluginHandle>
loadTraceCorePlugin(const std::string &plugin_name) {
    auto core_plugin = std::dynamic_pointer_cast<const nixlTraceCorePluginHandle>(
        nixlPluginManager::getInstance().loadPlugin(
            traceCorePluginPrefix, plugin_name, traceCoreLoader));
    if (!core_plugin) {
        NIXL_INFO << "Failed to load trace core plugin '" << plugin_name << "' from any directory";
    }
    return core_plugin;
}

} // namespace nixl::trace
