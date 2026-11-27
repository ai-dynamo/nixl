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
#ifndef NIXL_SRC_CORE_TRACING_CORE_TRACE_PLUGIN_LOADER_H
#define NIXL_SRC_CORE_TRACING_CORE_TRACE_PLUGIN_LOADER_H

#include <memory>
#include <string>

#include "plugin_manager.h"

class nixlTracePlugin;
struct nixlTraceBackendInitParams;
struct nixlTraceCorePlugin;
struct nixlTraceCoreConfig;

namespace nixl::trace {
class TraceBackend;
class Tracer;
} // namespace nixl::trace

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

class nixlTraceCorePluginHandle : public nixlPluginHandle {
public:
    nixlTraceCorePluginHandle(void *handle, nixlTraceCorePlugin *plugin);
    ~nixlTraceCorePluginHandle();

    std::unique_ptr<nixl::trace::Tracer>
    createTracer(nixlTraceCoreConfig config) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    nixlTraceCorePlugin *plugin_;
};

namespace nixl::trace {

// Load a specific trace-backend plugin
[[nodiscard]] std::shared_ptr<const nixlTracePluginHandle>
loadTracePlugin(const std::string &plugin_name);

[[nodiscard]] std::shared_ptr<const nixlTraceCorePluginHandle>
loadTraceCorePlugin(const std::string &plugin_name = "core");

} // namespace nixl::trace

#endif
