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
#ifndef NIXL_SRC_CORE_TRACING_API_TRACE_CORE_PLUGIN_H
#define NIXL_SRC_CORE_TRACING_API_TRACE_CORE_PLUGIN_H

#include <memory>
#include <string_view>
#include <vector>

#include "tracing/api/trace_backend.h"
#include "tracing/api/trace_plugin.h"
#include "tracing/api/tracer.h"

// V1 covers this descriptor, nixlTraceCoreConfig and every type the core shares
// with libnixl, data plugins and trace backends: the Tracer and
// nixlBackendTraceSink interfaces, Span (returned by value), TraceBackend and
// SpanBackend, and the values they pass (TraceContext, TraceIdentity, Name,
// Attribute and the record list). Bump it when any of them changes. A change to
// TraceBackend or SpanBackend also bumps nixl_trace_plugin_api_version, and a
// change to nixlBackendTraceSink also bumps NIXL_PLUGIN_API_VERSION.
enum class nixl_trace_core_api_version : unsigned int {
    V1 = 1,
};

struct nixlTraceCoreConfig {
    std::string_view agentName;
    std::vector<std::unique_ptr<nixl::trace::TraceBackend>> backends;
    double sampleRatio;
};

using trace_core_tracer_creator_fn_t =
    std::unique_ptr<nixl::trace::Tracer> (*)(nixlTraceCoreConfig config);

struct nixlTraceCorePlugin {
    nixl_trace_core_api_version api_version;
    trace_core_tracer_creator_fn_t create_tracer;
};

extern "C" {
NIXL_TRACE_PLUGIN_EXPORT nixlTraceCorePlugin *
nixl_trace_core_plugin_init();

NIXL_TRACE_PLUGIN_EXPORT void
nixl_trace_core_plugin_fini();
}

#endif
