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

#include "nvtx_backend.h"
#include "tracing/trace_plugin.h"

// Loaded by nixlPluginManager (prefix `libtrace_backend_`). The factory itself
// (createNvtxBackend) builds one NVTX domain per agent on demand.
extern "C" NIXL_TRACE_PLUGIN_EXPORT nixlTracePlugin *
nixl_trace_plugin_init() {
    static nixlTracePlugin plugin(nixl_trace_plugin_api_version::V1,
                                  "nvtx",
                                  "0.1.0",
                                  &nixl::trace::createNvtxBackend);
    return &plugin;
}

extern "C" NIXL_TRACE_PLUGIN_EXPORT void
nixl_trace_plugin_fini() {}
