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
#ifndef NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_BACKEND_H
#define NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_BACKEND_H

#include <memory>

#include "tracing/trace_plugin.h"

namespace nixl::trace {

/**
 * @brief Create the NVTX trace backend. Header-only nvtx3 under the hood; ranges
 *        are cheap no-op stubs until a profiler (Nsight Systems) is attached via
 *        NVTX_INJECTION64_PATH. Each agent gets its own NVTX domain named after
 *        @p init_params.agentName. This is the factory the plugin hands to the
 *        core via nixl_trace_plugin_init().
 */
[[nodiscard]] std::unique_ptr<TraceBackend>
createNvtxBackend(const nixlTraceBackendInitParams &init_params);

} // namespace nixl::trace

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_BACKEND_H
