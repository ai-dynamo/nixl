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
#ifndef NIXL_SRC_CORE_TRACING_API_BACKEND_TRACE_SINKS_H
#define NIXL_SRC_CORE_TRACING_API_BACKEND_TRACE_SINKS_H

#include <atomic>
#include <unordered_map>

#include "nixl_types.h"
#include "tracing/api/tracing.h"

class nixlBackendEngine;

namespace nixl::trace {

#if defined(NIXL_TRACE_ENABLED)

class BackendTraceSinks {
public:
    [[nodiscard]] static BackendTraceSinkPtr
    makeSink(Tracer *tracer, const nixl_backend_t &type, nixlTelemetry *telemetry);

    void
    adopt(const nixl_backend_t &type, BackendTraceSinkPtr sink, const nixlBackendEngine &backend);

    void
    warnIfContextUnsupported(const nixlBackendEngine &backend, const Traceable &traceable) noexcept;

private:
    std::unordered_map<nixl_backend_t, BackendTraceSinkPtr> sinks_;
    std::unordered_map<nixl_backend_t, std::atomic<bool>> contextUnsupported_;
};

#else

class BackendTraceSinks {
public:
    [[nodiscard]] static BackendTraceSinkPtr
    makeSink(Tracer *, const nixl_backend_t &, nixlTelemetry *) noexcept {
        return {};
    }

    void
    adopt(const nixl_backend_t &, BackendTraceSinkPtr, const nixlBackendEngine &) noexcept {}

    void
    warnIfContextUnsupported(const nixlBackendEngine &, const Traceable &) noexcept {}
};

#endif

} // namespace nixl::trace

#endif
