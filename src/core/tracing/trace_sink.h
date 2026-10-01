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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_SINK_H
#define NIXL_SRC_CORE_TRACING_TRACE_SINK_H

#include <atomic>
#include <span>
#include <string>
#include <string_view>

#include "tracing/backend_trace.h"
#include "tracing/trace.h"

class nixlTelemetry;

namespace nixl::trace {

class TracerPhaseSink final : public nixlBackendTraceSink {
public:
    TracerPhaseSink(Tracer &tracer,
                    std::string backend,
                    nixlTelemetry *telemetry = nullptr) noexcept
        : tracer_(tracer),
          backend_(std::move(backend)),
          telemetry_(telemetry) {}

    void
    recordPhase(nixl_trace_phase_t phase,
                std::string_view label,
                nixlTime::us_t timestamp,
                std::span<const nixlBackendTraceAttr> attrs) noexcept override;

private:
    Tracer &tracer_;
    const std::string backend_;
    nixlTelemetry *const telemetry_;
    std::atomic_flag dropWarned_;
};

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_TRACE_SINK_H
