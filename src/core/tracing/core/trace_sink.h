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
#ifndef NIXL_SRC_CORE_TRACING_CORE_TRACE_SINK_H
#define NIXL_SRC_CORE_TRACING_CORE_TRACE_SINK_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "tracing/api/backend_trace.h"
#include "tracing/core/trace.h"

namespace nixl::trace {

class TracerPhaseSink final : public nixlBackendTraceSink {
public:
    TracerPhaseSink(CompositeTracer &tracer,
                    std::string backend,
                    std::function<void(std::uint64_t)> on_dropped = {}) noexcept
        : tracer_(tracer),
          backend_(std::move(backend)),
          onDropped_(std::move(on_dropped)) {}

    using nixlBackendTraceSink::recordPhase;

    void
    recordPhase(nixl_trace_phase_t phase,
                nixlTime::us_t timestamp,
                const TraceIdentity &identity,
                std::span<const nixlBackendTraceAttr> attrs,
                std::string_view label) noexcept override;

    void
    recordSpan(nixl_trace_span_t span,
               nixl_xfer_op_t operation,
               nixlTime::us_t start,
               nixlTime::us_t end,
               const TraceIdentity &identity,
               std::span<const nixlBackendTraceAttr> attrs,
               std::string_view label) noexcept override;

    [[nodiscard]] bool
    needsTimestamps() const noexcept override {
        return tracer_.needsTimestamps();
    }

    [[nodiscard]] const TraceIdentity *
    currentIdentity() const noexcept override;

    void
    writeTraceTags(nixlSerDes &ser_des, const TraceIdentity &identity) const override;

    [[nodiscard]] std::optional<TraceIdentity>
    readTraceTags(nixlSerDes &ser_des) const override;

    void
    deferPhase(const DeferredPhase &deferred) noexcept override;

    void
    recordDeferredPhases() noexcept override;

private:
    template<typename Emit>
    void
    record(AttributeKey label_key,
           std::string_view label,
           std::optional<nixlTime::us_t> timestamp,
           const TraceIdentity &identity,
           std::span<const nixlBackendTraceAttr> attrs,
           const Emit &emit) noexcept;

    CompositeTracer &tracer_;
    const std::string backend_;
    const std::function<void(std::uint64_t)> onDropped_;
    std::atomic_flag dropWarned_;
};

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_CORE_TRACE_SINK_H
