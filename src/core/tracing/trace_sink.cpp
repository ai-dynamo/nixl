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
#include "tracing/trace_sink.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/nixl_log.h"
#include "telemetry.h"
#include "tracing/trace_context.h"

namespace nixl::trace {
namespace {

    constexpr std::size_t max_core_attrs = 4;

    [[nodiscard]] constexpr Kind
    phaseKind(nixl_trace_phase_t phase) noexcept {
        switch (phase) {
        case nixl_trace_phase_t::NOTIF_SENT:
            return Kind::CommSend;
        case nixl_trace_phase_t::NOTIF_RECEIVED:
            return Kind::CommRecv;
        case nixl_trace_phase_t::SUBMIT:
        case nixl_trace_phase_t::WIRE_SUBMITTED:
        case nixl_trace_phase_t::WIRE_COMPLETED:
        case nixl_trace_phase_t::REMOTE_OBSERVED:
        case nixl_trace_phase_t::OTHER:
            return Kind::Metadata;
        }
        return Kind::Metadata;
    }

} // namespace

void
TracerPhaseSink::recordPhase(nixl_trace_phase_t phase,
                             std::string_view label,
                             nixlTime::us_t timestamp,
                             std::span<const nixlBackendTraceAttr> attrs,
                             const TraceContext *context) noexcept {
    try {
        const TraceContext event_context = (context != nullptr) ? *context : TraceContext{};
        std::string traceparent;
        std::vector<Attribute> event_attrs;
        event_attrs.reserve(max_core_attrs + attrs.size());
        event_attrs.push_back({"nixl.backend", backend_});
        event_attrs.push_back({"nixl.phase.timestamp_us", static_cast<std::int64_t>(timestamp)});
        if (event_context.valid()) {
            traceparent = formatTraceparent(event_context);
            event_attrs.push_back({"nixl.traceparent", traceparent});
        }
        if (!label.empty()) {
            event_attrs.push_back({"nixl.phase.label", label});
        }
        for (const auto &attr : attrs) {
            event_attrs.push_back({attr.key, attr.value});
        }

        const bool named = (phase == nixl_trace_phase_t::OTHER) && !label.empty();
        tracer_.emitEvent(named ? label : toStringView(phase),
                          phaseKind(phase),
                          event_context,
                          timestamp,
                          event_attrs);
    }
    catch (...) {
        if (telemetry_ != nullptr) {
            telemetry_->updateTracePhasesDropped(1);
        }
        if (!dropWarned_.test_and_set(std::memory_order_relaxed)) {
            try {
                NIXL_ERROR << "Dropping a trace phase for backend " << backend_
                           << ": recording threw. Further drops from this sink are not logged.";
            }
            catch (...) {
            }
        }
    }
}

} // namespace nixl::trace
