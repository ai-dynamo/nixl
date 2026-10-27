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
#include <optional>
#include <string>
#include <vector>

#include "common/nixl_log.h"
#include "telemetry.h"
#include "tracing/trace_context.h"

namespace nixl::trace {
namespace {

    constexpr std::size_t max_core_attrs = 6;

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

    [[nodiscard]] constexpr Kind
    operationKind(nixl_xfer_op_t operation) noexcept {
        switch (operation) {
        case NIXL_WRITE:
            return Kind::CommSend;
        case NIXL_READ:
            return Kind::CommRecv;
        }
        return Kind::Generic;
    }

} // namespace

template<typename Emit>
void
TracerPhaseSink::record(std::string_view label_key,
                        std::string_view label,
                        std::optional<nixlTime::us_t> timestamp,
                        std::span<const nixlBackendTraceAttr> attrs,
                        const TraceContext *context,
                        std::uint64_t request_id,
                        std::uint64_t execution,
                        const Emit &emit) noexcept {
    try {
        const TraceContext record_context = (context != nullptr) ? *context : TraceContext{};
        std::string traceparent;
        std::string request_id_text;
        std::vector<Attribute> record_attrs;
        record_attrs.reserve(max_core_attrs + attrs.size());
        record_attrs.push_back({"nixl.backend", backend_});
        if (timestamp.has_value()) {
            record_attrs.push_back(
                {"nixl.phase.timestamp_us", static_cast<std::int64_t>(*timestamp)});
        }
        if (record_context.valid()) {
            traceparent = formatTraceparent(record_context);
            record_attrs.push_back({"nixl.traceparent", traceparent});
        }
        if (request_id != 0) {
            request_id_text = std::to_string(request_id);
            record_attrs.push_back({"nixl.xfer_request_id", request_id_text});
        }
        if (execution != 0) {
            record_attrs.push_back({"nixl.xfer_execution", static_cast<std::int64_t>(execution)});
        }
        if (!label.empty()) {
            record_attrs.push_back({label_key, label});
        }
        for (const auto &attr : attrs) {
            record_attrs.push_back({attr.key, attr.value});
        }

        emit(record_context, record_attrs);
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

void
TracerPhaseSink::recordPhase(nixl_trace_phase_t phase,
                             std::string_view label,
                             nixlTime::us_t timestamp,
                             std::span<const nixlBackendTraceAttr> attrs,
                             const TraceContext *context,
                             std::uint64_t request_id,
                             std::uint64_t execution) noexcept {
    const bool named = (phase == nixl_trace_phase_t::OTHER) && !label.empty();
    record("nixl.phase.label",
           label,
           timestamp,
           attrs,
           context,
           request_id,
           execution,
           [&](const TraceContext &event_context, std::span<const Attribute> event_attrs) {
               tracer_.emitEvent(named ? label : toStringView(phase),
                                 phaseKind(phase),
                                 event_context,
                                 request_id,
                                 timestamp,
                                 event_attrs);
           });
}

void
TracerPhaseSink::recordSpan(nixl_trace_span_t span,
                            nixl_xfer_op_t operation,
                            std::string_view label,
                            nixlTime::us_t start,
                            nixlTime::us_t end,
                            std::span<const nixlBackendTraceAttr> attrs,
                            const TraceContext *context,
                            std::uint64_t request_id,
                            std::uint64_t execution) noexcept {
    const bool named = (span == nixl_trace_span_t::OTHER) && !label.empty();
    record("nixl.span.label",
           label,
           std::nullopt,
           attrs,
           context,
           request_id,
           execution,
           [&](const TraceContext &span_context, std::span<const Attribute> span_attrs) {
               tracer_.emitSpan(named ? label : toStringView(span),
                                operationKind(operation),
                                span_context,
                                request_id,
                                start,
                                end,
                                span_attrs);
           });
}

} // namespace nixl::trace
