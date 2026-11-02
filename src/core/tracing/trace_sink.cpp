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

#include "absl/container/inlined_vector.h"
#include "common/nixl_log.h"
#include "telemetry.h"
#include "tracing/trace_context.h"

namespace nixl::trace {
namespace {

    constexpr std::size_t inline_record_attrs = 12;

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
TracerPhaseSink::record(AttributeKey label_key,
                        std::string_view label,
                        std::optional<nixlTime::us_t> timestamp,
                        const TraceIdentity &identity,
                        std::span<const nixlBackendTraceAttr> attrs,
                        const Emit &emit) noexcept {
    try {
        absl::InlinedVector<Attribute, inline_record_attrs> record_attrs;
        record_attrs.push_back({"nixl.backend", backend_});
        if (timestamp.has_value()) {
            record_attrs.push_back(
                {"nixl.phase.timestamp_us", static_cast<std::int64_t>(*timestamp)});
        }
        if (identity.requestId != 0) {
            record_attrs.push_back({"nixl.xfer_request_id", attributeValue(identity.requestId)});
        }
        if (identity.execution != 0) {
            record_attrs.push_back(
                {"nixl.xfer_execution", static_cast<std::int64_t>(identity.execution)});
        }
        if (!label.empty()) {
            record_attrs.push_back({label_key, label});
        }
        for (const auto &attr : attrs) {
            record_attrs.push_back({attr.key, attr.value});
        }

        emit(std::span<const Attribute>(record_attrs.data(), record_attrs.size()));
    }
    catch (...) {
        if (telemetry_ != nullptr) {
            telemetry_->updateTracePhasesDropped(1);
        }
        if (!dropWarned_.test_and_set(std::memory_order_relaxed)) {
            try {
                NIXL_ERROR << "Dropping a trace phase or span for backend " << backend_
                           << ": recording threw. Further drops from this sink are not logged.";
            }
            catch (...) {
            }
        }
    }
}

void
TracerPhaseSink::recordPhase(nixl_trace_phase_t phase,
                             nixlTime::us_t timestamp,
                             const TraceIdentity &identity,
                             std::span<const nixlBackendTraceAttr> attrs,
                             std::string_view label) noexcept {
    const bool named = (phase == nixl_trace_phase_t::OTHER) && !label.empty();
    record("nixl.phase.label",
           label,
           (timestamp != 0) ? std::optional<nixlTime::us_t>{timestamp} : std::nullopt,
           identity,
           attrs,
           [&](std::span<const Attribute> event_attrs) {
               tracer_.emitEvent(named ? label : toStringView(phase),
                                 phaseKind(phase),
                                 identity.context,
                                 identity.requestId,
                                 timestamp,
                                 event_attrs);
           });
}

void
TracerPhaseSink::recordSpan(nixl_trace_span_t span,
                            nixl_xfer_op_t operation,
                            nixlTime::us_t start,
                            nixlTime::us_t end,
                            const TraceIdentity &identity,
                            std::span<const nixlBackendTraceAttr> attrs,
                            std::string_view label) noexcept {
    const bool named = (span == nixl_trace_span_t::OTHER) && !label.empty();
    record("nixl.span.label",
           label,
           std::nullopt,
           identity,
           attrs,
           [&](std::span<const Attribute> span_attrs) {
               tracer_.emitSpan(named ? label : toStringView(span),
                                operationKind(operation),
                                identity.context,
                                identity.requestId,
                                start,
                                end,
                                span_attrs);
           });
}

} // namespace nixl::trace
