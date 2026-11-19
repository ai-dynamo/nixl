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
#include "tracing/core/trace_sink.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "common/nixl_log.h"
#include "tracing/core/trace_context.h"

namespace nixl::trace {
namespace {

    constexpr std::size_t inline_record_attrs = 12;

    struct DeferredPhaseQueue {
        std::array<DeferredPhase, deferredPhaseCapacity> entries;
        std::size_t count = 0;

        ~DeferredPhaseQueue() {
            if (count != 0) {
                pendingDeferredPhases.fetch_sub(count, std::memory_order_relaxed);
            }
        }
    };

    thread_local DeferredPhaseQueue deferred_phases;

    void
    recordDeferred(const DeferredPhase &deferred) noexcept {
        if (deferred.status == NIXL_SUCCESS) {
            deferred.trace.recordAt(deferred.phase, deferred.phaseAt);
        } else if (auto *const sink = static_cast<TracerPhaseSink *>(deferred.trace.sink())) {
            sink->recordFailedPhase(
                deferred.phase, deferred.phaseAt, deferred.trace.identity(), deferred.status);
        }
        if (deferred.phase == nixl_trace_phase_t::WIRE_COMPLETED) {
            deferred.trace.recordSpan(nixl_trace_span_t::WIRE_INFLIGHT,
                                      deferred.operation,
                                      deferred.inflightSince,
                                      deferred.completedAt);
        }
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
                        std::span<const Attribute> typed_attrs,
                        const Emit &emit) noexcept {
    try {
        absl::InlinedVector<Attribute, inline_record_attrs> record_attrs;
        record_attrs.push_back({AttributeKey::NixlBackend, backend_});
        if (timestamp.has_value()) {
            record_attrs.push_back(
                {AttributeKey::NixlPhaseTimestampUs, static_cast<std::int64_t>(*timestamp)});
        }
        if (*identity.requestId != 0) {
            record_attrs.push_back(
                {AttributeKey::NixlXferRequestId, attributeValue(*identity.requestId)});
        }
        if (identity.execution != 0) {
            record_attrs.push_back(
                {AttributeKey::NixlXferExecution, static_cast<std::int64_t>(identity.execution)});
        }
        if (!label.empty()) {
            record_attrs.push_back({label_key, label});
        }
        for (const auto &attr : attrs) {
            record_attrs.push_back({attr.key, attr.value});
        }
        record_attrs.insert(record_attrs.end(), typed_attrs.begin(), typed_attrs.end());

        emit(std::span<const Attribute>(record_attrs.data(), record_attrs.size()));
    }
    catch (...) {
        if (onDropped_) {
            onDropped_(1);
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
    record(AttributeKey::NixlPhaseLabel,
           label,
           (timestamp != 0) ? std::optional<nixlTime::us_t>{timestamp} : std::nullopt,
           identity,
           attrs,
           {},
           [&](std::span<const Attribute> event_attrs) {
               tracer_.emitEvent(named ? Name(label) : Name(toRecord(phase)),
                                 kindOf(toRecord(phase)),
                                 identity,
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
    const Record span_record = toRecord(span, operation);
    record(AttributeKey::NixlSpanLabel,
           label,
           std::nullopt,
           identity,
           attrs,
           {},
           [&](std::span<const Attribute> span_attrs) {
               tracer_.emitSpan(named ? Name(label) : Name(span_record),
                                named ? operationKind(operation) : kindOf(span_record),
                                identity,
                                start,
                                end,
                                span_attrs);
           });
}

void
TracerPhaseSink::recordFailedPhase(nixl_trace_phase_t phase,
                                   nixlTime::us_t timestamp,
                                   const TraceIdentity &identity,
                                   nixl_status_t status) noexcept {
    const Attribute status_attr{AttributeKey::NixlStatus, status};
    record(
        AttributeKey::NixlPhaseLabel,
        {},
        (timestamp != 0) ? std::optional<nixlTime::us_t>{timestamp} : std::nullopt,
        identity,
        {},
        {&status_attr, 1},
        [&](std::span<const Attribute> event_attrs) {
            tracer_.emitEvent(
                Name(toRecord(phase)), kindOf(toRecord(phase)), identity, timestamp, event_attrs);
        });
}

void
TracerPhaseSink::writeTraceTags(nixlSerDes &ser_des,
                                const TraceIdentity &identity,
                                std::string_view baggage) const {
    encodeTraceTags(ser_des, identity, baggage);
}

std::optional<TraceIdentity>
TracerPhaseSink::readTraceTags(nixlSerDes &ser_des) const {
    return decodeTraceTags(ser_des);
}

std::string
TracerPhaseSink::readTraceBaggage(nixlSerDes &ser_des) const {
    return decodeTraceBaggage(ser_des);
}

std::atomic<std::uint64_t> pendingDeferredPhases{0};

TracerPhaseSink::~TracerPhaseSink() {
    recordDeferredPhases();
}

void
TracerPhaseSink::deferPhase(const DeferredPhase &deferred) noexcept {
    DeferredPhaseQueue &queue = deferred_phases;
    if (queue.count == queue.entries.size()) {
        recordDeferred(deferred);
        return;
    }
    queue.entries[queue.count++] = deferred;
    pendingDeferredPhases.fetch_add(1, std::memory_order_relaxed);
}

void
TracerPhaseSink::recordDeferredPhases() noexcept {
    DeferredPhaseQueue &queue = deferred_phases;
    const std::size_t recorded = queue.count;
    for (std::size_t i = 0; i < recorded; ++i) {
        recordDeferred(queue.entries[i]);
    }
    queue.count = 0;
    if (recorded != 0) {
        pendingDeferredPhases.fetch_sub(recorded, std::memory_order_relaxed);
    }
}

} // namespace nixl::trace
