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

#include <cstdint>

#include "common/nixl_log.h"

namespace nixl::trace {

std::string_view
toStringView(nixl_trace_phase_t phase) noexcept {
    switch (phase) {
    case nixl_trace_phase_t::SUBMIT:
        return "nixl::submit";
    case nixl_trace_phase_t::WIRE_SUBMITTED:
        return "nixl::wire.submitted";
    case nixl_trace_phase_t::WIRE_COMPLETED:
        return "nixl::wire.completed";
    case nixl_trace_phase_t::NOTIF_SENT:
        return "nixl::notif.sent";
    case nixl_trace_phase_t::NOTIF_RECEIVED:
        return "nixl::notif.received";
    case nixl_trace_phase_t::REMOTE_OBSERVED:
        return "nixl::remote.observed";
    case nixl_trace_phase_t::OTHER:
        return "nixl::phase";
    }
    return "nixl::phase";
}

void
TracerPhaseSink::recordPhase(nixl_trace_phase_t phase,
                             std::string_view label,
                             nixlTime::us_t timestamp,
                             std::span<const nixlBackendTraceAttr> attrs) noexcept {
    try {
        const bool named = (phase == nixl_trace_phase_t::OTHER) && !label.empty();
        Span span = tracer_.beginSpan(named ? label : toStringView(phase), Kind::Metadata);
        if (!span.active()) {
            return;
        }

        span.addAttribute("nixl.backend", backend_);
        span.addAttribute("nixl.phase.timestamp_us", static_cast<std::int64_t>(timestamp));
        if (!label.empty()) {
            span.addAttribute("nixl.phase.label", label);
        }
        for (const auto &attr : attrs) {
            span.addAttribute(attr.key, attr.value);
        }
    }
    catch (...) {
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
