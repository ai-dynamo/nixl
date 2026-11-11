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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_TRACE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_TRACE_H

#include <array>
#include <string_view>

#include "nixl_types.h"
#include "tracing/api/backend_trace.h"

namespace nixl::ucx {

#if defined(NIXL_TRACE_ENABLED)

class PostTrace {
public:
    void
    begin(nixlBackendTraceSink *sink,
          nixl_xfer_op_t operation,
          bool progress_thread,
          bool composite) noexcept {
        trace_ = nixlBackendPostTrace{sink};
        operation_ = operation;
        completesOnFlush_ = trace_.active() && progress_thread && !composite;
        completedAtOnce_ = false;
        const nixlTime::us_t submit_at = trace_.now();
        const bool records_inflight = completesOnFlush_ || (trace_.active() && composite);
        inflightSince_ = (records_inflight && (submit_at == 0)) ? nixlTime::getUs() : submit_at;
        trace_.recordAt(nixl_trace_phase_t::SUBMIT, submit_at);
    }

    [[nodiscard]] bool
    active() const noexcept {
        return trace_.active();
    }

    [[nodiscard]] const nixlBackendPostTrace &
    post() const noexcept {
        return trace_;
    }

    [[nodiscard]] bool
    completesOnFlush() const noexcept {
        return completesOnFlush_;
    }

    void
    flushIssued(bool completed) noexcept {
        completedAtOnce_ = completed && completesOnFlush_;
    }

    void
    submitted() const noexcept {
        const nixlTime::us_t submitted_at = trace_.now();
        trace_.recordAt(nixl_trace_phase_t::WIRE_SUBMITTED, submitted_at);
        if (completedAtOnce_) {
            recordCompleted(submitted_at);
        }
    }

    void
    flushCompleted() const noexcept {
        if (!completesOnFlush_) {
            return;
        }
        const nixlTime::us_t phase_at = trace_.now();
        nixl::trace::deferPhase({.trace = trace_,
                                 .phase = nixl_trace_phase_t::WIRE_COMPLETED,
                                 .operation = operation_,
                                 .inflightSince = inflightSince_,
                                 .phaseAt = phase_at,
                                 .completedAt = (phase_at != 0) ? phase_at : nixlTime::getUs()});
    }

    void
    notifSent(nixl_status_t status) const noexcept {
        if (trace_.active() && ((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG))) {
            nixl::trace::deferPhase({.trace = trace_,
                                     .phase = nixl_trace_phase_t::NOTIF_SENT,
                                     .phaseAt = trace_.now()});
        }
    }

    void
    drain() const noexcept {
        if (trace_.active()) {
            nixl::trace::recordDeferredPhases();
        }
    }

    void
    completed() const noexcept {
        recordCompleted(trace_.now());
    }

private:
    void
    recordCompleted(nixlTime::us_t phase_at) const noexcept {
        const nixlTime::us_t completed_at = (phase_at != 0) ? phase_at : nixlTime::getUs();
        trace_.recordAt(nixl_trace_phase_t::WIRE_COMPLETED, phase_at);
        trace_.recordSpan(
            nixl_trace_span_t::WIRE_INFLIGHT, operation_, inflightSince_, completed_at);
    }

    nixlBackendPostTrace trace_;
    nixl_xfer_op_t operation_ = NIXL_WRITE;
    nixlTime::us_t inflightSince_ = 0;
    bool completesOnFlush_ = false;
    bool completedAtOnce_ = false;
};

inline void
traceReceived(const nixlBackendPostTrace &trace, std::string_view remote_agent) noexcept {
    const std::array attrs{nixlBackendTraceAttr{"nixl.remote_agent", remote_agent}};
    trace.record(nixl_trace_phase_t::NOTIF_RECEIVED, attrs);
}

#else

class PostTrace {
public:
    void
    begin(nixlBackendTraceSink *, nixl_xfer_op_t, bool, bool) noexcept {}

    [[nodiscard]] constexpr bool
    active() const noexcept {
        return false;
    }

    [[nodiscard]] nixlBackendPostTrace
    post() const noexcept {
        return {};
    }

    [[nodiscard]] constexpr bool
    completesOnFlush() const noexcept {
        return false;
    }

    void
    flushIssued(bool) noexcept {}

    void
    submitted() const noexcept {}

    void
    flushCompleted() const noexcept {}

    void
    notifSent(nixl_status_t) const noexcept {}

    void
    drain() const noexcept {}

    void
    completed() const noexcept {}
};

inline void
traceReceived(const nixlBackendPostTrace &, std::string_view) noexcept {}

#endif

} // namespace nixl::ucx

#endif
