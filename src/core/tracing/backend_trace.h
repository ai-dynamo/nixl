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
#ifndef NIXL_SRC_CORE_TRACING_BACKEND_TRACE_H
#define NIXL_SRC_CORE_TRACING_BACKEND_TRACE_H

/*
 * Plugin-facing trace phase sink.
 *
 * This header is internal: it is NOT installed, and `nixlBackendInitParams`
 * only forward-declares `nixlBackendTraceSink`, so an out-of-tree plugin sees
 * an opaque pointer. Only in-tree plugins can record phases, and nothing here
 * carries an ABI guarantee -- `NIXL_PLUGIN_API_VERSION` does not cover it.
 */

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "backend/backend_aux.h"
#include "common/nixl_time.h"
#include "nixl_types.h"
#include "tracing/attribute_key.h"
#include "tracing/record.h"
#include "tracing/trace_identity.h"

class nixlSerDes;

/**
 * @brief Fixed set of backend phases.
 *
 * Closed on purpose, so timelines from different backends stay comparable.
 * Backend authors map their internals onto it rather than inventing names,
 * e.g. libfabric's post_write/post_read/post_send all report as
 * @ref WIRE_SUBMITTED.
 */
enum class nixl_trace_phase_t : std::uint8_t {
    /** @brief The backend accepted the operation and owns it from here. */
    SUBMIT,
    /** @brief Handed to the transport (queued on the wire). */
    WIRE_SUBMITTED,
    /** @brief The transport signalled completion (e.g. a CQ entry was reaped). */
    WIRE_COMPLETED,
    /** @brief A notification was posted to the wire. */
    NOTIF_SENT,
    /** @brief A notification from a peer was observed locally. */
    NOTIF_RECEIVED,
    /** @brief The peer's side of the operation was observed. */
    REMOTE_OBSERVED,
    /** @brief Anything the vocabulary above does not cover; name it with the
     *         `label` argument of @ref nixlBackendTraceSink::recordPhase. Meant
     *         to be the rare escape hatch, not the default. */
    OTHER,
};

[[nodiscard]] constexpr nixl::trace::Record
toRecord(nixl_trace_phase_t phase) noexcept {
    switch (phase) {
    case nixl_trace_phase_t::SUBMIT:
        return nixl::trace::Record::Submit;
    case nixl_trace_phase_t::WIRE_SUBMITTED:
        return nixl::trace::Record::WireSubmitted;
    case nixl_trace_phase_t::WIRE_COMPLETED:
        return nixl::trace::Record::WireCompleted;
    case nixl_trace_phase_t::NOTIF_SENT:
        return nixl::trace::Record::NotifSent;
    case nixl_trace_phase_t::NOTIF_RECEIVED:
        return nixl::trace::Record::NotifReceived;
    case nixl_trace_phase_t::REMOTE_OBSERVED:
        return nixl::trace::Record::RemoteObserved;
    case nixl_trace_phase_t::OTHER:
        return nixl::trace::Record::Phase;
    }
    return nixl::trace::Record::Phase;
}

[[nodiscard]] constexpr std::string_view
toStringView(nixl_trace_phase_t phase) noexcept {
    return nixl::trace::toStringView(toRecord(phase));
}

/**
 * @brief Fixed set of backend spans, the intervals a backend records with
 *        @ref nixlBackendTraceSink::recordSpan.
 *
 * Closed on purpose, like @ref nixl_trace_phase_t, and apart from it: no span
 * is named like a phase.
 */
enum class nixl_trace_span_t : std::uint8_t {
    /** @brief The transfer in flight, from its @ref nixl_trace_phase_t::SUBMIT
     *         to its @ref nixl_trace_phase_t::WIRE_COMPLETED. */
    WIRE_INFLIGHT,
    /** @brief Anything the vocabulary above does not cover; name it with the
     *         `label` argument of @ref nixlBackendTraceSink::recordSpan. Meant
     *         to be the rare escape hatch, not the default. */
    OTHER,
};

[[nodiscard]] constexpr nixl::trace::Record
toRecord(nixl_trace_span_t span) noexcept {
    switch (span) {
    case nixl_trace_span_t::WIRE_INFLIGHT:
        return nixl::trace::Record::WireInflight;
    case nixl_trace_span_t::OTHER:
        return nixl::trace::Record::Span;
    }
    return nixl::trace::Record::Span;
}

[[nodiscard]] constexpr std::string_view
toStringView(nixl_trace_span_t span) noexcept {
    return nixl::trace::toStringView(toRecord(span));
}

/**
 * @brief One key/value attribute on a recorded phase or span.
 *
 * The key is a string literal. The value must stay valid only for the
 * duration of the @ref nixlBackendTraceSink::recordPhase or
 * @ref nixlBackendTraceSink::recordSpan call; the sink copies what it retains.
 */
struct nixlBackendTraceAttr {
    nixl::trace::AttributeKey key;
    std::string_view value;
};

/**
 * @brief Sink a backend uses to record a phase or a span of its own work.
 *
 * Implemented by core, never by a plugin. A backend receives one through
 * `nixlBackendInitParams::traceSink`, which is null when tracing is off -- so
 * every call site must null-check. When non-null the sink outlives the backend
 * engine that holds it.
 */
class nixlBackendTraceSink {
public:
    virtual ~nixlBackendTraceSink() = default;

    /**
     * @brief Record that @p phase happened at @p timestamp.
     *
     * @param phase     Which phase; see @ref nixl_trace_phase_t.
     * @param timestamp Microsecond reading of the monotonic `nixlTime` clock
     *                  (`nixlTime::getUs()`, backed by `std::chrono::steady_clock`),
     *                  not a wall clock. Recorded verbatim as the event's time and,
     *                  when non-zero, as `nixl.phase.timestamp_us`.
     * @param identity  Trace identity the phase belongs to: the post's own for a
     *                  local phase, the sender's for a phase observed on receipt.
     *                  Valid for the call only. Its context is the record's
     *                  own, its request id is recorded as the integer
     *                  `nixl.xfer_request_id` when non-zero (the event is
     *                  correlated on it), and its execution number as
     *                  `nixl.xfer_execution` when non-zero.
     * @param attrs     Extra attributes; see @ref nixlBackendTraceAttr for the
     *                  lifetime rule, which applies to @p label as well.
     * @param label     Name for a @ref nixl_trace_phase_t::OTHER phase. Recorded
     *                  as an attribute for every phase, and additionally used as
     *                  the event name for `OTHER`. May be empty.
     *
     * Safe to call concurrently from any thread, including a progress thread;
     * each call records one point event, whichever thread makes it. `noexcept`
     * because backends call it from completion callbacks and other contexts
     * that cannot absorb an exception -- a failure to record is dropped rather
     * than propagated, since tracing must never break a transfer.
     */
    virtual void
    recordPhase(nixl_trace_phase_t phase,
                nixlTime::us_t timestamp,
                const nixl::trace::TraceIdentity &identity,
                std::span<const nixlBackendTraceAttr> attrs = {},
                std::string_view label = {}) noexcept = 0;

    /** @brief Record that @p phase happened now, at nixlTime::getUs() when
     *         needsTimestamps(), otherwise at 0. */
    void
    recordPhase(nixl_trace_phase_t phase,
                const nixl::trace::TraceIdentity &identity,
                std::span<const nixlBackendTraceAttr> attrs = {},
                std::string_view label = {}) noexcept {
        recordPhase(phase,
                    needsTimestamps() ? nixlTime::getUs() : nixlTime::us_t{0},
                    identity,
                    attrs,
                    label);
    }

    /**
     * @brief Record that @p span lasted from @p start to @p end.
     *
     * @param span      Which span; see @ref nixl_trace_span_t.
     * @param operation Direction of the span's transfer: the span is recorded
     *                  as a send (`CommSend`) for `NIXL_WRITE` and as a receive
     *                  (`CommRecv`) for `NIXL_READ`.
     * @param start     When the span began, on the clock of @ref recordPhase's
     *                  `timestamp`. Recorded verbatim as the span's start.
     * @param end       When the span ended, on the same clock. Recorded verbatim
     *                  as the span's end.
     * @param identity  Trace identity the span belongs to, recorded as for
     *                  @ref recordPhase. Valid for the call only.
     * @param attrs     Extra attributes; see @ref nixlBackendTraceAttr for the
     *                  lifetime rule, which applies to @p label as well.
     * @param label     Name for a @ref nixl_trace_span_t::OTHER span. Recorded
     *                  as `nixl.span.label` for every span, and additionally used
     *                  as the span name for `OTHER`. May be empty.
     *
     * Safe to call concurrently from any thread, including a progress thread,
     * and `noexcept` for the same reasons as @ref recordPhase; each call records
     * one span, and a span that fails to record is dropped like a phase.
     */
    virtual void
    recordSpan(nixl_trace_span_t span,
               nixl_xfer_op_t operation,
               nixlTime::us_t start,
               nixlTime::us_t end,
               const nixl::trace::TraceIdentity &identity,
               std::span<const nixlBackendTraceAttr> attrs = {},
               std::string_view label = {}) noexcept = 0;

    [[nodiscard]] virtual bool
    needsTimestamps() const noexcept {
        return true;
    }
};

/**
 * @brief A post's trace for a backend's phases: the sink and the post's identity.
 *        It records only with a sink and a sampled identity; record() reads the
 *        clock when the sink needs it, recordAt() takes a time stamped earlier.
 */
class nixlBackendPostTrace {
public:
    nixlBackendPostTrace() = default;

    nixlBackendPostTrace(nixlBackendTraceSink *sink, const nixl_opt_b_args_t *opt_args) noexcept {
        if ((opt_args != nullptr) && (opt_args->traceIdentity != nullptr)) {
            *this = nixlBackendPostTrace(sink, *opt_args->traceIdentity);
        }
    }

    nixlBackendPostTrace(nixlBackendTraceSink *sink,
                         const nixl::trace::TraceIdentity &identity) noexcept
        : sink_(identity.context.sampled() ? sink : nullptr),
          identity_(identity),
          needsTimestamps_((sink_ != nullptr) && sink_->needsTimestamps()) {}

    [[nodiscard]] bool
    active() const noexcept {
        return sink_ != nullptr;
    }

    [[nodiscard]] const nixl::trace::TraceIdentity &
    identity() const noexcept {
        return identity_;
    }

    [[nodiscard]] nixlTime::us_t
    now() const noexcept {
        return needsTimestamps_ ? nixlTime::getUs() : nixlTime::us_t{0};
    }

    void
    record(nixl_trace_phase_t phase,
           std::span<const nixlBackendTraceAttr> attrs = {}) const noexcept {
        if (sink_ != nullptr) {
            sink_->recordPhase(phase, now(), identity_, attrs);
        }
    }

    void
    recordAt(nixl_trace_phase_t phase,
             nixlTime::us_t timestamp,
             std::span<const nixlBackendTraceAttr> attrs = {}) const noexcept {
        if (sink_ != nullptr) {
            sink_->recordPhase(phase, timestamp, identity_, attrs);
        }
    }

    void
    recordSpan(nixl_trace_span_t span,
               nixl_xfer_op_t operation,
               nixlTime::us_t start,
               nixlTime::us_t end) const noexcept {
        if (sink_ != nullptr) {
            sink_->recordSpan(span, operation, start, end, identity_);
        }
    }

private:
    nixlBackendTraceSink *sink_ = nullptr;
    nixl::trace::TraceIdentity identity_;
    bool needsTimestamps_ = false;
};

namespace nixl::trace {

void
addTraceTags(nixlSerDes &ser_des, const TraceIdentity &identity);

[[nodiscard]] std::optional<TraceIdentity>
getTraceTags(nixlSerDes &ser_des);

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_BACKEND_TRACE_H
