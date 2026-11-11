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
#ifndef NIXL_SRC_CORE_TRACING_API_BACKEND_TRACE_H
#define NIXL_SRC_CORE_TRACING_API_BACKEND_TRACE_H

/*
 * Plugin-facing trace phase sink.
 *
 * This header is internal: it is NOT installed, so only in-tree plugins can
 * record phases, and nothing here carries an ABI guarantee --
 * `NIXL_PLUGIN_API_VERSION` does not cover it.
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "common/nixl_time.h"
#include "nixl_types.h"
#include "tracing/api/attribute_key.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

class nixlSerDes;
class nixlBackendTraceSink;

namespace nixl::trace {
struct DeferredPhase;
} // namespace nixl::trace

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
toRecord(nixl_trace_span_t span, nixl_xfer_op_t operation) noexcept {
    switch (span) {
    case nixl_trace_span_t::WIRE_INFLIGHT:
        return (operation == NIXL_READ) ? nixl::trace::Record::WireInflightRead :
                                          nixl::trace::Record::WireInflightWrite;
    case nixl_trace_span_t::OTHER:
        return nixl::trace::Record::Span;
    }
    return nixl::trace::Record::Span;
}

[[nodiscard]] constexpr std::string_view
toStringView(nixl_trace_span_t span, nixl_xfer_op_t operation) noexcept {
    return nixl::trace::toStringView(toRecord(span, operation));
}

/**
 * @brief One key/value attribute on a recorded phase or span.
 *
 * The key is one of nixl::trace::attributeKeys, written as its string literal.
 * The value must stay valid only for the
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
 * Implemented by core, never by a plugin. A backend takes its own from
 * nixlBackendInitParams::traceSink when it is constructed; it is null when
 * tracing is off -- so every call site must null-check. When non-null the sink
 * outlives the backend engine that holds it.
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

    /**
     * @brief The trace identity of the agent's call into this backend: non-null
     * only on the calling thread, while prepXfer, postXfer or a genNotif given a
     * context runs, and only when tracing is active. It is valid for that call; a
     * backend that needs it later copies it.
     */
    [[nodiscard]] virtual const nixl::trace::TraceIdentity *
    currentIdentity() const noexcept = 0;

    virtual void
    writeTraceTags(nixlSerDes &ser_des, const nixl::trace::TraceIdentity &identity) const = 0;

    [[nodiscard]] virtual std::optional<nixl::trace::TraceIdentity>
    readTraceTags(nixlSerDes &ser_des) const = 0;

    virtual void
    deferPhase(const nixl::trace::DeferredPhase &deferred) noexcept = 0;

    virtual void
    recordDeferredPhases() noexcept = 0;
};

#if defined(NIXL_TRACE_ENABLED)

/**
 * @brief A post's trace for a backend's phases: the sink and the post's identity.
 *        It records only with a sink and a sampled identity; record() reads the
 *        clock when the sink needs it, recordAt() takes a time stamped earlier.
 */
class nixlBackendPostTrace {
public:
    nixlBackendPostTrace() = default;

    explicit nixlBackendPostTrace(nixlBackendTraceSink *sink) noexcept {
        if (sink == nullptr) {
            return;
        }
        if (const nixl::trace::TraceIdentity *identity = sink->currentIdentity()) {
            *this = nixlBackendPostTrace(sink, *identity);
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

    [[nodiscard]] nixlBackendTraceSink *
    sink() const noexcept {
        return sink_;
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

#else

class nixlBackendPostTrace {
public:
    nixlBackendPostTrace() = default;

    explicit nixlBackendPostTrace(nixlBackendTraceSink *) noexcept {}

    nixlBackendPostTrace(nixlBackendTraceSink *, const nixl::trace::TraceIdentity &) noexcept {}

    [[nodiscard]] constexpr bool
    active() const noexcept {
        return false;
    }

    [[nodiscard]] nixl::trace::TraceIdentity
    identity() const noexcept {
        return {};
    }

    [[nodiscard]] constexpr nixlTime::us_t
    now() const noexcept {
        return 0;
    }

    void
    record(nixl_trace_phase_t, std::span<const nixlBackendTraceAttr> = {}) const noexcept {}

    void
    recordAt(nixl_trace_phase_t,
             nixlTime::us_t,
             std::span<const nixlBackendTraceAttr> = {}) const noexcept {}

    void
    recordSpan(nixl_trace_span_t, nixl_xfer_op_t, nixlTime::us_t, nixlTime::us_t) const noexcept {}
};

#endif

namespace nixl::trace {

inline constexpr std::size_t deferredPhaseCapacity = 64;

/**
 * @brief A phase deferred with deferPhase(). Deferring
 *        @ref nixl_trace_phase_t::WIRE_COMPLETED also records
 *        @ref nixl_trace_span_t::WIRE_INFLIGHT for `operation`, from
 *        `inflightSince` to `completedAt`, so a backend sets all three; left at
 *        their defaults, the span runs from 0 to 0.
 */
struct DeferredPhase {
    nixlBackendPostTrace trace;
    nixl_trace_phase_t phase = nixl_trace_phase_t::WIRE_COMPLETED;
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixlTime::us_t inflightSince = 0;
    nixlTime::us_t phaseAt = 0;
    nixlTime::us_t completedAt = 0;
};

#if defined(NIXL_TRACE_ENABLED)

inline void
addTraceTags(nixlSerDes &ser_des, const nixlBackendPostTrace &trace) {
    if (trace.active()) {
        trace.sink()->writeTraceTags(ser_des, trace.identity());
    }
}

[[nodiscard]] inline std::optional<TraceIdentity>
getTraceTags(nixlSerDes &ser_des, const nixlBackendTraceSink *sink) {
    return (sink != nullptr) ? sink->readTraceTags(ser_des) : std::nullopt;
}

/**
 * @brief Queues @p deferred on the calling thread, or records it at once when
 *        `deferredPhaseCapacity` phases are already queued there. The entry
 *        keeps the post's sink, so the same thread must call
 *        recordDeferredPhases() before that sink is destroyed.
 */
inline void
deferPhase(const DeferredPhase &deferred) noexcept {
    if (deferred.trace.active()) {
        deferred.trace.sink()->deferPhase(deferred);
    }
}

/** @brief Records the phases the calling thread deferred, in order, and empties its queue. */
inline void
recordDeferredPhases(nixlBackendTraceSink *sink) noexcept {
    if (sink != nullptr) {
        sink->recordDeferredPhases();
    }
}

#else

inline void
addTraceTags(nixlSerDes &, const nixlBackendPostTrace &) noexcept {}

[[nodiscard]] inline std::optional<TraceIdentity>
getTraceTags(nixlSerDes &, const nixlBackendTraceSink *) noexcept {
    return std::nullopt;
}

inline void
deferPhase(const DeferredPhase &) noexcept {}

inline void
recordDeferredPhases(nixlBackendTraceSink *) noexcept {}

#endif

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_API_BACKEND_TRACE_H
