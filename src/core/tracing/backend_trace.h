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
#include <span>
#include <string_view>

#include "common/nixl_time.h"

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

/**
 * @brief One key/value attribute on a recorded phase.
 *
 * Both views must stay valid only for the duration of the
 * @ref nixlBackendTraceSink::recordPhase call; the sink copies what it retains.
 */
struct nixlBackendTraceAttr {
    std::string_view key;
    std::string_view value;
};

/**
 * @brief Sink a backend uses to record a phase of its own work.
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
     * @param label     Name for a @ref nixl_trace_phase_t::OTHER phase. Recorded
     *                  as an attribute for every phase, and additionally used as
     *                  the span name for `OTHER`. May be empty.
     * @param timestamp Microsecond reading of the monotonic `nixlTime` clock
     *                  (`nixlTime::getUs()`, backed by `std::chrono::steady_clock`),
     *                  not a wall clock. Recorded verbatim on the span as
     *                  `nixl.phase.timestamp_us`.
     * @param attrs     Extra attributes; see @ref nixlBackendTraceAttr for the
     *                  lifetime rule, which applies to @p label as well.
     *
     * Safe to call concurrently from any thread, including a progress thread;
     * each call produces a span scoped to the calling thread. `noexcept`
     * because backends call it from completion callbacks and other contexts
     * that cannot absorb an exception -- a failure to record is dropped rather
     * than propagated, since tracing must never break a transfer.
     */
    virtual void
    recordPhase(nixl_trace_phase_t phase,
                std::string_view label,
                nixlTime::us_t timestamp,
                std::span<const nixlBackendTraceAttr> attrs = {}) noexcept = 0;
};

#endif // NIXL_SRC_CORE_TRACING_BACKEND_TRACE_H
