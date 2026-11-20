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
#ifndef NIXL_SRC_CORE_TRACING_TRACE_H
#define NIXL_SRC_CORE_TRACING_TRACE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tracing/trace_context.h"
#include "tracing/tracing.h"

namespace nixl::trace {

/**
 * @brief One enabled backend type (NVTX, Chakra, ...). Each record names the
 *        span id of its context, pushed or passed, as its parent, except the
 *        span beginContextSpan() begins, which has that id as its own.
 */
class TraceBackend {
public:
    virtual ~TraceBackend() = default;

    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginSpan(std::string_view name, Kind kind) = 0;

    virtual void
    mark(std::string_view name, Kind kind) = 0;

    virtual void
    pushContext(const TraceContext &context, std::uint64_t request_id) = 0;
    virtual void
    popContext() = 0;

    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;

    /**
     * @brief Record a span from @p start to @p end, or an event at @p at, that
     *        already happened. Times are `nixlTime` microseconds on
     *        `std::chrono::steady_clock`; @p context is the record's own, not
     *        the pushed one, and an invalid context means none. @p request_id
     *        is likewise the record's own, and 0 means none.
     */
    virtual void
    emitSpan(std::string_view name,
             Kind kind,
             const TraceContext &context,
             std::uint64_t request_id,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs) = 0;
    virtual void
    emitEvent(std::string_view name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              nixlTime::us_t at,
              std::span<const Attribute> attrs) = 0;

    /**
     * @brief Begin the span that @p context identifies, rather than a span
     *        inside the pushed context. @p context and @p request_id are the
     *        span's own, as in emitSpan().
     */
    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginContextSpan(std::string_view name,
                     Kind kind,
                     const TraceContext &context,
                     std::uint64_t request_id) = 0;
};

/**
 * @brief Fills its buffer with random bytes for trace-context ids, or throws a
 *        std::exception. An all-zero id is drawn again; null selects the default
 *        getentropy-backed source.
 */
using random_bytes_fn_t = void (*)(std::span<std::uint8_t>);

/**
 * @brief Composite tracer owned by the Agent. A single beginSpan(),
 *        beginContextSpan(), mark(), emitSpan() or emitEvent() call fans out to
 *        every enabled backend. No singleton; injected explicitly.
 */
class Tracer {
public:
    explicit Tracer(std::vector<std::unique_ptr<TraceBackend>> backends,
                    double sample_ratio = 0.0,
                    random_bytes_fn_t random_bytes = nullptr) noexcept;
    ~Tracer();
    Tracer(const Tracer &) = delete;
    Tracer &
    operator=(const Tracer &) = delete;

    /** @brief True when no backend is active. */
    [[nodiscard]] bool
    empty() const noexcept {
        return backends_.empty();
    }

    /** @brief Sampling ratio applied when a trace context is generated for one
     *         of this agent's requests. Carried here because the tracer pointer
     *         is what the generation call sites have; the tracer never reads it
     *         itself. */
    [[nodiscard]] double
    sampleRatio() const noexcept {
        return sampleRatio_;
    }

    /** @brief Random-byte source for this agent's trace contexts; null means the
     *         default. Carried here for the same reason as sampleRatio(). */
    [[nodiscard]] random_bytes_fn_t
    randomBytes() const noexcept {
        return randomBytes_;
    }

    /** @brief Logs this tracer's first trace-context generation failure at ERROR;
     *         later ones are not logged. */
    void
    reportContextFailure(std::string_view reason) const noexcept;

    [[nodiscard]] std::span<const std::unique_ptr<TraceBackend>>
    backends() const noexcept {
        return backends_;
    }

    [[nodiscard]] Span
    beginSpan(std::string_view name, Kind kind = Kind::Generic) {
        return detail::beginSpan(*this, name, kind);
    }

    [[nodiscard]] Span
    beginContextSpan(std::string_view name,
                     Kind kind,
                     const TraceContext &context,
                     std::uint64_t request_id) {
        return detail::beginContextSpan(*this, name, kind, context, request_id);
    }

    void
    mark(std::string_view name, Kind kind = Kind::Metadata) {
        detail::mark(*this, name, kind);
    }

    void
    pushContext(const TraceContext &context, std::uint64_t request_id) {
        detail::pushContext(*this, context, request_id);
    }

    void
    popContext() {
        detail::popContext(*this);
    }

    void
    emitSpan(std::string_view name,
             Kind kind,
             const TraceContext &context,
             std::uint64_t request_id,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs);

    void
    emitEvent(std::string_view name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              nixlTime::us_t at,
              std::span<const Attribute> attrs) {
        detail::emitEvent(*this, name, kind, context, request_id, at, attrs);
    }

private:
    std::vector<std::unique_ptr<TraceBackend>> backends_;
    const double sampleRatio_;
    const random_bytes_fn_t randomBytes_;
    mutable std::atomic<bool> contextFailureLogged_ = false;
};

struct TracerConfig {
    /** @brief Agent name; used e.g. as the NVTX domain name. */
    std::string agentName;
    /** @brief Backend names requested at runtime (e.g. {"nvtx"}). */
    std::vector<std::string> backends;
    /** @brief Head-based sampling probability in [0, 1]; 0 samples nothing. */
    double sampleRatio{0.0};
};

/**
 * @brief Build a tracer from (requested backends) intersected with (backends
 *        compiled in). Returns nullptr when the result is empty, so callers can
 *        cheaply null-check before tracing.
 */
[[nodiscard]] std::unique_ptr<Tracer>
makeTracer(const TracerConfig &config);

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_TRACE_H
