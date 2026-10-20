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

/*
 * nixl::trace -- internal NIXL core tracing API.
 *
 * A single set of call sites fans out to every enabled tracing backend at
 * runtime (NVTX today; Chakra and others later). This is an internal core API
 * (compiled into libnixl, not installed as a public header). It is kept
 * deliberately self-contained -- depending only on standard headers and on
 * trace_context.h and nixl_time.h -- so it stays easy to reason about and
 * could be promoted to a public header later without churn if an external
 * consumer appears.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common/nixl_time.h"
#include "tracing/trace_context.h"

namespace nixl::trace {

/**
 * @brief Operation kind. Aligns 1:1 with the Chakra NodeType vocabulary and is
 *        used as a color/label hint by NVTX.
 */
enum class Kind : std::uint8_t {
    Generic = 0,
    Compute,
    MemoryR,
    MemoryW,
    CommSend,
    CommRecv,
    CommColl,
    Metadata,
};

/**
 * @brief Opaque span identifier. Meaningful on backends that build a DAG
 *        (Chakra Node.id); returned as {0} by backends that do not (NVTX).
 */
struct SpanId {
    std::uint64_t value{0};
};

/**
 * @brief Typed attribute of an emitted span or event. The key and a string
 *        value are views, valid only for the call.
 */
struct Attribute {
    std::string_view key;
    std::variant<std::string_view, std::int64_t, double> value;
};

/**
 * @brief A single active span within one backend. Its destructor ends the
 *        range / fixes the duration for that backend.
 */
class SpanBackend {
public:
    virtual ~SpanBackend() = default;

    virtual void
    addAttribute(std::string_view key, std::string_view value) = 0;
    virtual void
    addAttribute(std::string_view key, std::int64_t value) = 0;
    virtual void
    addAttribute(std::string_view key, double value) = 0;

    virtual void
    addCtrlDep(SpanId parent) = 0;
    virtual void
    addDataDep(SpanId parent) = 0;

    [[nodiscard]] virtual SpanId
    id() const noexcept = 0;
};

/** @brief One enabled backend type (NVTX, Chakra, ...). */
class TraceBackend {
public:
    virtual ~TraceBackend() = default;

    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginSpan(std::string_view name, Kind kind) = 0;

    virtual void
    mark(std::string_view name, Kind kind) = 0;

    virtual void
    pushContext(const TraceContext &context) = 0;
    virtual void
    popContext() = 0;

    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;

    /**
     * @brief Record a span from @p start to @p end, or an event at @p at, that
     *        already happened. Times are `nixlTime` microseconds on
     *        `std::chrono::steady_clock`; @p context is the record's own, not
     *        the pushed one, and an invalid context means none.
     */
    virtual void
    emitSpan(std::string_view name,
             Kind kind,
             const TraceContext &context,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs) = 0;
    virtual void
    emitEvent(std::string_view name,
              Kind kind,
              const TraceContext &context,
              nixlTime::us_t at,
              std::span<const Attribute> attrs) = 0;
};

class Tracer;

/**
 * @brief Public span handle. Fans out attribute/dependency calls to every
 *        backend span it owns and ends them all on destruction. Move-only.
 */
class Span {
public:
    Span() = default;
    Span(Span &&) noexcept = default;
    Span &
    operator=(Span &&) noexcept = default;
    Span(const Span &) = delete;
    Span &
    operator=(const Span &) = delete;
    ~Span() = default;

    /** @brief True when at least one backend produced a span (cheap gate). */
    [[nodiscard]] bool
    active() const noexcept {
        return !backends_.empty();
    }

    void
    addAttribute(std::string_view key, std::string_view value);
    void
    addAttribute(std::string_view key, std::int64_t value);
    void
    addAttribute(std::string_view key, double value);

    void
    addCtrlDep(SpanId parent);
    void
    addDataDep(SpanId parent);

    /** @brief Id of the first backend that provides a non-zero id, or {0}
     *         when none do (e.g. NVTX-only). Order-independent across backends. */
    [[nodiscard]] SpanId
    id() const noexcept;

private:
    friend class Tracer;

    explicit Span(std::vector<std::unique_ptr<SpanBackend>> backends) noexcept
        : backends_(std::move(backends)) {}

    std::vector<std::unique_ptr<SpanBackend>> backends_;
};

/**
 * @brief Fills its buffer with random bytes for trace-context ids, or throws a
 *        std::exception. An all-zero id is drawn again; null selects the default
 *        getentropy-backed source.
 */
using random_bytes_fn_t = void (*)(std::uint8_t *, std::size_t);

/**
 * @brief Composite tracer owned by the Agent. A single beginSpan(), mark(),
 *        emitSpan() or emitEvent() call fans out to every enabled backend. No
 *        singleton; injected explicitly.
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

    [[nodiscard]] Span
    beginSpan(std::string_view name, Kind kind = Kind::Generic);

    void
    mark(std::string_view name, Kind kind = Kind::Metadata);

    void
    pushContext(const TraceContext &context);
    void
    popContext();

    void
    emitSpan(std::string_view name,
             Kind kind,
             const TraceContext &context,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs);
    void
    emitEvent(std::string_view name,
              Kind kind,
              const TraceContext &context,
              nixlTime::us_t at,
              std::span<const Attribute> attrs);

private:
    std::vector<std::unique_ptr<TraceBackend>> backends_;
    const double sampleRatio_;
    const random_bytes_fn_t randomBytes_;
    mutable std::atomic<bool> contextFailureLogged_ = false;
};

/**
 * @brief RAII guard that keeps a trace context active on the tracer for its
 *        lifetime, so every span/mark created while it is in scope carries
 *        that context. Null-tracer-safe; an invalid context pushes nothing.
 *        Used to link spans emitted on different threads (e.g. postXferReq on
 *        the caller thread vs. the completion on the polling thread) to the
 *        same logical request.
 */
class CorrelationScope {
public:
    [[nodiscard]] CorrelationScope(Tracer *tracer, const TraceContext &context)
        : tracer_(((tracer != nullptr) && context.valid()) ? tracer : nullptr) {
        if (tracer_ != nullptr) {
            tracer_->pushContext(context);
        }
    }

    ~CorrelationScope() {
        if (tracer_ != nullptr) {
            tracer_->popContext();
        }
    }

    CorrelationScope(const CorrelationScope &) = delete;
    CorrelationScope &
    operator=(const CorrelationScope &) = delete;
    CorrelationScope(CorrelationScope &&) = delete;
    CorrelationScope &
    operator=(CorrelationScope &&) = delete;

private:
    Tracer *tracer_;
};

/** @brief Inputs used to build the composite tracer from enabled backends. */
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
