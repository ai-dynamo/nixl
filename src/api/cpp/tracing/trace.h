/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_API_CPP_TRACING_TRACE_H
#define NIXL_SRC_API_CPP_TRACING_TRACE_H

/*
 * nixl::trace -- shared tracing API.
 *
 * A single set of call sites fans out to every enabled tracing backend at
 * runtime (NVTX today; Chakra and others later). This header is deliberately
 * self-contained: it depends only on standard headers and on no NIXL type, so
 * external consumers can include it in isolation.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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

/** @brief Visual hint for live profilers (NVTX). Ignored by offline backends. */
struct Color {
    std::uint8_t r{0};
    std::uint8_t g{0};
    std::uint8_t b{0};
};

/**
 * @brief Opaque span identifier. Meaningful on backends that build a DAG
 *        (Chakra Node.id); returned as {0} by backends that do not (NVTX).
 */
struct SpanId {
    std::uint64_t value{0};
};

/**
 * @brief A single active span within one backend. Its destructor ends the
 *        range / fixes the duration for that backend.
 */
class iSpanBackend {
public:
    virtual ~iSpanBackend() = default;

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
class iTraceBackend {
public:
    virtual ~iTraceBackend() = default;

    [[nodiscard]] virtual std::unique_ptr<iSpanBackend>
    beginSpan(std::string_view name, Kind kind, Color color) = 0;

    virtual void
    mark(std::string_view name, Kind kind) = 0;

    virtual void
    pushCorrelationId(std::uint64_t id) = 0;
    virtual void
    popCorrelationId() = 0;

    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;
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

    /** @brief Id of the first backend span, or {0} when inactive. */
    [[nodiscard]] SpanId
    id() const noexcept;

private:
    friend class Tracer;

    explicit Span(std::vector<std::unique_ptr<iSpanBackend>> backends) noexcept
        : backends_(std::move(backends)) {}

    std::vector<std::unique_ptr<iSpanBackend>> backends_;
};

/**
 * @brief Composite tracer owned by the Agent. A single beginSpan()/mark() call
 *        fans out to every enabled backend. No singleton; injected explicitly.
 */
class Tracer {
public:
    explicit Tracer(std::vector<std::unique_ptr<iTraceBackend>> backends) noexcept;
    ~Tracer();
    Tracer(const Tracer &) = delete;
    Tracer &
    operator=(const Tracer &) = delete;

    /** @brief True when no backend is active. */
    [[nodiscard]] bool
    empty() const noexcept {
        return backends_.empty();
    }

    [[nodiscard]] Span
    beginSpan(std::string_view name, Kind kind = Kind::Generic, Color color = {});

    void
    mark(std::string_view name, Kind kind = Kind::Metadata);

    void
    pushCorrelationId(std::uint64_t id);
    void
    popCorrelationId();

private:
    std::vector<std::unique_ptr<iTraceBackend>> backends_;
};

/** @brief Inputs used to build the composite tracer from enabled backends. */
struct TracerConfig {
    /** @brief Agent name; used e.g. as the NVTX domain name. */
    std::string agentName;
    /** @brief Backend names requested at runtime (e.g. {"nvtx"}). */
    std::vector<std::string> backends;
};

/**
 * @brief Build a tracer from (requested backends) intersected with (backends
 *        compiled in). Returns nullptr when the result is empty, so callers can
 *        cheaply null-check before tracing.
 */
[[nodiscard]] std::unique_ptr<Tracer>
makeTracer(const TracerConfig &config);

} // namespace nixl::trace

#endif // NIXL_SRC_API_CPP_TRACING_TRACE_H
