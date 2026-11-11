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
#ifndef NIXL_SRC_CORE_TRACING_CORE_TRACE_H
#define NIXL_SRC_CORE_TRACING_CORE_TRACE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common/nixl_time.h"
#include "tracing/api/attribute_key.h"
#include "tracing/api/trace_backend.h"
#include "tracing/api/tracer.h"
#include "tracing/core/trace_context.h"
#include "tracing/api/tracing.h"

namespace nixl::trace {

/**
 * @brief Fills its buffer with random bytes for trace-context ids, or throws a
 *        std::exception. An all-zero id is drawn again; null selects the default
 *        getentropy-backed source.
 */
using random_bytes_fn_t = void (*)(std::span<std::uint8_t>);

/**
 * @brief Composite tracer owned by the Agent. A single beginSpan(),
 *        beginContextSpan(), emitSpan(), emitContextSpan() or emitEvent() call
 *        fans out to every enabled backend. No singleton; injected explicitly.
 */
class CompositeTracer final : public Tracer {
public:
    explicit CompositeTracer(std::vector<std::unique_ptr<TraceBackend>> backends,
                             double sample_ratio,
                             random_bytes_fn_t random_bytes = nullptr) noexcept;
    ~CompositeTracer() override;
    CompositeTracer(const CompositeTracer &) = delete;
    CompositeTracer &
    operator=(const CompositeTracer &) = delete;

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

    [[nodiscard]] bool
    needsTimestamps() const noexcept {
        return needsTimestamps_;
    }

    /** @brief Logs this tracer's first trace-context generation failure at ERROR;
     *         later ones are not logged. */
    void
    reportContextFailure(std::string_view reason) const noexcept;

    /** @brief Logs this tracer's first supplied trace context that is neither a
     *         traceparent nor a trace context record at WARN; later ones are not
     *         logged. */
    void
    reportUnparsableParent(std::size_t size) const noexcept;

    [[nodiscard]] std::span<const std::unique_ptr<TraceBackend>>
    backends() const noexcept {
        return backends_;
    }

    [[nodiscard]] Span
    beginSpan(Name name, Kind kind = Kind::Generic) override;

    [[nodiscard]] Span
    beginContextSpan(Name name,
                     Kind kind,
                     const TraceContext &context,
                     std::uint64_t request_id) override;

    void
    pushContext(const TraceContext &context, std::uint64_t request_id) override;

    void
    popContext() override;

    void
    emitSpan(Name name, Kind kind, nixlTime::us_t start) override;

    void
    emitSpan(Name name,
             Kind kind,
             const TraceContext &context,
             std::uint64_t request_id,
             nixlTime::us_t start,
             nixlTime::us_t end,
             std::span<const Attribute> attrs);

    void
    emitEvent(Name name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              std::span<const Attribute> attrs) override;

    void
    emitEvent(Name name,
              Kind kind,
              const TraceContext &context,
              std::uint64_t request_id,
              nixlTime::us_t at,
              std::span<const Attribute> attrs);

    void
    emitContextSpan(Name name,
                    Kind kind,
                    const TraceContext &context,
                    std::uint64_t request_id,
                    nixlTime::us_t start,
                    nixlTime::us_t end,
                    std::span<const Attribute> attrs) override;

    [[nodiscard]] TraceIdentity
    generateRequestIdentity(const nixl_blob_t &parent) const override;

    [[nodiscard]] TraceContext
    generatePostContext(const TraceContext &request) const override;

    [[nodiscard]] std::optional<TraceContext>
    parseParentContext(const nixl_blob_t &value) const override;

    [[nodiscard]] std::string
    traceparent(const TraceContext &context) const override;

    [[nodiscard]] std::unique_ptr<nixlBackendTraceSink>
    makeSink(std::string backend, std::function<void(std::uint64_t)> on_dropped) override;

private:
    std::vector<std::unique_ptr<TraceBackend>> backends_;
    const double sampleRatio_;
    const random_bytes_fn_t randomBytes_;
    const bool needsTimestamps_;
    mutable std::atomic<bool> contextFailureLogged_ = false;
    mutable std::atomic<bool> unparsableParentLogged_ = false;
};

/** @brief Inputs used to build the composite tracer from enabled backends. */
struct TracerConfig {
    /** @brief Agent name; used e.g. as the NVTX domain name. */
    std::string agentName;
    /** @brief Backend names requested at runtime (e.g. {"nvtx"}). */
    std::vector<std::string> backends;
    /** @brief Head-based sampling probability in [0, 1]; 0 samples nothing. */
    double sampleRatio{defaultSampleRatio};
};

/**
 * @brief Build a tracer from (requested backends) intersected with (backends
 *        compiled in). Returns nullptr when the result is empty, so callers can
 *        cheaply null-check before tracing.
 */
[[nodiscard]] std::unique_ptr<CompositeTracer>
makeTracer(const TracerConfig &config);

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_CORE_TRACE_H
