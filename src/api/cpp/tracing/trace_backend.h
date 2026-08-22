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
#ifndef NIXL_SRC_API_CPP_TRACING_TRACE_BACKEND_H
#define NIXL_SRC_API_CPP_TRACING_TRACE_BACKEND_H

#include <cstdint>
#include <memory>
#include <string_view>

namespace nixl::trace {

/**
 * @brief Operation kind. Aligns 1:1 with the Chakra NodeType vocabulary and is
 *        used as a color/label hint by tracing backends.
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
 *        and returned as {0} by backends that do not.
 */
struct SpanId {
    std::uint64_t value{0};
};

/**
 * @brief A single active span within one backend. Its destructor ends the
 *        backend-specific range or duration.
 */
class SpanBackend {
public:
    /** @brief Ends the backend-specific span and releases its resources. */
    virtual ~SpanBackend() = default;

    /**
     * @brief Adds a string attribute to this span.
     * @param key Attribute name, consumed during the call.
     * @param value Attribute value, consumed during the call.
     */
    virtual void
    addAttribute(std::string_view key, std::string_view value) = 0;

    /**
     * @brief Adds an integer attribute to this span.
     * @param key Attribute name, consumed during the call.
     * @param value Attribute value.
     */
    virtual void
    addAttribute(std::string_view key, std::int64_t value) = 0;

    /**
     * @brief Adds a floating-point attribute to this span.
     * @param key Attribute name, consumed during the call.
     * @param value Attribute value.
     */
    virtual void
    addAttribute(std::string_view key, double value) = 0;

    /**
     * @brief Records a control dependency on another span.
     * @param parent Identifier of the parent span.
     */
    virtual void
    addCtrlDep(SpanId parent) = 0;

    /**
     * @brief Records a data dependency on another span.
     * @param parent Identifier of the parent span.
     */
    virtual void
    addDataDep(SpanId parent) = 0;

    /**
     * @brief Returns this span's backend-defined identifier.
     * @return A non-zero identifier when supported, otherwise `{0}`.
     */
    [[nodiscard]] virtual SpanId
    id() const noexcept = 0;
};

/** @brief One enabled backend type (NVTX, Chakra, ...). */
class TraceBackend {
public:
    /** @brief Releases the backend after all of its spans have been destroyed. */
    virtual ~TraceBackend() = default;

    /**
     * @brief Starts a backend-specific span.
     * @param name Span name, consumed during the call.
     * @param kind Operation kind used as a backend-specific label or hint.
     * @return An owned active span, or `nullptr` when this backend does not
     *         produce a span for the operation. Destroying it ends the span.
     */
    [[nodiscard]] virtual std::unique_ptr<SpanBackend>
    beginSpan(std::string_view name, Kind kind) = 0;

    /**
     * @brief Emits an instantaneous marker.
     * @param name Marker name, consumed during the call.
     * @param kind Operation kind used as a backend-specific label or hint.
     */
    virtual void
    mark(std::string_view name, Kind kind) = 0;

    /**
     * @brief Pushes a correlation identifier onto this thread's backend context.
     * @param id Identifier to associate with subsequent backend events.
     */
    virtual void
    pushCorrelationId(std::uint64_t id) = 0;

    /**
     * @brief Pops the most recently pushed correlation identifier.
     *
     * Calls must balance successful calls to pushCorrelationId() on the same
     * thread.
     */
    virtual void
    popCorrelationId() = 0;

    /**
     * @brief Returns the backend's stable name.
     * @return A view that remains valid for this object's lifetime.
     */
    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;
};

} // namespace nixl::trace

#endif // NIXL_SRC_API_CPP_TRACING_TRACE_BACKEND_H
