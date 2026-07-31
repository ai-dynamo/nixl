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
    pushCorrelationId(std::uint64_t id) = 0;
    virtual void
    popCorrelationId() = 0;

    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;
};

} // namespace nixl::trace

#endif // NIXL_SRC_API_CPP_TRACING_TRACE_BACKEND_H
