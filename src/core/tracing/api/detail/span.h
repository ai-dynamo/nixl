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
#ifndef NIXL_SRC_CORE_TRACING_API_DETAIL_SPAN_H
#define NIXL_SRC_CORE_TRACING_API_DETAIL_SPAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include "tracing/api/attribute.h"
#include "tracing/api/attribute_key.h"
#include "tracing/api/detail/trace_impl.h"
#include "tracing/api/record.h"
#include "tracing/api/trace_identity.h"

namespace nixl::trace {

/**
 * @brief Opaque span identifier. Meaningful on backends that build a DAG
 *        (Chakra Node.id); returned as {0} by backends that do not (NVTX).
 */
struct SpanId {
    std::uint64_t value{0};
};

class Name {
public:
    constexpr Name(Record record) noexcept
        : Name(toStringView(record),
               (static_cast<std::size_t>(record) < recordCount) ?
                   static_cast<std::uint8_t>(record) :
                   noRecord) {}

    template<std::size_t Size>
    consteval Name(const char (&text)[Size]) noexcept : Name(std::string_view(text, Size - 1)) {}

    explicit constexpr Name(std::string_view text) noexcept : Name(text, noRecord) {}

    [[nodiscard]] constexpr std::string_view
    view() const noexcept {
        return {data_, size_};
    }

    [[nodiscard]] constexpr std::optional<Record>
    record() const noexcept {
        if (record_ == noRecord) {
            return std::nullopt;
        }
        return static_cast<Record>(record_);
    }

private:
    static constexpr std::uint8_t noRecord = 0xff;

    constexpr Name(std::string_view text, std::uint8_t record) noexcept
        : data_(text.data()),
          size_(static_cast<std::uint32_t>(text.size())),
          record_(record) {}

    const char *data_;
    std::uint32_t size_;
    std::uint8_t record_;
};

inline constexpr std::size_t maxTraceBackends = 4;

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
        return backends_[0] != nullptr;
    }

    void
    addAttribute(AttributeKey key, std::string_view value);
    void
    addAttribute(AttributeKey key, std::int64_t value);
    void
    addAttribute(AttributeKey key, double value);

    void
    addAttributes(std::span<const Attribute> attrs);

    void
    addCtrlDep(SpanId parent);
    void
    addDataDep(SpanId parent);

    /** @brief Id of the first backend that provides a non-zero id, or {0}
     *         when none do (e.g. NVTX-only). Order-independent across backends. */
    [[nodiscard]] SpanId
    id() const noexcept;

private:
    friend Span
    detail::beginSpan(Tracer &tracer, Name name, Kind kind);
    friend Span
    detail::beginContextSpan(Tracer &tracer,
                             Name name,
                             Kind kind,
                             const TraceContext &context,
                             std::uint64_t request_id);

    template<typename Call>
    void
    forEachBackend(const Call &call) const;

    std::array<std::unique_ptr<SpanBackend>, maxTraceBackends> backends_;
};


} // namespace nixl::trace

#endif
