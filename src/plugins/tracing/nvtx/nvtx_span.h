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
#ifndef NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_SPAN_H
#define NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_SPAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <string_view>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtPayload.h>

#include "nvtx_payload_schemas.h"
#include "tracing/trace.h"

namespace nixl::trace::nvtx_internal {

/**
 * @brief One active NVTX range. Attributes are buffered as typed payloads and
 *        attached to the range when it is popped in the destructor.
 */
class NvtxSpan final : public SpanBackend {
public:
    NvtxSpan(nvtxDomainHandle_t domain, PayloadSchemaIds schema_ids) noexcept;

    ~NvtxSpan() override;

    [[nodiscard]] static void *
    operator new(std::size_t size);

    static void
    operator delete(void *memory) noexcept;

    void
    addAttribute(AttributeKey key, std::string_view value) override;
    void
    addAttribute(AttributeKey key, std::int64_t value) override;
    void
    addAttribute(AttributeKey key, double value) override;

    void
    addAttributes(std::span<const Attribute> attrs) override;

    void
    addCtrlDep(SpanId) override {}

    void
    addDataDep(SpanId) override {}

    [[nodiscard]] SpanId
    id() const noexcept override {
        return {};
    }

    static constexpr std::size_t inlineAttributes = 12;
    static constexpr std::size_t inlineValueSize = 64;

private:
    struct InlinePayload {
        std::array<char, inlineValueSize> value;
        AttrPayload record;
    };

    // Backing storage for an attribute that does not fit inline. Held in a
    // std::list so nodes never move: record.value.string_value points into
    // string_value and must stay valid until the range is popped.
    struct StoredPayload {
        std::string string_value;
        AttrPayload record{};
        nvtxPayloadData_t data{};
    };

    [[nodiscard]] bool
    fitsInline(std::size_t value_size) const noexcept;

    [[nodiscard]] InlinePayload &
    nextInline(AttributeKey key, std::uint64_t schema_id) noexcept;

    [[nodiscard]] StoredPayload &
    nextStored(AttributeKey key, std::uint64_t schema_id);

    [[nodiscard]] AttrPayload &
    addPayload(AttributeKey key, std::uint64_t schema_id);

    nvtxDomainHandle_t domain_;
    PayloadSchemaIds schemaIds_;
    std::size_t inlineCount_ = 0;
    std::array<InlinePayload, inlineAttributes> inline_;
    std::array<nvtxPayloadData_t, inlineAttributes> inlineRefs_;
    std::list<StoredPayload> overflow_;
};

} // namespace nixl::trace::nvtx_internal

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_SPAN_H
