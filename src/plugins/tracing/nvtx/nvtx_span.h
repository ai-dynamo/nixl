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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "nvtx_payload_schemas.h"
#include "tracing/trace.h"

namespace nixl::trace::nvtx_internal {

/** @brief One active NVTX range; typed attributes are attached on pop. */
class NvtxSpan final : public SpanBackend {
public:
    NvtxSpan(nvtxDomainHandle_t domain, PayloadSchemaIds schema_ids) noexcept;

    ~NvtxSpan() override;

    void
    addAttribute(std::string_view key, std::string_view value) override;
    void
    addAttribute(std::string_view key, std::int64_t value) override;
    void
    addAttribute(std::string_view key, double value) override;

    void
    addCtrlDep(SpanId) override {}

    void
    addDataDep(SpanId) override {}

    [[nodiscard]] SpanId
    id() const noexcept override {
        return {};
    }

private:
    struct StoredPayload;

    void
    emitInt64Attr(std::string_view key, std::int64_t value);
    void
    emitDoubleAttr(std::string_view key, double value);
    void
    emitStrAttr(std::string_view key, std::string_view value);

    nvtxDomainHandle_t domain_;
    PayloadSchemaIds schemaIds_;
    std::vector<std::unique_ptr<StoredPayload>> payloads_;
};

} // namespace nixl::trace::nvtx_internal

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_SPAN_H
