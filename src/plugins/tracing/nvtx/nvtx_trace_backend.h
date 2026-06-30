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
#ifndef NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_TRACE_BACKEND_H
#define NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_TRACE_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "nvtx_payload_schemas.h"
#include "tracing/trace.h"

namespace nixl::trace::nvtx_internal {

class NvtxTraceBackend final : public TraceBackend {
public:
    explicit NvtxTraceBackend(std::string_view domain);
    ~NvtxTraceBackend() override;

    [[nodiscard]] std::unique_ptr<SpanBackend>
    beginSpan(std::string_view name, Kind kind) override;

    void
    mark(std::string_view name, Kind kind) override;

    void
    pushCorrelationId(std::uint64_t) override {}

    void
    popCorrelationId() override {}

    [[nodiscard]] std::string_view
    name() const noexcept override {
        return "nvtx";
    }

private:
    const std::string domainName_;
    nvtxDomainHandle_t domain_;
    PayloadSchemaIds schemaIds_;
    std::vector<nvtxStringHandle_t> registeredHandles_;
};

} // namespace nixl::trace::nvtx_internal

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_TRACE_BACKEND_H
