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
#ifndef NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_PAYLOAD_SCHEMAS_H
#define NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_PAYLOAD_SCHEMAS_H

#include <cstdint>

#include <nvtx3/nvToolsExt.h>

namespace nixl::trace::nvtx_internal {

/**
 * @brief Record backing every typed attribute payload: a key plus a value at a
 *        fixed offset. Each schema (int64/double/string) describes the value
 *        with its own NVTX entry type but identical offsets, so one struct backs
 *        all three (only one value member is live per record).
 */
struct AttrPayload {
    const char *key;

    union {
        std::int64_t int64_value;
        double double_value;
        const char *string_value;
    } value;
};

/** @brief Domain-registered schema IDs for typed span attribute payloads. */
struct PayloadSchemaIds {
    std::uint64_t int64_attr{};
    std::uint64_t double_attr{};
    std::uint64_t string_attr{};
};

/** @brief Register the static attribute payload schemas on @p domain. */
[[nodiscard]] PayloadSchemaIds
registerPayloadSchemas(nvtxDomainHandle_t domain);

} // namespace nixl::trace::nvtx_internal

#endif // NIXL_SRC_PLUGINS_TRACING_NVTX_NVTX_PAYLOAD_SCHEMAS_H
