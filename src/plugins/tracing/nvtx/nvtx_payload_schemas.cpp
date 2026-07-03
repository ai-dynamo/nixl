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

#include "nvtx_payload_schemas.h"

#include <cstddef>
#include <iterator>

#include <nvtx3/nvToolsExtPayload.h>

namespace nixl::trace::nvtx_internal {
namespace {

    // All three schemas describe the shared AttrPayload layout: a "key" C-string
    // followed by a "value" whose only difference is the NVTX entry type.
    [[nodiscard]] nvtxPayloadSchemaEntry_t
    keyEntry() {
        return {0,
                NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
                "key",
                nullptr,
                0,
                offsetof(AttrPayload, key),
                nullptr,
                nullptr};
    }

    [[nodiscard]] nvtxPayloadSchemaEntry_t
    valueEntry(const std::uint64_t entry_type) {
        return {0, entry_type, "value", nullptr, 0, offsetof(AttrPayload, value), nullptr, nullptr};
    }

    [[nodiscard]] nvtxPayloadSchemaAttr_t
    makeStaticSchemaAttr(const char *name,
                         nvtxPayloadSchemaEntry_t *entries,
                         const std::size_t num_entries) {
        nvtxPayloadSchemaAttr_t attr{};
        attr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
        attr.name = name;
        attr.type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
        attr.entries = entries;
        attr.numEntries = num_entries;
        attr.payloadStaticSize = sizeof(AttrPayload);
        return attr;
    }

    [[nodiscard]] std::uint64_t
    registerSchema(const nvtxDomainHandle_t domain,
                   const char *name,
                   const std::uint64_t value_type) {
        nvtxPayloadSchemaEntry_t entries[] = {keyEntry(), valueEntry(value_type)};
        const nvtxPayloadSchemaAttr_t attr =
            makeStaticSchemaAttr(name, entries, std::size(entries));
        return nvtxPayloadSchemaRegister(domain, &attr);
    }

} // namespace

PayloadSchemaIds
registerPayloadSchemas(const nvtxDomainHandle_t domain) {
    PayloadSchemaIds ids;
    ids.int64_attr = registerSchema(domain, "nixl.trace.int64_attr", NVTX_PAYLOAD_ENTRY_TYPE_INT64);
    ids.double_attr =
        registerSchema(domain, "nixl.trace.double_attr", NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE);
    ids.string_attr =
        registerSchema(domain, "nixl.trace.string_attr", NVTX_PAYLOAD_ENTRY_TYPE_CSTRING);
    return ids;
}

} // namespace nixl::trace::nvtx_internal
