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

    struct Int64Attr {
        const char *key;
        std::int64_t value;
    };

    struct DoubleAttr {
        const char *key;
        double value;
    };

    struct StrAttr {
        const char *key;
        const char *value;
    };

    [[nodiscard]] nvtxPayloadSchemaAttr_t
    makeStaticSchemaAttr(const char *name,
                         nvtxPayloadSchemaEntry_t *entries,
                         const std::size_t num_entries,
                         const std::size_t static_size) {
        nvtxPayloadSchemaAttr_t attr{};
        attr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
            NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
        attr.name = name;
        attr.type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
        attr.entries = entries;
        attr.numEntries = num_entries;
        attr.payloadStaticSize = static_size;
        return attr;
    }

    nvtxPayloadSchemaEntry_t kInt64AttrSchemaEntries[] = {
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
         "key",
         nullptr,
         0,
         offsetof(Int64Attr, key),
         nullptr,
         nullptr},
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_INT64,
         "value",
         nullptr,
         0,
         offsetof(Int64Attr, value),
         nullptr,
         nullptr},
    };

    nvtxPayloadSchemaEntry_t kDoubleAttrSchemaEntries[] = {
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
         "key",
         nullptr,
         0,
         offsetof(DoubleAttr, key),
         nullptr,
         nullptr},
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE,
         "value",
         nullptr,
         0,
         offsetof(DoubleAttr, value),
         nullptr,
         nullptr},
    };

    nvtxPayloadSchemaEntry_t kStrAttrSchemaEntries[] = {
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
         "key",
         nullptr,
         0,
         offsetof(StrAttr, key),
         nullptr,
         nullptr},
        {0,
         NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
         "value",
         nullptr,
         0,
         offsetof(StrAttr, value),
         nullptr,
         nullptr},
    };

    const nvtxPayloadSchemaAttr_t kInt64AttrSchemaAttr =
        makeStaticSchemaAttr("nixl.trace.int64_attr",
                             kInt64AttrSchemaEntries,
                             std::size(kInt64AttrSchemaEntries),
                             sizeof(Int64Attr));

    const nvtxPayloadSchemaAttr_t kDoubleAttrSchemaAttr =
        makeStaticSchemaAttr("nixl.trace.double_attr",
                             kDoubleAttrSchemaEntries,
                             std::size(kDoubleAttrSchemaEntries),
                             sizeof(DoubleAttr));

    const nvtxPayloadSchemaAttr_t kStrAttrSchemaAttr =
        makeStaticSchemaAttr("nixl.trace.str_attr",
                             kStrAttrSchemaEntries,
                             std::size(kStrAttrSchemaEntries),
                             sizeof(StrAttr));

} // namespace

PayloadSchemaIds
registerPayloadSchemas(const nvtxDomainHandle_t domain) {
    PayloadSchemaIds ids;
    ids.int64_attr = nvtxPayloadSchemaRegister(domain, &kInt64AttrSchemaAttr);
    ids.double_attr = nvtxPayloadSchemaRegister(domain, &kDoubleAttrSchemaAttr);
    ids.str_attr = nvtxPayloadSchemaRegister(domain, &kStrAttrSchemaAttr);
    return ids;
}

} // namespace nixl::trace::nvtx_internal
