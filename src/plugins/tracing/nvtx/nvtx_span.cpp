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

#include "nvtx_span.h"

#include <vector>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtPayload.h>

namespace nixl::trace::nvtx_internal {

NvtxSpan::NvtxSpan(const nvtxDomainHandle_t domain, const PayloadSchemaIds schema_ids) noexcept
    : domain_(domain),
      schemaIds_(schema_ids) {}

NvtxSpan::~NvtxSpan() {
    if (payloads_.empty()) {
        nvtxDomainRangePop(domain_);
        return;
    }

    std::vector<nvtxPayloadData_t> refs;
    refs.reserve(payloads_.size());
    for (const StoredPayload &stored : payloads_) {
        refs.push_back(stored.data);
    }
    nvtxRangePopPayload(domain_, refs.data(), refs.size());
}

NvtxSpan::StoredPayload &
NvtxSpan::addPayload(const std::string_view key) {
    StoredPayload &payload = payloads_.emplace_back();
    payload.key.assign(key);
    payload.record.key = payload.key.c_str();
    return payload;
}

void
NvtxSpan::addAttribute(const std::string_view key, const std::string_view value) {
    StoredPayload &payload = addPayload(key);
    payload.string_value.assign(value);
    payload.record.value.string_value = payload.string_value.c_str();
    payload.data = {schemaIds_.string_attr, sizeof(AttrPayload), &payload.record};
}

void
NvtxSpan::addAttribute(const std::string_view key, const std::int64_t value) {
    StoredPayload &payload = addPayload(key);
    payload.record.value.int64_value = value;
    payload.data = {schemaIds_.int64_attr, sizeof(AttrPayload), &payload.record};
}

void
NvtxSpan::addAttribute(const std::string_view key, const double value) {
    StoredPayload &payload = addPayload(key);
    payload.record.value.double_value = value;
    payload.data = {schemaIds_.double_attr, sizeof(AttrPayload), &payload.record};
}

} // namespace nixl::trace::nvtx_internal
