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

#include <utility>

#include <nvtx3/nvToolsExt.h>
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

} // namespace

struct NvtxSpan::StoredPayload {
    nvtxPayloadData_t data{};
    std::string key_storage;
    std::string value_storage;
    Int64Attr int64{};
    DoubleAttr dbl{};
    StrAttr str{};
};

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
    for (const auto &stored : payloads_) {
        refs.push_back(stored->data);
    }
    nvtxRangePopPayload(domain_, refs.data(), refs.size());
}

void
NvtxSpan::addAttribute(const std::string_view key, const std::string_view value) {
    emitStrAttr(key, value);
}

void
NvtxSpan::addAttribute(const std::string_view key, const std::int64_t value) {
    emitInt64Attr(key, value);
}

void
NvtxSpan::addAttribute(const std::string_view key, const double value) {
    emitDoubleAttr(key, value);
}

void
NvtxSpan::emitInt64Attr(const std::string_view key, const std::int64_t value) {
    auto stored = std::make_unique<StoredPayload>();
    stored->key_storage.assign(key);
    stored->int64.key = stored->key_storage.c_str();
    stored->int64.value = value;
    stored->data = {schemaIds_.int64_attr, sizeof(Int64Attr), &stored->int64};
    payloads_.push_back(std::move(stored));
}

void
NvtxSpan::emitDoubleAttr(const std::string_view key, const double value) {
    auto stored = std::make_unique<StoredPayload>();
    stored->key_storage.assign(key);
    stored->dbl.key = stored->key_storage.c_str();
    stored->dbl.value = value;
    stored->data = {schemaIds_.double_attr, sizeof(DoubleAttr), &stored->dbl};
    payloads_.push_back(std::move(stored));
}

void
NvtxSpan::emitStrAttr(const std::string_view key, const std::string_view value) {
    auto stored = std::make_unique<StoredPayload>();
    stored->key_storage.assign(key);
    stored->value_storage.assign(value);
    stored->str.key = stored->key_storage.c_str();
    stored->str.value = stored->value_storage.c_str();
    stored->data = {schemaIds_.str_attr, sizeof(StrAttr), &stored->str};
    payloads_.push_back(std::move(stored));
}

} // namespace nixl::trace::nvtx_internal
