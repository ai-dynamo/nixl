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

#include <algorithm>
#include <new>
#include <vector>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtPayload.h>

namespace nixl::trace::nvtx_internal {
namespace {

    constexpr std::size_t maxPooledSpans = 32;

    class SpanPool {
    public:
        constexpr SpanPool() noexcept = default;
        ~SpanPool();

        SpanPool(const SpanPool &) = delete;
        SpanPool &
        operator=(const SpanPool &) = delete;

        [[nodiscard]] void *
        take() noexcept;

        [[nodiscard]] bool
        give(void *memory) noexcept;

    private:
        struct FreeSpan {
            FreeSpan *next;
        };

        FreeSpan *head_ = nullptr;
        std::size_t size_ = 0;
    };

    constinit thread_local bool pool_gone = false;
    constinit thread_local SpanPool pool;

    SpanPool::~SpanPool() {
        pool_gone = true;
        while (head_ != nullptr) {
            FreeSpan *const next = head_->next;
            ::operator delete(head_);
            head_ = next;
        }
    }

    void *
    SpanPool::take() noexcept {
        if (head_ == nullptr) {
            return nullptr;
        }
        FreeSpan *const span = head_;
        head_ = span->next;
        --size_;
        return span;
    }

    bool
    SpanPool::give(void *const memory) noexcept {
        if (size_ == maxPooledSpans) {
            return false;
        }
        auto *const span = static_cast<FreeSpan *>(memory);
        span->next = head_;
        head_ = span;
        ++size_;
        return true;
    }

} // namespace

NvtxSpan::NvtxSpan(const nvtxDomainHandle_t domain, const PayloadSchemaIds schema_ids) noexcept
    : domain_(domain),
      schemaIds_(schema_ids) {}

NvtxSpan::~NvtxSpan() {
    if (overflow_.empty()) {
        if (inlineCount_ == 0) {
            nvtxDomainRangePop(domain_);
        } else {
            nvtxRangePopPayload(domain_, inlineRefs_.data(), inlineCount_);
        }
        return;
    }

    std::vector<nvtxPayloadData_t> refs;
    refs.reserve(inlineCount_ + overflow_.size());
    refs.insert(refs.end(), inlineRefs_.begin(), inlineRefs_.begin() + inlineCount_);
    for (const StoredPayload &stored : overflow_) {
        refs.push_back(stored.data);
    }
    nvtxRangePopPayload(domain_, refs.data(), refs.size());
}

void *
NvtxSpan::operator new(const std::size_t size) {
    if (!pool_gone) {
        if (void *const memory = pool.take()) {
            return memory;
        }
    }
    return ::operator new(size);
}

void
NvtxSpan::operator delete(void *const memory) noexcept {
    if (memory == nullptr) {
        return;
    }
    if (pool_gone || !pool.give(memory)) {
        ::operator delete(memory);
    }
}

bool
NvtxSpan::fitsInline(const std::size_t value_size) const noexcept {
    return overflow_.empty() && (inlineCount_ < inlineAttributes) && (value_size < inlineValueSize);
}

NvtxSpan::InlinePayload &
NvtxSpan::nextInline(const AttributeKey key, const std::uint64_t schema_id) noexcept {
    InlinePayload &payload = inline_[inlineCount_];
    payload.record.key = key.c_str();
    inlineRefs_[inlineCount_] = {schema_id, sizeof(AttrPayload), &payload.record};
    ++inlineCount_;
    return payload;
}

NvtxSpan::StoredPayload &
NvtxSpan::nextStored(const AttributeKey key, const std::uint64_t schema_id) {
    StoredPayload &stored = overflow_.emplace_back();
    stored.record.key = key.c_str();
    stored.data = {schema_id, sizeof(AttrPayload), &stored.record};
    return stored;
}

AttrPayload &
NvtxSpan::addPayload(const AttributeKey key, const std::uint64_t schema_id) {
    return fitsInline(0) ? nextInline(key, schema_id).record : nextStored(key, schema_id).record;
}

void
NvtxSpan::addAttribute(const AttributeKey key, const std::string_view value) {
    if (fitsInline(value.size())) {
        InlinePayload &payload = nextInline(key, schemaIds_.string_attr);
        *std::copy(value.begin(), value.end(), payload.value.begin()) = '\0';
        payload.record.value.string_value = payload.value.data();
        return;
    }
    StoredPayload &stored = nextStored(key, schemaIds_.string_attr);
    stored.string_value.assign(value);
    stored.record.value.string_value = stored.string_value.c_str();
}

void
NvtxSpan::addAttribute(const AttributeKey key, const std::int64_t value) {
    addPayload(key, schemaIds_.int64_attr).value.int64_value = value;
}

void
NvtxSpan::addAttribute(const AttributeKey key, const double value) {
    addPayload(key, schemaIds_.double_attr).value.double_value = value;
}

} // namespace nixl::trace::nvtx_internal
