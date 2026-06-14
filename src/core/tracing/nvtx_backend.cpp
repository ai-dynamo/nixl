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

#include "nvtx_backend.h"

#include <cstdint>
#include <string>

#include <nvtx3/nvToolsExt.h>

namespace nixl::trace {
namespace {

    // Map an operation kind to a stable ARGB color so the Nsight timeline is
    // visually distinguishable. An explicit (non-zero) Color overrides the default.
    [[nodiscard]] constexpr std::uint32_t
    colorFor(const Kind kind, const Color c) noexcept {
        if (c.r != 0 || c.g != 0 || c.b != 0) {
            return 0xFF000000u | (static_cast<std::uint32_t>(c.r) << 16) |
                (static_cast<std::uint32_t>(c.g) << 8) | static_cast<std::uint32_t>(c.b);
        }
        switch (kind) {
        case Kind::CommSend:
            return 0xFF2E7D32u; // green
        case Kind::CommRecv:
            return 0xFF1565C0u; // blue
        case Kind::CommColl:
            return 0xFF6A1B9Au; // purple
        case Kind::MemoryR:
            return 0xFFEF6C00u; // orange
        case Kind::MemoryW:
            return 0xFFF9A825u; // amber
        case Kind::Compute:
            return 0xFF00838Fu; // teal
        case Kind::Metadata:
            return 0xFF757575u; // gray
        case Kind::Generic:
        default:
            return 0xFF455A64u; // blue-gray
        }
    }

    // NVTX copies the message at submit time, so a null-terminated C string that
    // outlives the call is sufficient.
    nvtxEventAttributes_t
    makeEvent(const char *msg, Kind kind, Color color) {
        nvtxEventAttributes_t ev{};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = colorFor(kind, color);
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = msg;
        return ev;
    }

    // A single NVTX range. The matching push happens in the backend's beginSpan();
    // the pop happens here on destruction (per-thread, per-domain LIFO).
    class NvtxSpan final : public SpanBackend {
    public:
        explicit NvtxSpan(nvtxDomainHandle_t domain) noexcept : domain_(domain) {}

        ~NvtxSpan() override {
            nvtxDomainRangePop(domain_);
        }

        // An NVTX range's message/color are fixed at push time, so attributes
        // (added afterwards) are surfaced as "key=value" marks inside the range.
        // They show on the Nsight timeline; a structured NVTX payload schema is a
        // future enhancement. Dependencies have no NVTX representation (no-op).
        void
        addAttribute(std::string_view key, std::string_view value) override {
            emitAttr(key, value);
        }

        void
        addAttribute(std::string_view key, std::int64_t value) override {
            emitAttr(key, std::to_string(value));
        }

        void
        addAttribute(std::string_view key, double value) override {
            emitAttr(key, std::to_string(value));
        }

        void
        addCtrlDep(SpanId) override {}

        void
        addDataDep(SpanId) override {}

        [[nodiscard]] SpanId
        id() const noexcept override {
            return {};
        }

    private:
        // Surface one attribute as a "key=value" mark inside the active range.
        void
        emitAttr(std::string_view key, std::string_view value) {
            std::string kv;
            kv.reserve(key.size() + value.size() + 1);
            kv.append(key).append("=").append(value);
            const nvtxEventAttributes_t ev = makeEvent(kv.c_str(), Kind::Metadata, Color{});
            nvtxDomainMarkEx(domain_, &ev);
        }

        nvtxDomainHandle_t domain_;
    };

    class NvtxTraceBackend final : public TraceBackend {
    public:
        explicit NvtxTraceBackend(std::string_view domain)
            : domainName_(domain),
              domain_(nvtxDomainCreateA(domainName_.c_str())) {}

        ~NvtxTraceBackend() override {
            nvtxDomainDestroy(domain_);
        }

        [[nodiscard]] std::unique_ptr<SpanBackend>
        beginSpan(std::string_view name, Kind kind, Color color) override {
            const std::string msg(name);
            const nvtxEventAttributes_t ev = makeEvent(msg.c_str(), kind, color);
            nvtxDomainRangePushEx(domain_, &ev);
            return std::make_unique<NvtxSpan>(domain_);
        }

        void
        mark(std::string_view name, Kind kind) override {
            const std::string msg(name);
            const nvtxEventAttributes_t ev = makeEvent(msg.c_str(), kind, Color{});
            nvtxDomainMarkEx(domain_, &ev);
        }

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
    };

} // namespace

std::unique_ptr<TraceBackend>
makeNvtxBackend(std::string_view domain) {
    return std::make_unique<NvtxTraceBackend>(domain);
}

} // namespace nixl::trace
