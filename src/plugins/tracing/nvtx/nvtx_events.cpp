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

#include "nvtx_events.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>

namespace nixl::trace::nvtx_internal {
namespace {

    constexpr std::string_view kRegisteredSpanNames[] = {
        "nixl::registerMem",      "nixl::deregisterMem",   "nixl::makeConnection",
        "nixl::makeXferReq",      "nixl::createXferReq",   "nixl::postXferReq.write",
        "nixl::postXferReq.read", "nixl::genNotif",        "nixl::getNotifs",
        "nixl::xfer.complete",    "nixl::loadRemoteMD",    "nixl::fetchRemoteMD",
        "nixl::prepMemView",      "nixl::releaseMemView",  "nixl::submit",
        "nixl::wire.submitted",   "nixl::wire.completed",  "nixl::notif.sent",
        "nixl::notif.received",   "nixl::remote.observed", "nixl::span",
        "nixl::wire.inflight",    "nixl::phase",           "nixl::xfer.error",
        "nixl::releaseXferReq",
    };

    [[nodiscard]] constexpr std::uint32_t
    colorFor(const Kind kind) noexcept {
        switch (kind) {
        case Kind::Generic:
            return 0xFF455A64u; // blue-gray
        case Kind::Compute:
            return 0xFF00838Fu; // teal
        case Kind::MemoryR:
            return 0xFFEF6C00u; // orange
        case Kind::MemoryW:
            return 0xFFF9A825u; // amber
        case Kind::CommSend:
            return 0xFF2E7D32u; // green
        case Kind::CommRecv:
            return 0xFF1565C0u; // blue
        case Kind::CommColl:
            return 0xFF6A1B9Au; // purple
        case Kind::Metadata:
            return 0xFF757575u; // gray
        }
        return 0xFF455A64u; // blue-gray fallback for out-of-range Kind values
    }

    void
    setEventBase(nvtxEventAttributes_t &ev, const Kind kind) noexcept {
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.category = 0;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = colorFor(kind);
        ev.payloadType = NVTX_PAYLOAD_UNKNOWN;
        ev.reserved0 = 0;
        ev.payload.ullValue = 0;
    }

    void
    setRegistered(nvtxEventAttributes_t &ev, const nvtxStringHandle_t registered_name) noexcept {
        ev.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        ev.message.registered = registered_name;
    }

    void
    setAscii(nvtxEventAttributes_t &ev, const char *text) noexcept {
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = text;
    }

    constexpr std::size_t kWordSize = sizeof(std::uint64_t);
    constexpr std::size_t kMaxRegisteredName = 3 * kWordSize;

    [[nodiscard]] constexpr std::uint64_t
    wordAt(const std::string_view name, const std::size_t offset) noexcept {
        const char *const bytes = name.data() + offset;
        if (std::is_constant_evaluated()) {
            std::uint64_t word = 0;
            for (std::size_t i = 0; i < kWordSize; ++i) {
                const std::size_t shift =
                    (std::endian::native == std::endian::little) ? i : (kWordSize - 1 - i);
                word |= std::uint64_t{static_cast<unsigned char>(bytes[i])} << (8 * shift);
            }
            return word;
        }
        std::uint64_t word;
        std::memcpy(&word, bytes, kWordSize);
        return word;
    }

    struct NameWords {
        std::uint64_t head;
        std::uint64_t middle;
        std::uint64_t tail;
        std::size_t size;

        [[nodiscard]] constexpr bool
        operator==(const NameWords &) const noexcept = default;
    };

    [[nodiscard]] constexpr NameWords
    nameWords(const std::string_view name) noexcept {
        const std::size_t tail = name.size() - kWordSize;
        return {wordAt(name, 0),
                wordAt(name, std::min(kWordSize, tail)),
                wordAt(name, tail),
                name.size()};
    }

    constexpr unsigned kNameSlotBits = 6;
    constexpr std::size_t kNameSlots = std::size_t{1} << kNameSlotBits;
    static_assert(std::size(kRegisteredSpanNames) * 2 <= kNameSlots);
    static_assert(std::ranges::all_of(kRegisteredSpanNames, [](const std::string_view name) {
        return (name.size() >= kWordSize) && (name.size() <= kMaxRegisteredName);
    }));

    [[nodiscard]] constexpr std::size_t
    firstSlot(const NameWords &words) noexcept {
        return ((words.tail ^ words.size) * 0x9e3779b97f4a7c15ULL) >> (64 - kNameSlotBits);
    }

    constexpr auto kRegisteredWords = [] {
        std::array<NameWords, std::size(kRegisteredSpanNames)> words{};
        for (std::size_t i = 0; i < words.size(); ++i) {
            words[i] = nameWords(kRegisteredSpanNames[i]);
        }
        return words;
    }();

    constexpr auto kNameTable = [] {
        std::array<std::int8_t, kNameSlots> table{};
        table.fill(-1);
        for (std::size_t i = 0; i < kRegisteredWords.size(); ++i) {
            std::size_t slot = firstSlot(kRegisteredWords[i]);
            while (table[slot] >= 0) {
                slot = (slot + 1) % kNameSlots;
            }
            table[slot] = static_cast<std::int8_t>(i);
        }
        return table;
    }();

    [[nodiscard]] std::optional<std::size_t>
    registeredIndex(const std::string_view name) noexcept {
        if ((name.size() < kWordSize) || (name.size() > kMaxRegisteredName)) {
            return std::nullopt;
        }
        const NameWords words = nameWords(name);
        for (std::size_t slot = firstSlot(words); kNameTable[slot] >= 0;
             slot = (slot + 1) % kNameSlots) {
            const auto index = static_cast<std::size_t>(kNameTable[slot]);
            if (kRegisteredWords[index] == words) {
                return index;
            }
        }
        return std::nullopt;
    }

    [[gnu::noinline]] nvtxStringHandle_t
    registerName(const nvtxDomainHandle_t domain, const std::string_view name) {
        const std::string owned(name);
        return nvtxDomainRegisterStringA(domain, owned.c_str());
    }

} // namespace

std::vector<nvtxStringHandle_t>
registerSpanNames(const nvtxDomainHandle_t domain) {
    std::vector<nvtxStringHandle_t> handles;
    handles.reserve(std::size(kRegisteredSpanNames));
    for (const std::string_view name : kRegisteredSpanNames) {
        handles.push_back(nvtxDomainRegisterStringA(domain, name.data()));
    }
    return handles;
}

void
eventForName(NvtxEvent &event,
             const std::string_view name,
             const Kind kind,
             const nvtxDomainHandle_t domain,
             const std::vector<nvtxStringHandle_t> &registered_handles) {
    setEventBase(event.attributes, kind);
    if (const auto index = registeredIndex(name)) {
        if (registered_handles[*index] != nullptr) {
            setRegistered(event.attributes, registered_handles[*index]);
        } else {
            setAscii(event.attributes, kRegisteredSpanNames[*index].data());
        }
        return;
    }
    if (name.size() < event.message.size()) {
        *std::copy(name.begin(), name.end(), event.message.begin()) = '\0';
        setAscii(event.attributes, event.message.data());
        return;
    }
    setRegistered(event.attributes, registerName(domain, name));
}

} // namespace nixl::trace::nvtx_internal
