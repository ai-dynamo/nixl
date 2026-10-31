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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>

namespace nixl::trace::nvtx_internal {
namespace {

    constexpr const char *kRegisteredSpanNames[] = {
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

    [[nodiscard]] nvtxEventAttributes_t
    makeEventBase(const Kind kind) {
        nvtxEventAttributes_t ev{};
        ev.version = NVTX_VERSION;
        ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        ev.colorType = NVTX_COLOR_ARGB;
        ev.color = colorFor(kind);
        return ev;
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

    [[nodiscard]] constexpr std::uint64_t
    nameHash(const std::string_view name) noexcept {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const char c : name) {
            hash = (hash ^ static_cast<unsigned char>(c)) * 0x100000001b3ULL;
        }
        return hash;
    }

    constexpr std::size_t kNameSlots = 64;
    static_assert(std::size(kRegisteredSpanNames) * 2 <= kNameSlots);

    constexpr auto kNameTable = [] {
        std::array<std::int8_t, kNameSlots> table{};
        table.fill(-1);
        for (std::size_t i = 0; i < std::size(kRegisteredSpanNames); ++i) {
            std::size_t slot = nameHash(kRegisteredSpanNames[i]) % kNameSlots;
            while (table[slot] >= 0) {
                slot = (slot + 1) % kNameSlots;
            }
            table[slot] = static_cast<std::int8_t>(i);
        }
        return table;
    }();

    [[nodiscard]] std::optional<std::size_t>
    registeredIndex(const std::string_view name) noexcept {
        for (std::size_t slot = nameHash(name) % kNameSlots; kNameTable[slot] >= 0;
             slot = (slot + 1) % kNameSlots) {
            const auto index = static_cast<std::size_t>(kNameTable[slot]);
            if (name == kRegisteredSpanNames[index]) {
                return index;
            }
        }
        return std::nullopt;
    }

} // namespace

std::vector<nvtxStringHandle_t>
registerSpanNames(const nvtxDomainHandle_t domain) {
    std::vector<nvtxStringHandle_t> handles;
    handles.reserve(std::size(kRegisteredSpanNames));
    for (const char *name : kRegisteredSpanNames) {
        handles.push_back(nvtxDomainRegisterStringA(domain, name));
    }
    return handles;
}

void
eventForName(NvtxEvent &event,
             const std::string_view name,
             const Kind kind,
             const nvtxDomainHandle_t domain,
             const std::vector<nvtxStringHandle_t> &registered_handles) {
    event.attributes = makeEventBase(kind);
    if (const auto index = registeredIndex(name)) {
        if (registered_handles[*index] != nullptr) {
            setRegistered(event.attributes, registered_handles[*index]);
        } else {
            setAscii(event.attributes, kRegisteredSpanNames[*index]);
        }
        return;
    }
    if (name.size() < event.message.size()) {
        *std::copy(name.begin(), name.end(), event.message.begin()) = '\0';
        setAscii(event.attributes, event.message.data());
        return;
    }
    const std::string owned(name);
    setRegistered(event.attributes, nvtxDomainRegisterStringA(domain, owned.c_str()));
}

} // namespace nixl::trace::nvtx_internal
