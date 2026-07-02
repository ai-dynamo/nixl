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

#include <cstdint>
#include <iterator>

namespace nixl::trace::nvtx_internal {
namespace {

    constexpr const char *kRegisteredSpanNames[] = {
        "nixl::registerMem",
        "nixl::deregisterMem",
        "nixl::makeConnection",
        "nixl::makeXferReq",
        "nixl::createXferReq",
        "nixl::postXferReq.write",
        "nixl::postXferReq.read",
        "nixl::genNotif",
        "nixl::getNotifs",
        "nixl::xfer.complete",
        "nixl::loadRemoteMD",
        "nixl::fetchRemoteMD",
        "nixl::prepMemView",
        "nixl::releaseMemView",
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

    [[nodiscard]] nvtxEventAttributes_t
    makeEvent(const nvtxStringHandle_t registered_name, const Kind kind) {
        nvtxEventAttributes_t ev = makeEventBase(kind);
        ev.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        ev.message.registered = registered_name;
        return ev;
    }

    [[nodiscard]] nvtxEventAttributes_t
    makeEvent(const char *ascii_name, const Kind kind) {
        nvtxEventAttributes_t ev = makeEventBase(kind);
        ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
        ev.message.ascii = ascii_name;
        return ev;
    }

    [[nodiscard]] nvtxStringHandle_t
    lookupRegistered(std::string_view name,
                     const std::vector<nvtxStringHandle_t> &handles) noexcept {
        for (std::size_t i = 0; i < std::size(kRegisteredSpanNames); ++i) {
            if (name == kRegisteredSpanNames[i]) {
                return handles[i];
            }
        }
        return nullptr;
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

nvtxEventAttributes_t
eventForName(const std::string_view name,
             const Kind kind,
             const std::vector<nvtxStringHandle_t> &registered_handles,
             std::string &fallback_storage) {
    if (const nvtxStringHandle_t registered = lookupRegistered(name, registered_handles)) {
        return makeEvent(registered, kind);
    }
    fallback_storage.assign(name);
    return makeEvent(fallback_storage.c_str(), kind);
}

} // namespace nixl::trace::nvtx_internal
