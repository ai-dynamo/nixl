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
#ifndef NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H
#define NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nixl::trace {

/** @brief Values cross the trace plugin interface: add new keys at the end. */
enum class AttributeKey : std::uint16_t {
    Bytes,
    DescCount,
    MemType,
    NixlBackend,
    NixlEndUs,
    NixlPhaseLabel,
    NixlPhaseTimestampUs,
    NixlRemoteAgent,
    NixlSpanLabel,
    NixlStartUs,
    NixlStatus,
    NixlXferExecution,
    NixlXferRequestId,
    RemoteAgent,
};

inline constexpr std::size_t attributeKeyCount =
    static_cast<std::size_t>(AttributeKey::RemoteAgent) + 1;

[[nodiscard]] constexpr std::string_view
toStringView(AttributeKey key) noexcept {
    switch (key) {
    case AttributeKey::Bytes:
        return "bytes";
    case AttributeKey::DescCount:
        return "desc_count";
    case AttributeKey::MemType:
        return "mem_type";
    case AttributeKey::NixlBackend:
        return "nixl.backend";
    case AttributeKey::NixlEndUs:
        return "nixl.end_us";
    case AttributeKey::NixlPhaseLabel:
        return "nixl.phase.label";
    case AttributeKey::NixlPhaseTimestampUs:
        return "nixl.phase.timestamp_us";
    case AttributeKey::NixlRemoteAgent:
        return "nixl.remote_agent";
    case AttributeKey::NixlSpanLabel:
        return "nixl.span.label";
    case AttributeKey::NixlStartUs:
        return "nixl.start_us";
    case AttributeKey::NixlStatus:
        return "nixl.status";
    case AttributeKey::NixlXferExecution:
        return "nixl.xfer_execution";
    case AttributeKey::NixlXferRequestId:
        return "nixl.xfer_request_id";
    case AttributeKey::RemoteAgent:
        return "remote_agent";
    }
    return "nixl.attribute";
}

static_assert(toStringView(static_cast<AttributeKey>(attributeKeyCount)) == "nixl.attribute");

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_API_ATTRIBUTE_KEY_H
