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

#include <array>
#include <bit>
#include <cstdint>
#include <optional>

#include "backend_trace.h"
#include "serdes/serdes.h"
#include "trace_context.h"

namespace {

static_assert(std::endian::native == std::endian::little);

struct requestTag {
    std::uint64_t requestId;
    std::uint64_t execution;
};

} // namespace

void
nixl::trace::addTraceTags(nixlSerDes &ser_des, const TraceIdentity &identity) {
    if (!identity.context.sampled()) {
        return;
    }
    std::array<std::uint8_t, traceContextWireSize> record;
    const auto size = encodeTraceContext(identity.context, record);
    if (!size) {
        return;
    }
    ser_des.addBuf("tctx", record.data(), *size);
    if (identity.requestId != 0) {
        const requestTag tag{identity.requestId, identity.execution};
        ser_des.addBuf("treq", &tag, sizeof(tag));
    }
}

std::optional<nixl::trace::TraceIdentity>
nixl::trace::getTraceTags(nixlSerDes &ser_des) {
    std::array<std::uint8_t, traceContextWireSize> record;
    TraceIdentity identity;
    if ((ser_des.peekBufLen("tctx") != record.size()) ||
        (ser_des.getBuf("tctx", record.data(), record.size()) != NIXL_SUCCESS) ||
        (decodeTraceContext(record, identity.context) != WireDecodeResult::Ok)) {
        return std::nullopt;
    }
    requestTag tag;
    if ((ser_des.peekBufLen("treq") == sizeof(tag)) &&
        (ser_des.getBuf("treq", &tag, sizeof(tag)) == NIXL_SUCCESS) && (tag.requestId != 0)) {
        identity.requestId = tag.requestId;
        identity.execution = tag.execution;
    }
    return identity;
}
