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
#ifndef NIXL_SRC_CORE_TRACING_RECORD_H
#define NIXL_SRC_CORE_TRACING_RECORD_H

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nixl::trace {

/**
 * @brief Operation kind. Aligns 1:1 with the Chakra NodeType vocabulary and is
 *        used as a color/label hint by NVTX.
 */
enum class Kind : std::uint8_t {
    Generic = 0,
    Compute,
    MemoryR,
    MemoryW,
    CommSend,
    CommRecv,
    CommColl,
    Metadata,
};

enum class Record : std::uint8_t {
    RegisterMem,
    DeregisterMem,
    MakeConnection,
    MakeXferReq,
    CreateXferReq,
    PostXferReqWrite,
    PostXferReqRead,
    ReleaseXferReq,
    XferComplete,
    XferError,
    GenNotif,
    GetNotifs,
    LoadRemoteMd,
    FetchRemoteMd,
    PrepMemView,
    ReleaseMemView,
    Submit,
    WireSubmitted,
    WireCompleted,
    NotifSent,
    NotifReceived,
    RemoteObserved,
    Phase,
    WireInflight,
    Span,
};

inline constexpr std::size_t recordCount = static_cast<std::size_t>(Record::Span) + 1;

[[nodiscard]] constexpr std::string_view
toStringView(Record record) noexcept {
    switch (record) {
    case Record::RegisterMem:
        return "nixl::registerMem";
    case Record::DeregisterMem:
        return "nixl::deregisterMem";
    case Record::MakeConnection:
        return "nixl::makeConnection";
    case Record::MakeXferReq:
        return "nixl::makeXferReq";
    case Record::CreateXferReq:
        return "nixl::createXferReq";
    case Record::PostXferReqWrite:
        return "nixl::postXferReq.write";
    case Record::PostXferReqRead:
        return "nixl::postXferReq.read";
    case Record::ReleaseXferReq:
        return "nixl::releaseXferReq";
    case Record::XferComplete:
        return "nixl::xfer.complete";
    case Record::XferError:
        return "nixl::xfer.error";
    case Record::GenNotif:
        return "nixl::genNotif";
    case Record::GetNotifs:
        return "nixl::getNotifs";
    case Record::LoadRemoteMd:
        return "nixl::loadRemoteMD";
    case Record::FetchRemoteMd:
        return "nixl::fetchRemoteMD";
    case Record::PrepMemView:
        return "nixl::prepMemView";
    case Record::ReleaseMemView:
        return "nixl::releaseMemView";
    case Record::Submit:
        return "nixl::submit";
    case Record::WireSubmitted:
        return "nixl::wire.submitted";
    case Record::WireCompleted:
        return "nixl::wire.completed";
    case Record::NotifSent:
        return "nixl::notif.sent";
    case Record::NotifReceived:
        return "nixl::notif.received";
    case Record::RemoteObserved:
        return "nixl::remote.observed";
    case Record::Phase:
        return "nixl::phase";
    case Record::WireInflight:
        return "nixl::wire.inflight";
    case Record::Span:
        return "nixl::span";
    }
    return "nixl::record";
}

[[nodiscard]] constexpr Kind
kindOf(Record record) noexcept {
    switch (record) {
    case Record::RegisterMem:
        return Kind::MemoryW;
    case Record::PrepMemView:
        return Kind::MemoryR;
    case Record::PostXferReqWrite:
    case Record::NotifSent:
        return Kind::CommSend;
    case Record::PostXferReqRead:
    case Record::NotifReceived:
        return Kind::CommRecv;
    case Record::XferComplete:
    case Record::XferError:
    case Record::GenNotif:
    case Record::GetNotifs:
    case Record::LoadRemoteMd:
    case Record::FetchRemoteMd:
    case Record::Submit:
    case Record::WireSubmitted:
    case Record::WireCompleted:
    case Record::RemoteObserved:
    case Record::Phase:
        return Kind::Metadata;
    case Record::DeregisterMem:
    case Record::MakeConnection:
    case Record::MakeXferReq:
    case Record::CreateXferReq:
    case Record::ReleaseXferReq:
    case Record::ReleaseMemView:
    case Record::WireInflight:
    case Record::Span:
        return Kind::Generic;
    }
    return Kind::Generic;
}

static_assert(toStringView(static_cast<Record>(recordCount)) == "nixl::record");

} // namespace nixl::trace

#endif // NIXL_SRC_CORE_TRACING_RECORD_H
