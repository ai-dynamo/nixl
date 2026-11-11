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

#include "serdes/serdes.h"
#include "ucx_backend.h"
#include "ucx_trace.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr const char *sender_name = "sender";
constexpr std::size_t trace_context_record_size = 26;

[[nodiscard]] std::string
notification(const std::vector<std::pair<std::string, std::string>> &tags) {
    nixlSerDes ser_des;
    ser_des.addStr("name", sender_name);
    ser_des.addStr("msg", "msg");
    for (const auto &[tag, value] : tags) {
        ser_des.addStr(tag, value);
    }
    return ser_des.exportStr();
}

} // namespace

TEST(UcxTraceCompiledOut, PostTraceIsEmpty) {
    static_assert(std::is_empty_v<nixl::ucx::PostTrace>);
}

TEST(UcxTraceCompiledOut, SampledPostSendsTheOlderNotification) {
    nixl::trace::TraceIdentity sampled;
    sampled.context.traceId.fill(0x11);
    sampled.context.spanId.fill(0x22);
    sampled.context.flags = 0x01;
    sampled.requestId = 7;
    sampled.execution = 1;

    EXPECT_EQ(nixl::ucx::serializeNotif(sender_name, "msg", nixlBackendPostTrace{nullptr, sampled}),
              notification({}));
}

TEST(UcxTraceCompiledOut, TracingSendersNotificationKeepsItsMessage) {
    const auto notif = nixl::ucx::deserializeNotif(
        notification({{"tctx", std::string(trace_context_record_size, '\x01')},
                      {"treq", std::string(2 * sizeof(std::uint64_t), '\x02')}}),
        nullptr);

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_FALSE(notif.trace.active());
}
