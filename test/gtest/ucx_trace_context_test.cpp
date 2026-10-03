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
#include "common.h"
#include "serdes/serdes.h"
#include "tracing/trace_context.h"
#include "ucx_backend.h"

#include <gtest/gtest.h>

#include <string>

namespace {

constexpr const char *sender_name = "sender";

[[nodiscard]] std::string
olderSenderNotif(const std::string &agent, const std::string &msg) {
    nixlSerDes ser_des;
    ser_des.addStr("name", agent);
    ser_des.addStr("msg", msg);
    return ser_des.exportStr();
}

} // namespace

TEST(SerDesProbe, NextTagIsNeitherConsumesNorLogs) {
    nixlSerDes writer;
    writer.addStr("first", "1");
    writer.addStr("second", "2");
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(writer.exportStr()), NIXL_SUCCESS);
    const size_t problems = gtest::LogProblemCounter::getProblemCount();

    EXPECT_FALSE(reader.nextTagIs("second"));
    EXPECT_TRUE(reader.nextTagIs("first"));
    EXPECT_EQ(reader.getStr("first"), "1");
    EXPECT_TRUE(reader.nextTagIs("second"));
    EXPECT_EQ(reader.getStr("second"), "2");
    EXPECT_FALSE(reader.nextTagIs("second"));
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
}

TEST(UcxTraceContextWire, OlderReceiverStillReadsTheMessage) {
    const auto context = nixl::trace::generateTraceContext(1.0);
    nixlSerDes older_receiver;
    ASSERT_EQ(older_receiver.importStr(nixl::ucx::serializeNotif(sender_name, "msg", &context)),
              NIXL_SUCCESS);

    EXPECT_EQ(older_receiver.getStr("name"), sender_name);
    EXPECT_EQ(older_receiver.getStr("msg"), "msg");
}

TEST(UcxTraceContextWire, UnsampledRequestWritesNoContextBytes) {
    const auto unsampled = nixl::trace::generateTraceContext(0.0);
    ASSERT_TRUE(unsampled.valid());

    EXPECT_EQ(nixl::ucx::serializeNotif(sender_name, "msg", &unsampled),
              olderSenderNotif(sender_name, "msg"));
    EXPECT_EQ(nixl::ucx::serializeNotif(sender_name, "msg", nullptr),
              olderSenderNotif(sender_name, "msg"));
}

TEST(UcxTraceContextWire, SampledContextRoundTrips) {
    const auto context = nixl::trace::generateTraceContext(1.0);
    const auto notif =
        nixl::ucx::deserializeNotif(nixl::ucx::serializeNotif(sender_name, "msg", &context));

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_EQ(notif.traceContext, context);
}

TEST(UcxTraceContextWire, UndecodableContextKeepsTheMessage) {
    const std::string later_version(nixl::trace::traceContextWireSize, '\x02');
    const std::string malformed(nixl::trace::traceContextWireSize - 1, '\x01');
    for (const auto &record : {later_version, malformed}) {
        nixlSerDes sender;
        sender.addStr("name", sender_name);
        sender.addStr("msg", "msg");
        sender.addStr("tctx", record);
        const size_t problems = gtest::LogProblemCounter::getProblemCount();

        const auto notif = nixl::ucx::deserializeNotif(sender.exportStr());

        EXPECT_EQ(notif.agent, sender_name);
        EXPECT_EQ(notif.msg, "msg");
        EXPECT_FALSE(notif.traceContext.has_value());
        EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
    }
}

TEST(UcxTraceContextWire, OlderSenderNotificationParsesWithoutLogging) {
    const size_t problems = gtest::LogProblemCounter::getProblemCount();
    const auto notif = nixl::ucx::deserializeNotif(olderSenderNotif(sender_name, "msg"));

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_FALSE(notif.traceContext.has_value());
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
}
