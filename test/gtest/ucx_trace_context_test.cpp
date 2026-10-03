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
#include "common/scoped_fd.h"
#include "nixl.h"
#include "serdes/serdes.h"
#include "tracing/trace_context.h"
#include "transfer_request.h"
#include "ucx_backend.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using span_record_t = std::vector<std::string>;

constexpr const char *sender_name = "sender";
constexpr const char *receiver_name = "receiver";
constexpr const char *tagged_msg = "tagged";
constexpr const char *untagged_msg = "untagged";
constexpr const char *efa_warning =
    "Amazon EFA\\(s\\) were detected, but the UCX backend was configured";
constexpr size_t buffer_size = 4096;
constexpr std::chrono::seconds wait_timeout{30};
constexpr int wait_timeout_ms = static_cast<int>(std::chrono::milliseconds(wait_timeout).count());
constexpr std::chrono::milliseconds poll_interval{10};

[[nodiscard]] std::string
olderSenderNotif(const std::string &agent, const std::string &msg) {
    nixlSerDes ser_des;
    ser_des.addStr("name", agent);
    ser_des.addStr("msg", msg);
    return ser_des.exportStr();
}

[[nodiscard]] bool
sendByte(int fd) {
    const char byte = 1;
    return ::write(fd, &byte, 1) == 1;
}

[[nodiscard]] bool
awaitByte(int fd) {
    pollfd pfd{fd, POLLIN, 0};
    char byte = 0;
    return (::poll(&pfd, 1, wait_timeout_ms) > 0) && (::read(fd, &byte, 1) == 1);
}

[[nodiscard]] nixlAgentConfig
agentConfig() {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    return cfg;
}

[[nodiscard]] uintptr_t
address(const std::vector<char> &buffer) {
    return reinterpret_cast<uintptr_t>(buffer.data());
}

[[nodiscard]] std::vector<span_record_t>
recordedSpans(const std::filesystem::path &path, std::string_view agent, std::string_view name) {
    std::vector<span_record_t> spans;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        std::istringstream fields(line);
        span_record_t span{std::istream_iterator<std::string>(fields),
                           std::istream_iterator<std::string>()};
        if ((span.size() >= 2) && (span[0] == agent) && (span[1] == name)) {
            spans.push_back(std::move(span));
        }
    }
    return spans;
}

[[nodiscard]] int
runReceiver(int up_fd,
            int down_fd,
            const std::vector<char> &buffer,
            const std::filesystem::path &md_file) {
    const size_t problems = gtest::LogProblemCounter::getProblemCount();
    {
        const gtest::LogIgnoreGuard efa(efa_warning);
        nixlAgent receiver(receiver_name, agentConfig());
        nixlBackendH *backend = nullptr;
        nixl_reg_dlist_t regs(DRAM_SEG);
        regs.addDesc(nixlBlobDesc(address(buffer), buffer.size(), 0));
        std::string md;
        if ((receiver.createBackend("UCX", {}, backend) != NIXL_SUCCESS) ||
            (receiver.registerMem(regs) != NIXL_SUCCESS) ||
            (receiver.getLocalMD(md) != NIXL_SUCCESS)) {
            return 1;
        }
        std::ofstream(md_file, std::ios::binary) << md;
        if (!sendByte(up_fd)) {
            return 2;
        }

        nixl_notifs_t notifs;
        const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
        while ((notifs[sender_name].size() < 2) && (std::chrono::steady_clock::now() < deadline)) {
            if (receiver.getNotifs(notifs) != NIXL_SUCCESS) {
                return 3;
            }
            std::this_thread::sleep_for(poll_interval);
        }
        const std::multiset<std::string> received(notifs[sender_name].begin(),
                                                  notifs[sender_name].end());
        if (received != std::multiset<std::string>{tagged_msg, untagged_msg}) {
            return 4;
        }
        if (!sendByte(up_fd)) {
            return 5;
        }

        char quit = 0;
        while (::read(down_fd, &quit, 1) > 0) {}
        if (receiver.deregisterMem(regs) != NIXL_SUCCESS) {
            return 6;
        }
    }
    return (gtest::LogProblemCounter::getProblemCount() == problems) ? 0 : 7;
}

class ucxTraceContextPropagation : public testing::Test {
protected:
    void
    SetUp() override {
        const auto prefix = std::filesystem::temp_directory_path() /
            ("nixl_trace_recorder_" + std::to_string(::getpid()));
        traceFile = prefix.string() + ".spans";
        mdFile = prefix.string() + ".md";
        std::filesystem::remove(traceFile);
        std::filesystem::remove(mdFile);
        env.addVar("NIXL_TELEMETRY_ENABLE", "n");
        env.unsetVar("NIXL_ETCD_ENDPOINTS");
        env.addVar("NIXL_TRACE_BACKENDS", "recorder");
        env.addVar("NIXL_TRACE_SAMPLE_RATIO", "1");
        env.addVar("NIXL_TEST_TRACE_FILE", traceFile.string());
    }

    void
    TearDown() override {
        if (receiverPid > 0) {
            ::kill(receiverPid, SIGKILL);
            ::waitpid(receiverPid, nullptr, 0);
        }
        std::filesystem::remove(traceFile);
        std::filesystem::remove(mdFile);
    }

    gtest::ScopedEnv env;
    std::filesystem::path traceFile;
    std::filesystem::path mdFile;
    pid_t receiverPid = -1;
};

} // namespace

TEST(SerDesProbe, PeekBufLenNeitherConsumesNorLogs) {
    nixlSerDes writer;
    writer.addStr("first", "1");
    writer.addStr("other", "22");
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(writer.exportStr()), NIXL_SUCCESS);
    const size_t problems = gtest::LogProblemCounter::getProblemCount();

    EXPECT_FALSE(reader.peekBufLen("other").has_value());
    EXPECT_EQ(reader.peekBufLen("first"), 1u);
    EXPECT_EQ(reader.getStr("first"), "1");
    EXPECT_EQ(reader.peekBufLen("other"), 2u);
    EXPECT_EQ(reader.getStr("other"), "22");
    EXPECT_FALSE(reader.peekBufLen("other").has_value());
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
}

TEST(SerDesProbe, PeekBufLenReportsEmptyValuesAndRejectsTruncatedOnes) {
    nixlSerDes writer;
    writer.addStr("empty", "");
    writer.addStr("value", "12345");
    const std::string complete = writer.exportStr();
    nixlSerDes reader;
    ASSERT_EQ(reader.importStr(complete.substr(0, complete.size() - 2)), NIXL_SUCCESS);
    const size_t problems = gtest::LogProblemCounter::getProblemCount();

    EXPECT_EQ(reader.peekBufLen("empty"), 0u);
    char unused;
    ASSERT_EQ(reader.getBuf("empty", &unused, 0), NIXL_SUCCESS);
    EXPECT_FALSE(reader.peekBufLen("value").has_value());
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
        nixl::ucx::deserializeNotif(nixl::ucx::serializeNotif(sender_name, "msg", &context), true);

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_EQ(notif.traceContext, context);
}

TEST(UcxTraceContextWire, ReceiverWithoutTracingIgnoresTheContext) {
    const auto context = nixl::trace::generateTraceContext(1.0);
    const auto notif =
        nixl::ucx::deserializeNotif(nixl::ucx::serializeNotif(sender_name, "msg", &context), false);

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_FALSE(notif.traceContext.has_value());
}

TEST(UcxTraceContextWire, UndecodableContextKeepsTheMessage) {
    const std::string later_version(nixl::trace::traceContextWireSize,
                                    static_cast<char>(nixl::trace::traceContextWireVersion + 1));
    const std::string malformed(nixl::trace::traceContextWireSize - 1, '\x01');
    std::vector<std::string> notifications;
    for (const auto &record : {later_version, malformed, std::string{}}) {
        nixlSerDes sender;
        sender.addStr("name", sender_name);
        sender.addStr("msg", "msg");
        sender.addStr("tctx", record);
        notifications.push_back(sender.exportStr());
    }
    const auto context = nixl::trace::generateTraceContext(1.0);
    const std::string sampled = nixl::ucx::serializeNotif(sender_name, "msg", &context);
    notifications.push_back(sampled.substr(0, sampled.size() - 2));

    for (const auto &notification : notifications) {
        const size_t problems = gtest::LogProblemCounter::getProblemCount();

        const auto notif = nixl::ucx::deserializeNotif(notification, true);

        EXPECT_EQ(notif.agent, sender_name);
        EXPECT_EQ(notif.msg, "msg");
        EXPECT_FALSE(notif.traceContext.has_value());
        EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
    }
}

TEST(UcxTraceContextWire, OlderSenderNotificationParsesWithoutLogging) {
    const size_t problems = gtest::LogProblemCounter::getProblemCount();
    const auto notif = nixl::ucx::deserializeNotif(olderSenderNotif(sender_name, "msg"), true);

    EXPECT_EQ(notif.agent, sender_name);
    EXPECT_EQ(notif.msg, "msg");
    EXPECT_FALSE(notif.traceContext.has_value());
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);
}

TEST_F(ucxTraceContextPropagation, OneTraceIdIsObservedOnBothAgents) {
    std::vector<char> receiver_buffer(buffer_size);
    const std::vector<char> sender_buffer(buffer_size, 'x');
    int up[2];
    int down[2];
    ASSERT_EQ(::pipe(up), 0);
    nixl::scopedFd up_read(up[0]);
    nixl::scopedFd up_write(up[1]);
    ASSERT_EQ(::pipe(down), 0);
    nixl::scopedFd down_read(down[0]);
    nixl::scopedFd down_write(down[1]);

    // Fork before creating any agent, while this process is still single-threaded;
    // the child registers receiver_buffer at the address the parent writes to.
    receiverPid = ::fork();
    ASSERT_GE(receiverPid, 0);
    if (receiverPid == 0) {
        up_read.reset();
        down_write.reset();
        ::_exit(runReceiver(up_write.get(), down_read.get(), receiver_buffer, mdFile));
    }
    up_write.reset();
    down_read.reset();

    const size_t problems = gtest::LogProblemCounter::getProblemCount();
    const gtest::LogIgnoreGuard efa(efa_warning);
    nixlAgent sender(sender_name, agentConfig());
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(sender.createBackend("UCX", {}, backend), NIXL_SUCCESS);
    nixl_reg_dlist_t regs(DRAM_SEG);
    regs.addDesc(nixlBlobDesc(address(sender_buffer), buffer_size, 0));
    ASSERT_EQ(sender.registerMem(regs), NIXL_SUCCESS);
    ASSERT_TRUE(awaitByte(up_read.get())) << "the receiver did not publish its metadata";
    std::ifstream md_stream(mdFile, std::ios::binary);
    const std::string receiver_md{std::istreambuf_iterator<char>(md_stream), {}};
    std::string loaded_name;
    ASSERT_EQ(sender.loadRemoteMD(receiver_md, loaded_name), NIXL_SUCCESS);

    nixl_xfer_dlist_t src(DRAM_SEG);
    src.addDesc(nixlBasicDesc(address(sender_buffer), buffer_size, 0));
    nixl_xfer_dlist_t dst(DRAM_SEG);
    dst.addDesc(nixlBasicDesc(address(receiver_buffer), buffer_size, 0));
    nixl_opt_args_t extra_params;
    extra_params.notif = tagged_msg;
    nixlXferReqH *req = nullptr;
    ASSERT_EQ(sender.createXferReq(NIXL_WRITE, src, dst, receiver_name, req, &extra_params),
              NIXL_SUCCESS);
    const nixl::trace::TraceContext context = req->traceContext();
    ASSERT_TRUE(context.sampled());

    nixl_status_t status = sender.postXferReq(req);
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    while ((status == NIXL_IN_PROG) && (std::chrono::steady_clock::now() < deadline)) {
        std::this_thread::sleep_for(poll_interval);
        status = sender.getXferStatus(req);
    }
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(sender.releaseXferReq(req), NIXL_SUCCESS);
    ASSERT_EQ(sender.genNotif(receiver_name, untagged_msg), NIXL_SUCCESS);

    ASSERT_TRUE(awaitByte(up_read.get())) << "the receiver did not get both notifications";
    ASSERT_EQ(sender.invalidateRemoteMD(receiver_name), NIXL_SUCCESS);
    ASSERT_EQ(sender.deregisterMem(regs), NIXL_SUCCESS);
    down_write.reset();
    int receiver_status = 0;
    ASSERT_EQ(::waitpid(receiverPid, &receiver_status, 0), receiverPid);
    receiverPid = -1;
    ASSERT_TRUE(WIFEXITED(receiver_status));
    EXPECT_EQ(WEXITSTATUS(receiver_status), 0);
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);

    const auto received = recordedSpans(traceFile, receiver_name, "nixl::notif.received");
    ASSERT_EQ(received.size(), 1u);
    EXPECT_THAT(
        received[0],
        testing::IsSupersetOf({"correlation=" + std::to_string(context.correlationId64()),
                               "nixl.traceparent=" + nixl::trace::formatTraceparent(context),
                               std::string("nixl.remote_agent=") + sender_name}));
}
