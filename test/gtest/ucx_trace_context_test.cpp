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
#include "ucx_trace_peer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

extern "C" char **environ;

namespace {

using span_record_t = std::vector<std::string>;
using ucx_trace_peer::address;
using ucx_trace_peer::agentConfig;
using ucx_trace_peer::buffer_size;
using ucx_trace_peer::efa_warning;
using ucx_trace_peer::poll_interval;
using ucx_trace_peer::receiver_name;
using ucx_trace_peer::sender_name;
using ucx_trace_peer::tagged_msg;
using ucx_trace_peer::untagged_msg;
using ucx_trace_peer::wait_timeout;

constexpr int wait_timeout_ms = static_cast<int>(std::chrono::milliseconds(wait_timeout).count());
constexpr const char *target_name = "target";
constexpr const char *unsampled_name = "unsampled";
constexpr const char *untraced_name = "untraced";
constexpr std::array<const char *, 3> sender_phases{"nixl::submit",
                                                    "nixl::wire.submitted",
                                                    "nixl::notif.sent"};

[[nodiscard]] std::string
olderSenderNotif(const std::string &agent, const std::string &msg) {
    nixlSerDes ser_des;
    ser_des.addStr("name", agent);
    ser_des.addStr("msg", msg);
    return ser_des.exportStr();
}

[[nodiscard]] bool
awaitBytes(int fd, void *data, size_t size) {
    pollfd pfd{fd, POLLIN, 0};
    return (::poll(&pfd, 1, wait_timeout_ms) > 0) &&
        (::read(fd, data, size) == static_cast<ssize_t>(size));
}

[[nodiscard]] nixl_status_t
awaitCompletion(nixlAgent &agent, nixlXferReqH *req, nixl_status_t status) {
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    while ((status == NIXL_IN_PROG) && (std::chrono::steady_clock::now() < deadline)) {
        std::this_thread::sleep_for(poll_interval);
        status = agent.getXferStatus(req);
    }
    return status;
}

void
writeToSelf(const std::string &name,
            const nixl_b_params_t &backend_params,
            size_t desc_count,
            const std::vector<std::optional<std::string>> &post_notifs,
            nixl::trace::TraceContext &context) {
    const gtest::LogIgnoreGuard efa(efa_warning);
    nixlAgent agent(name, agentConfig());
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("UCX", backend_params, backend), NIXL_SUCCESS);
    const std::vector<char> src_buffer(desc_count * buffer_size, 'x');
    const std::vector<char> dst_buffer(desc_count * buffer_size);
    nixl_reg_dlist_t regs(DRAM_SEG);
    regs.addDesc(nixlBlobDesc(address(src_buffer), src_buffer.size(), 0));
    regs.addDesc(nixlBlobDesc(address(dst_buffer), dst_buffer.size(), 0));
    ASSERT_EQ(agent.registerMem(regs), NIXL_SUCCESS);

    nixl_xfer_dlist_t src(DRAM_SEG);
    nixl_xfer_dlist_t dst(DRAM_SEG);
    for (size_t i = 0; i < desc_count; ++i) {
        src.addDesc(nixlBasicDesc(address(src_buffer) + i * buffer_size, buffer_size, 0));
        dst.addDesc(nixlBasicDesc(address(dst_buffer) + i * buffer_size, buffer_size, 0));
    }
    nixlXferReqH *req = nullptr;
    ASSERT_EQ(agent.createXferReq(NIXL_WRITE, src, dst, name, req), NIXL_SUCCESS);
    context = req->traceContext();
    for (const auto &notif : post_notifs) {
        nixl_opt_args_t params;
        params.notif = notif;
        ASSERT_EQ(awaitCompletion(agent, req, agent.postXferReq(req, &params)), NIXL_SUCCESS);
    }
    ASSERT_EQ(agent.releaseXferReq(req), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterMem(regs), NIXL_SUCCESS);
}

[[nodiscard]] std::optional<std::string>
pluginDirsArg() {
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    for (std::string arg; std::getline(cmdline, arg, '\0');) {
        if (arg.starts_with("--tests_plugin_dirs=")) {
            return arg;
        }
    }
    return std::nullopt;
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

[[nodiscard]] testing::Matcher<const std::string &>
phaseEventTime(const span_record_t &span) {
    constexpr std::string_view key = "nixl.phase.timestamp_us=";
    for (const auto &field : span) {
        if (field.starts_with(key)) {
            return testing::Eq("at_us=" + field.substr(key.size()));
        }
    }
    return testing::StartsWith(std::string(key));
}

class ucxTraceContextPropagation : public testing::Test {
protected:
    void
    SetUp() override {
        const auto prefix = std::filesystem::temp_directory_path() /
            ("nixl_trace_recorder_" + std::to_string(::getpid()));
        traceFile = prefix.string() + ".spans";
        mdFile = prefix.string() + ".md";
        receiverLog = prefix.string() + ".log";
        std::filesystem::remove(traceFile);
        std::filesystem::remove(mdFile);
        std::filesystem::remove(receiverLog);
        env.addVar("NIXL_TELEMETRY_ENABLE", "n");
        env.unsetVar("NIXL_ETCD_ENDPOINTS");
        env.unsetVar("NIXL_TCPSTORE_ENDPOINT");
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
        std::filesystem::remove(receiverLog);
    }

    void
    startReceiver(int up_fd, int down_fd) {
        const auto binary = std::filesystem::read_symlink("/proc/self/exe").parent_path() /
            ucx_trace_peer::receiver_binary;
        ASSERT_TRUE(std::filesystem::exists(binary)) << binary;
        std::vector<std::string> args{binary.string()};
        if (const auto dirs = pluginDirsArg()) {
            args.push_back(*dirs);
        }
        std::vector<std::string> vars{
            std::string(ucx_trace_peer::up_fd_var) + "=" + std::to_string(up_fd),
            std::string(ucx_trace_peer::down_fd_var) + "=" + std::to_string(down_fd),
            std::string(ucx_trace_peer::md_file_var) + "=" + mdFile.string()};
        for (char **entry = environ; *entry != nullptr; ++entry) {
            const std::string_view var(*entry);
            if (!var.starts_with("GTEST_") && !var.starts_with("XML_OUTPUT_FILE=") &&
                !var.starts_with("TEST_PREMATURE_EXIT_FILE=")) {
                vars.emplace_back(var);
            }
        }
        std::vector<char *> argv;
        for (std::string &arg : args) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);
        std::vector<char *> envp;
        for (std::string &var : vars) {
            envp.push_back(var.data());
        }
        envp.push_back(nullptr);

        posix_spawn_file_actions_t actions;
        ASSERT_EQ(::posix_spawn_file_actions_init(&actions), 0);
        ::posix_spawn_file_actions_addopen(
            &actions, STDOUT_FILENO, receiverLog.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ::posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
        const int rc = ::posix_spawn(
            &receiverPid, binary.c_str(), &actions, nullptr, argv.data(), envp.data());
        ::posix_spawn_file_actions_destroy(&actions);
        ASSERT_EQ(rc, 0) << std::strerror(rc);
    }

    [[nodiscard]] std::string
    receiverOutput() const {
        std::ifstream log(receiverLog);
        return "\nreceiver output:\n" + std::string(std::istreambuf_iterator<char>(log), {});
    }

    void
    expectSenderPhases(std::string_view agent,
                       const nixl::trace::TraceContext &context,
                       size_t posts,
                       size_t notifying_posts) const {
        ASSERT_TRUE(context.sampled());
        const std::vector<std::string> expected{
            "correlation=" + std::to_string(context.correlationId64()),
            "context=" + nixl::trace::formatTraceparent(context),
            "nixl.traceparent=" + nixl::trace::formatTraceparent(context),
            "nixl.backend=UCX"};
        const std::array<std::pair<const char *, size_t>, 3> expected_counts{
            {{"nixl::submit", posts},
             {"nixl::wire.submitted", posts},
             {"nixl::notif.sent", notifying_posts}}};
        for (const auto &[phase, count] : expected_counts) {
            const auto spans = recordedSpans(traceFile, agent, phase);
            EXPECT_EQ(spans.size(), count) << phase;
            for (const auto &span : spans) {
                EXPECT_THAT(span, testing::IsSupersetOf(expected)) << phase;
                EXPECT_THAT(span, testing::Contains(phaseEventTime(span))) << phase;
            }
        }
    }

    gtest::ScopedEnv env;
    std::filesystem::path traceFile;
    std::filesystem::path mdFile;
    std::filesystem::path receiverLog;
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
    const std::vector<char> sender_buffer(buffer_size, 'x');
    int up[2];
    int down[2];
    ASSERT_EQ(::pipe(up), 0);
    nixl::scopedFd up_read(up[0]);
    nixl::scopedFd up_write(up[1]);
    ASSERT_EQ(::pipe(down), 0);
    nixl::scopedFd down_read(down[0]);
    nixl::scopedFd down_write(down[1]);
    ASSERT_EQ(::fcntl(up_read.get(), F_SETFD, FD_CLOEXEC), 0);
    ASSERT_EQ(::fcntl(down_write.get(), F_SETFD, FD_CLOEXEC), 0);
    ASSERT_NO_FATAL_FAILURE(startReceiver(up_write.get(), down_read.get()));
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
    uintptr_t receiver_address = 0;
    ASSERT_TRUE(awaitBytes(up_read.get(), &receiver_address, sizeof(receiver_address)))
        << "the receiver did not publish its metadata" << receiverOutput();
    std::ifstream md_stream(mdFile, std::ios::binary);
    const std::string receiver_md{std::istreambuf_iterator<char>(md_stream), {}};
    std::string loaded_name;
    ASSERT_EQ(sender.loadRemoteMD(receiver_md, loaded_name), NIXL_SUCCESS);

    nixl_xfer_dlist_t src(DRAM_SEG);
    src.addDesc(nixlBasicDesc(address(sender_buffer), buffer_size, 0));
    nixl_xfer_dlist_t dst(DRAM_SEG);
    dst.addDesc(nixlBasicDesc(receiver_address, buffer_size, 0));
    nixl_opt_args_t extra_params;
    extra_params.notif = tagged_msg;
    nixlXferReqH *req = nullptr;
    ASSERT_EQ(sender.createXferReq(NIXL_WRITE, src, dst, receiver_name, req, &extra_params),
              NIXL_SUCCESS);
    const nixl::trace::TraceContext context = req->traceContext();
    ASSERT_TRUE(context.sampled());

    ASSERT_EQ(awaitCompletion(sender, req, sender.postXferReq(req)), NIXL_SUCCESS);
    ASSERT_EQ(sender.releaseXferReq(req), NIXL_SUCCESS);
    ASSERT_EQ(sender.genNotif(receiver_name, untagged_msg), NIXL_SUCCESS);

    char ack = 0;
    ASSERT_TRUE(awaitBytes(up_read.get(), &ack, 1))
        << "the receiver did not get both notifications" << receiverOutput();
    ASSERT_EQ(sender.invalidateRemoteMD(receiver_name), NIXL_SUCCESS);
    ASSERT_EQ(sender.deregisterMem(regs), NIXL_SUCCESS);
    down_write.reset();
    int receiver_status = 0;
    ASSERT_EQ(::waitpid(receiverPid, &receiver_status, 0), receiverPid);
    receiverPid = -1;
    ASSERT_TRUE(WIFEXITED(receiver_status)) << receiverOutput();
    EXPECT_EQ(WEXITSTATUS(receiver_status), 0) << receiverOutput();
    EXPECT_EQ(gtest::LogProblemCounter::getProblemCount(), problems);

    const std::string correlation = "correlation=" + std::to_string(context.correlationId64());
    const std::string traceparent = nixl::trace::formatTraceparent(context);
#if defined(NIXL_TRACE_ENABLED)
    const auto posted = recordedSpans(traceFile, sender_name, "nixl::postXferReq.write");
    ASSERT_EQ(posted.size(), 1u);
    EXPECT_THAT(
        posted[0],
        testing::IsSupersetOf(
            {correlation, "context=" + traceparent, std::string("remote_agent=") + receiver_name}));
#endif
    const auto received = recordedSpans(traceFile, receiver_name, "nixl::notif.received");
    ASSERT_EQ(received.size(), 1u);
    EXPECT_THAT(received[0],
                testing::IsSupersetOf({correlation,
                                       "context=" + traceparent,
                                       "nixl.traceparent=" + traceparent,
                                       std::string("nixl.remote_agent=") + sender_name}));
    EXPECT_THAT(received[0], testing::Contains(phaseEventTime(received[0])));
    expectSenderPhases(sender_name, context, 1, 1);
}

TEST_F(ucxTraceContextPropagation, SenderRecordsItsPhasesOncePerPost) {
    nixl::trace::TraceContext context;
    ASSERT_NO_FATAL_FAILURE(writeToSelf(sender_name, {}, 1, {tagged_msg, std::nullopt}, context));
    expectSenderPhases(sender_name, context, 2, 1);
}

TEST_F(ucxTraceContextPropagation, ThreadPoolRecordsItsPhasesOncePerPostNotPerChunk) {
    nixl::trace::TraceContext context;
    ASSERT_NO_FATAL_FAILURE(writeToSelf(sender_name,
                                        {{"num_threads", "2"}, {"split_batch_size", "2"}},
                                        2,
                                        {tagged_msg, std::nullopt},
                                        context));
    expectSenderPhases(sender_name, context, 2, 1);
}

TEST_F(ucxTraceContextPropagation, DeferredNotificationIsRecordedByTheStatusCheck) {
    const gtest::LogIgnoreGuard efa(efa_warning);
    nixlAgentConfig idle_config;
    idle_config.useProgThread = false;
    nixlAgent target(target_name, idle_config);
    nixlBackendH *target_backend = nullptr;
    ASSERT_EQ(target.createBackend("UCX", {}, target_backend), NIXL_SUCCESS);
    const std::vector<char> dst_buffer(buffer_size);
    nixl_reg_dlist_t target_regs(DRAM_SEG);
    target_regs.addDesc(nixlBlobDesc(address(dst_buffer), buffer_size, 0));
    ASSERT_EQ(target.registerMem(target_regs), NIXL_SUCCESS);
    std::string target_md;
    ASSERT_EQ(target.getLocalMD(target_md), NIXL_SUCCESS);

    nixlAgent sender(sender_name, agentConfig());
    nixlBackendH *sender_backend = nullptr;
    ASSERT_EQ(sender.createBackend("UCX", {}, sender_backend), NIXL_SUCCESS);
    const std::vector<char> src_buffer(buffer_size, 'x');
    nixl_reg_dlist_t sender_regs(DRAM_SEG);
    sender_regs.addDesc(nixlBlobDesc(address(src_buffer), buffer_size, 0));
    ASSERT_EQ(sender.registerMem(sender_regs), NIXL_SUCCESS);
    std::string loaded_name;
    ASSERT_EQ(sender.loadRemoteMD(target_md, loaded_name), NIXL_SUCCESS);

    nixl_xfer_dlist_t src(DRAM_SEG);
    src.addDesc(nixlBasicDesc(address(src_buffer), buffer_size, 0));
    nixl_xfer_dlist_t dst(DRAM_SEG);
    dst.addDesc(nixlBasicDesc(address(dst_buffer), buffer_size, 0));
    nixl_opt_args_t params;
    params.notif = tagged_msg;
    nixlXferReqH *req = nullptr;
    ASSERT_EQ(sender.createXferReq(NIXL_WRITE, src, dst, target_name, req, &params), NIXL_SUCCESS);
    const nixl::trace::TraceContext context = req->traceContext();

    nixl_status_t status = sender.postXferReq(req);
    ASSERT_EQ(status, NIXL_IN_PROG);
    EXPECT_TRUE(recordedSpans(traceFile, sender_name, "nixl::notif.sent").empty());
    nixl_notifs_t notifs;
    const auto deadline = std::chrono::steady_clock::now() + wait_timeout;
    while ((status == NIXL_IN_PROG) && (std::chrono::steady_clock::now() < deadline)) {
        ASSERT_EQ(target.getNotifs(notifs), NIXL_SUCCESS);
        std::this_thread::sleep_for(poll_interval);
        status = sender.getXferStatus(req);
    }
    ASSERT_EQ(status, NIXL_SUCCESS);
    ASSERT_EQ(sender.releaseXferReq(req), NIXL_SUCCESS);
    ASSERT_EQ(sender.invalidateRemoteMD(target_name), NIXL_SUCCESS);
    ASSERT_EQ(sender.deregisterMem(sender_regs), NIXL_SUCCESS);
    ASSERT_EQ(target.deregisterMem(target_regs), NIXL_SUCCESS);
    expectSenderPhases(sender_name, context, 1, 1);
}

TEST_F(ucxTraceContextPropagation, UnsampledAndUntracedPostsRecordNoPhases) {
    nixl::trace::TraceContext unsampled;
    env.addVar("NIXL_TRACE_SAMPLE_RATIO", "0");
    ASSERT_NO_FATAL_FAILURE(writeToSelf(unsampled_name, {}, 1, {tagged_msg}, unsampled));
    ASSERT_TRUE(unsampled.valid());
    ASSERT_FALSE(unsampled.sampled());

    nixl::trace::TraceContext untraced;
    env.addVar("NIXL_TRACE_BACKENDS", "");
    ASSERT_NO_FATAL_FAILURE(writeToSelf(untraced_name, {}, 1, {tagged_msg}, untraced));
    ASSERT_FALSE(untraced.valid());

    for (const char *phase : sender_phases) {
        EXPECT_TRUE(recordedSpans(traceFile, unsampled_name, phase).empty()) << phase;
        EXPECT_TRUE(recordedSpans(traceFile, untraced_name, phase).empty()) << phase;
    }
#if defined(NIXL_TRACE_ENABLED)
    EXPECT_EQ(recordedSpans(traceFile, unsampled_name, "nixl::postXferReq.write").size(), 1u);
#endif
}
