/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"
#include "transfer_request.h"
#include "tracing/api/agent_trace.h"
#include "tracing/core/trace_context.h"
#include "mocks/gmock_engine.h"

namespace gtest {
namespace agent {
    static constexpr const char *local_agent_name = "LocalAgent";
    static constexpr const char *remote_agent_name = "RemoteAgent";
    static constexpr const char *nonexisting_plugin = "NonExistingPlugin";

    /* Generates a random number in [0,255] (byte range). */
    unsigned char
    GetRandomByte() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> distr(0, 255);
        return static_cast<unsigned char>(distr(gen));
    }

    class blob {
    protected:
        static constexpr size_t bufLen = 256;
        static constexpr uint32_t devId = 0;

        std::unique_ptr<char[]> buf_;
        const nixlBlobDesc desc_;
        const char buf_pattern_;

    public:
        blob()
            : buf_(std::make_unique<char[]>(bufLen)),
              desc_(reinterpret_cast<uintptr_t>(buf_.get()), bufLen, devId),
              buf_pattern_(GetRandomByte()) {
            memset(buf_.get(), buf_pattern_, bufLen);
        }

        nixlBlobDesc
        getDesc() const {
            return desc_;
        }
    };

    struct selfXferSetup {
        nixlBackendH *backend = nullptr;
        blob localBlob;
        nixl_reg_dlist_t regDlist{DRAM_SEG};
        nixl_opt_args_t extraParams;
        nixl_xfer_dlist_t xferDlist{DRAM_SEG};
    };

    class agentHelper {
    protected:
        ScopedEnv trace_env_;
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;
        std::unique_ptr<nixlAgent> agent_;
        const std::string name_;

    public:
        agentHelper(const std::string &name,
                    const std::string &trace_backends = "",
                    const std::string &sample_ratio = "0")
            : name_(name) {
            trace_env_.addVar("NIXL_TRACE_BACKENDS", trace_backends);
            trace_env_.addVar("NIXL_TRACE_NVTX_WITHOUT_PROFILER", "1");
            trace_env_.addVar("NIXL_TRACE_SAMPLE_RATIO", sample_ratio);
            nixlAgentConfig cfg;
            cfg.useProgThread = true;
            agent_ = std::make_unique<nixlAgent>(name, cfg);
        }

        ~agentHelper() {
            /* We must release nixlAgent first (i.e. explicitly in the destructor), as it calls
               cleanup functions in gmock_engine, which must stay alive during the process. */
            agent_.reset();
        }

        nixlAgent *
        getAgent() const {
            return agent_.get();
        }

        const mocks::GMockBackendEngine &
        getGMockEngine() const {
            return gmock_engine_;
        }

        nixl_status_t
        createBackendWithGMock(nixl_b_params_t &params, nixlBackendH *&backend) {
            gmock_engine_.SetToParams(params);
            return agent_->createBackend(GetMockBackendName(), params, backend);
        }

        nixl_status_t
        getAndLoadRemoteMd(nixlAgent *remote_agent, std::string &remote_agent_name_out) {
            std::string remote_metadata;
            EXPECT_EQ(remote_agent->getLocalMD(remote_metadata), NIXL_SUCCESS);
            return agent_->loadRemoteMD(remote_metadata, remote_agent_name_out);
        }

        nixl_status_t
        initAndRegisterMemory(blob &blob,
                              nixl_reg_dlist_t &reg_dlist,
                              nixl_opt_args_t &extra_params,
                              nixlBackendH *backend) {
            reg_dlist.addDesc(blob.getDesc());
            extra_params.backends.push_back(backend);
            return agent_->registerMem(reg_dlist, &extra_params);
        }

        void
        setupSelfXfer(selfXferSetup &setup) {
            nixl_b_params_t params;
            EXPECT_EQ(createBackendWithGMock(params, setup.backend), NIXL_SUCCESS);
            EXPECT_EQ(initAndRegisterMemory(
                          setup.localBlob, setup.regDlist, setup.extraParams, setup.backend),
                      NIXL_SUCCESS);
            setup.xferDlist.addDesc(setup.localBlob.getDesc());
        }

        nixl_status_t
        createSelfXferReq(selfXferSetup &setup,
                          nixlXferReqH *&req,
                          nixl_xfer_op_t op = NIXL_WRITE) {
            return agent_->createXferReq(
                op, setup.xferDlist, setup.xferDlist, name_, req, &setup.extraParams);
        }
    };

#if defined(NIXL_TRACE_ENABLED)
    static constexpr const char *untraced_log = "cannot carry trace contexts";

    class tracingEnabledAgentFixture : public testing::Test {
    protected:
        static bool nvtxPluginAvailable_;
        std::unique_ptr<agentHelper> agent_helper_;

        static void
        SetUpTestSuite() {
            if (nvtxPluginAvailable_) {
                return;
            }
            const std::string dir = std::string(BUILD_DIR) + "/src/plugins/tracing/nvtx";
            nvtxPluginAvailable_ = std::filesystem::exists(dir);
            if (nvtxPluginAvailable_) {
                nixlPluginManager::getInstance().addPluginDirectory(dir);
            }
        }

        [[nodiscard]] virtual std::string
        sampleRatio() const {
            return "0";
        }

        void
        SetUp() override {
            if (!nvtxPluginAvailable_) {
                GTEST_SKIP() << "NVTX trace plugin (libtrace_backend_nvtx.so) was not built";
            }
            agent_helper_ = std::make_unique<agentHelper>(local_agent_name, "nvtx", sampleRatio());
        }
    };

    bool tracingEnabledAgentFixture::nvtxPluginAvailable_ = false;

    class sampledTracingAgentFixture : public tracingEnabledAgentFixture {
    protected:
        [[nodiscard]] std::string
        sampleRatio() const override {
            return "1";
        }
    };

    TEST_F(tracingEnabledAgentFixture, TracingOnHandsPluginARealTraceSink) {
        nixl_b_params_t params;
        nixlBackendH *backend = nullptr;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        const auto observed = agent_helper_->getGMockEngine().observedTraceSink();
        ASSERT_TRUE(observed.has_value());
        EXPECT_NE(*observed, nullptr);
    }

    TEST_F(tracingEnabledAgentFixture, TracingOnHandsPluginEachRequestsContext) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();

        nixlXferReqH *first = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, first), NIXL_SUCCESS);
        EXPECT_TRUE(first->traceable().requestIdentity().context.valid());
        EXPECT_THAT(engine.observedTraceContext(),
                    testing::Optional(first->traceable().requestIdentity().context));

        nixlXferReqH *second = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, second), NIXL_SUCCESS);
        ASSERT_NE(first->traceable().requestIdentity().context,
                  second->traceable().requestIdentity().context);
        EXPECT_THAT(engine.observedTraceContext(),
                    testing::Optional(second->traceable().requestIdentity().context));

        for (nixlXferReqH *req : {first, second, first}) {
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_THAT(engine.observedTraceContext(),
                        testing::Optional(req->traceable().postIdentity().context));
        }

        EXPECT_EQ(agent->releaseXferReq(first), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(second), NIXL_SUCCESS);
    }

    TEST_F(tracingEnabledAgentFixture, TracingOnHandsPluginThePreppedRequestsContext) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup.xferDlist, remote_side),
                  NIXL_SUCCESS);

        const std::vector<int> indices{0};
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(
            agent->makeXferReq(
                NIXL_WRITE, *local_side, indices, *remote_side, indices, req, &setup.extraParams),
            NIXL_SUCCESS);
        EXPECT_TRUE(req->traceable().requestIdentity().context.valid());
        EXPECT_THAT(agent_helper_->getGMockEngine().observedTraceContext(),
                    testing::Optional(req->traceable().requestIdentity().context));

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(tracingEnabledAgentFixture, UnsampledRequestsAreNotReportedUntraced) {
        LogMessageCounter untraced(untraced_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_TRUE(req->traceable().requestIdentity().context.valid());
        EXPECT_FALSE(req->traceable().requestIdentity().context.sampled());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

    TEST_F(sampledTracingAgentFixture, UnsupportedBackendIsReportedOnce) {
        LogMessageCounter untraced(untraced_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();

        for (int i = 0; i < 3; ++i) {
            nixlXferReqH *req = nullptr;
            ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
            EXPECT_TRUE(req->traceable().requestIdentity().context.sampled());
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_THAT(engine.observedTraceContext(),
                        testing::Optional(req->traceable().postIdentity().context));
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }
        EXPECT_EQ(untraced.count(), 1u);
    }

    TEST_F(sampledTracingAgentFixture, UnsupportedBackendIsReportedOnceForPreppedRequests) {
        LogMessageCounter untraced(untraced_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup.xferDlist, remote_side),
                  NIXL_SUCCESS);

        const std::vector<int> indices{0};
        for (int i = 0; i < 2; ++i) {
            nixlXferReqH *req = nullptr;
            ASSERT_EQ(agent->makeXferReq(NIXL_WRITE,
                                         *local_side,
                                         indices,
                                         *remote_side,
                                         indices,
                                         req,
                                         &setup.extraParams),
                      NIXL_SUCCESS);
            EXPECT_TRUE(req->traceable().requestIdentity().context.sampled());
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }
        EXPECT_EQ(untraced.count(), 1u);

        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(sampledTracingAgentFixture, SupportingBackendSeesTheSampledContextUnchanged) {
        LogMessageCounter untraced(untraced_log);
        const auto &engine = agent_helper_->getGMockEngine();
        ON_CALL(engine, supportsTraceContext()).WillByDefault(testing::Return(true));
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        ASSERT_TRUE(req->traceable().requestIdentity().context.sampled());
        EXPECT_THAT(engine.observedTraceContext(),
                    testing::Optional(req->traceable().requestIdentity().context));
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(engine.observedTraceContext(),
                    testing::Optional(req->traceable().postIdentity().context));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

    TEST_F(sampledTracingAgentFixture, LocalOnlyBackendIsNotReportedUntraced) {
        LogMessageCounter untraced(untraced_log);
        ON_CALL(agent_helper_->getGMockEngine(), supportsRemote())
            .WillByDefault(testing::Return(false));
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_TRUE(req->traceable().requestIdentity().context.sampled());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

    class recorderTracingAgentFixture : public testing::Test {
    protected:
        const std::filesystem::path trace_file_ = std::filesystem::temp_directory_path() /
            ("nixl_agent_trace_" + std::to_string(::getpid()) + ".spans");
        ScopedEnv env_;
        selfXferSetup setup_;
        std::unique_ptr<agentHelper> agent_helper_;

        static void
        SetUpTestSuite() {
            static bool added = false;
            if (!added) {
                nixlPluginManager::getInstance().addPluginDirectory(std::string(BUILD_DIR) +
                                                                    "/test/gtest/mocks");
                added = true;
            }
        }

        [[nodiscard]] virtual std::string
        sampleRatio() const {
            return "1";
        }

        void
        SetUp() override {
            std::filesystem::remove(trace_file_);
            env_.addVar("NIXL_TEST_TRACE_FILE", trace_file_.string());
            agent_helper_ =
                std::make_unique<agentHelper>(local_agent_name, "recorder", sampleRatio());
            ON_CALL(agent_helper_->getGMockEngine(), supportsTraceContext())
                .WillByDefault(testing::Return(true));
            agent_helper_->setupSelfXfer(setup_);
        }

        void
        TearDown() override {
            agent_helper_.reset();
            std::filesystem::remove(trace_file_);
        }

        [[nodiscard]] std::vector<std::vector<std::string>>
        records(std::string_view name) const {
            std::vector<std::vector<std::string>> records;
            std::ifstream file(trace_file_);
            for (std::string line; std::getline(file, line);) {
                std::istringstream fields(line);
                std::vector<std::string> record{std::istream_iterator<std::string>(fields),
                                                std::istream_iterator<std::string>()};
                if ((record.size() >= 2) && (record[1] == name)) {
                    records.push_back(std::move(record));
                }
            }
            return records;
        }

        [[nodiscard]] std::vector<std::string>
        recordNames() const {
            std::vector<std::string> names;
            std::ifstream file(trace_file_);
            for (std::string line; std::getline(file, line);) {
                std::istringstream fields(line);
                std::string agent;
                std::string name;
                if (fields >> agent >> name) {
                    names.push_back(std::move(name));
                }
            }
            return names;
        }

        [[nodiscard]] std::vector<std::vector<std::string>>
        completionMarks() const {
            return records("nixl::xfer.complete");
        }

        [[nodiscard]] static auto
        correlatedWith(const nixlXferReqH *req) {
            return testing::Contains("correlation=" +
                                     std::to_string(req->traceable().postIdentity().requestId));
        }

        [[nodiscard]] static auto
        identifies(const nixlXferReqH *req) {
            return testing::Contains("nixl.xfer_request_id=" +
                                     std::to_string(req->traceable().postIdentity().requestId));
        }

        [[nodiscard]] static auto
        withStatus(nixl_status_t status) {
            return testing::Contains("nixl.status=" + std::to_string(status));
        }

        [[nodiscard]] static auto
        postedAs(const nixlXferReqH *req, int execution) {
            return testing::AllOf(
                identifies(req),
                correlatedWith(req),
                testing::Contains("nixl.xfer_execution=" + std::to_string(execution)));
        }

        [[nodiscard]] nixl_xfer_trace_info_t
        traceInfo(const nixlXferReqH *req) const {
            nixl_xfer_trace_info_t info{};
            EXPECT_EQ(agent_helper_->getAgent()->getXferTraceInfo(req, info), NIXL_SUCCESS);
            return info;
        }

        [[nodiscard]] static auto
        carriesTraceInfo(const nixl_xfer_trace_info_t &info, std::string_view context_key) {
            return testing::AllOf(
                testing::Contains("correlation=" + std::to_string(info.requestId)),
                testing::Contains("nixl.xfer_execution=" + std::to_string(info.execution)),
                testing::Contains(std::string(context_key) + info.traceparent));
        }

        [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>>
        handedToPlugin() const {
            return agent_helper_->getGMockEngine().observedTraceRequests();
        }

        [[nodiscard]] static auto
        handed(const nixlXferReqH *req, std::uint64_t execution) {
            return testing::Pair(req->traceable().postIdentity().requestId, execution);
        }

        [[nodiscard]] static auto
        carries(const nixl::trace::TraceContext &context) {
            return testing::Contains("context=" + nixl::trace::formatTraceparent(context));
        }

        [[nodiscard]] static auto
        isSpanOf(const nixlXferReqH *req, const nixl::trace::TraceContext &context) {
            return testing::AllOf(
                correlatedWith(req),
                testing::Contains("span=" + nixl::trace::formatTraceparent(context)),
                testing::Not(testing::Contains(testing::StartsWith("context="))));
        }

        [[nodiscard]] static auto
        childOf(const nixl::trace::TraceContext &parent) {
            static constexpr char digits[] = "0123456789abcdef";
            std::string field = "parent=";
            for (const std::uint8_t byte : parent.spanId) {
                field += digits[byte >> 4];
                field += digits[byte & 0x0f];
            }
            return testing::Contains(field);
        }

        [[nodiscard]] static nixl::trace::TraceContext
        applicationContext(std::string_view flags) {
            return nixl::trace::parseTraceparent(
                       "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-" + std::string(flags))
                .value();
        }

        [[nodiscard]] static nixl::trace::TraceContext
        contextOf(const std::vector<std::string> &record, std::string_view key) {
            for (const auto &field : record) {
                if (field.starts_with(key)) {
                    return nixl::trace::parseTraceparent(field.substr(key.size()))
                        .value_or(nixl::trace::TraceContext{});
                }
            }
            return {};
        }
    };

    TEST_F(recorderTracingAgentFixture, PostCompletedAtOnceIsMarkedOnce) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(), testing::ElementsAre(correlatedWith(req)));
        EXPECT_EQ(agent->getXferStatus(req), NIXL_SUCCESS);
        EXPECT_EQ(completionMarks().size(), 1u);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, PostCompletedLaterIsMarkedByTheStatusCheck) {
        nixlAgent *agent = agent_helper_->getAgent();
        EXPECT_CALL(agent_helper_->getGMockEngine(), postXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_TRUE(completionMarks().empty());
        EXPECT_EQ(agent->getXferStatus(req), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(), testing::ElementsAre(correlatedWith(req)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, PostCompletedLaterIsMarkedByTheRepostCheck) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_TRUE(completionMarks().empty());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(),
                    testing::ElementsAre(correlatedWith(req), correlatedWith(req)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, PostCompletedLaterIsMarkedByRelease) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).Times(2).WillRepeatedly(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_SUCCESS))
            .WillOnce(testing::Return(NIXL_IN_PROG));
        nixlXferReqH *completed = nullptr;
        nixlXferReqH *cancelled = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, completed), NIXL_SUCCESS);
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, cancelled), NIXL_SUCCESS);
        const auto completed_mark = correlatedWith(completed);
        const auto completed_post = postedAs(completed, 1);

        EXPECT_EQ(agent->postXferReq(completed), NIXL_IN_PROG);
        EXPECT_EQ(agent->postXferReq(cancelled), NIXL_IN_PROG);
        EXPECT_EQ(agent->releaseXferReq(completed), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(cancelled), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(), testing::ElementsAre(completed_mark));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(completed_post));
    }

    TEST_F(recorderTracingAgentFixture, EveryPostCompletedAtOnceIsMarked) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(),
                    testing::ElementsAre(correlatedWith(req), correlatedWith(req)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, FailedPostIsNotMarked) {
        nixlAgent *agent = agent_helper_->getAgent();
        EXPECT_CALL(agent_helper_->getGMockEngine(), postXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_BACKEND);
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_BACKEND);
        EXPECT_TRUE(completionMarks().empty());
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, CreationRefusedBeforeTheRequestIsNotRecorded) {
        nixlAgent *agent = agent_helper_->getAgent();
        const nixl_xfer_dlist_t shorter(DRAM_SEG);
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(
            agent->createXferReq(
                NIXL_WRITE, setup_.xferDlist, shorter, local_agent_name, req, &setup_.extraParams),
            NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(req, nullptr);
        EXPECT_TRUE(records("nixl::createXferReq").empty());
        EXPECT_TRUE(records("nixl::xferReq").empty());
    }

    TEST_F(recorderTracingAgentFixture, CreationFailedInPrepIsRecorded) {
        EXPECT_CALL(agent_helper_->getGMockEngine(), prepXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_ERR_BACKEND);
        EXPECT_EQ(req, nullptr);
        EXPECT_THAT(records("nixl::createXferReq"),
                    testing::ElementsAre(testing::Contains(
                        testing::AllOf(testing::StartsWith("nixl.xfer_request_id="),
                                       testing::Ne("nixl.xfer_request_id=0")))));
    }

    TEST_F(recorderTracingAgentFixture, RequestIdIsOnCreationPostAndCompletion) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        EXPECT_NE(req->traceable().requestIdentity().requestId, 0u);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::createXferReq"), testing::ElementsAre(identifies(req)));
        EXPECT_THAT(records("nixl::postXferReq.write"), testing::ElementsAre(postedAs(req, 1)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 1)));
        EXPECT_THAT(handedToPlugin(), testing::ElementsAre(handed(req, 0), handed(req, 1)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, RepostTakesTheNextExecution) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(req->traceable().postIdentity().execution, 2u);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(handedToPlugin(),
                    testing::ElementsAre(handed(req, 0), handed(req, 1), handed(req, 2)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, RejectedRepostRecordsNoSpanAndTakesNoExecution) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_EQ(agent->getXferStatus(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(handedToPlugin(),
                    testing::ElementsAre(handed(req, 0), handed(req, 1), handed(req, 2)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, FailedPostTakesAnExecution) {
        nixlAgent *agent = agent_helper_->getAgent();
        EXPECT_CALL(agent_helper_->getGMockEngine(), postXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_BACKEND);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 2)));
        EXPECT_THAT(handedToPlugin(),
                    testing::ElementsAre(handed(req, 0), handed(req, 1), handed(req, 2)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, PostSpanCarriesItsStatus) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_SUCCESS))
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_BACKEND);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(withStatus(NIXL_SUCCESS),
                                         withStatus(NIXL_IN_PROG),
                                         withStatus(NIXL_ERR_BACKEND)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, ImmediateFailureIsRecordedOnce) {
        nixlAgent *agent = agent_helper_->getAgent();
        EXPECT_CALL(agent_helper_->getGMockEngine(), postXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto failure = testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_BACKEND));

        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_BACKEND);
        EXPECT_THAT(records("nixl::xfer.error"), testing::ElementsAre(failure));
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_BACKEND);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xfer.error"), testing::ElementsAre(failure));
    }

    TEST_F(recorderTracingAgentFixture, FailureSeenByTheStatusCheckIsRecordedOnce) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_TRUE(records("nixl::xfer.error").empty());
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_BACKEND);
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_BACKEND);
        EXPECT_THAT(
            records("nixl::xfer.error"),
            testing::ElementsAre(testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_BACKEND))));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, RemoteDisconnectIsRecorded) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_ERR_REMOTE_DISCONNECT));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_REMOTE_DISCONNECT);
        EXPECT_THAT(records("nixl::xfer.error"),
                    testing::ElementsAre(
                        testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_REMOTE_DISCONNECT))));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, FailureSeenByARepostIsTheEarlierPosts) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(
            records("nixl::xfer.error"),
            testing::ElementsAre(testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_BACKEND))));
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(postedAs(req, 1), postedAs(req, 2)));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(postedAs(req, 2)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, FailureSeenByReleaseIsRecorded) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto failure = testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_BACKEND));
        const auto release = testing::AllOf(postedAs(req, 1), withStatus(NIXL_SUCCESS));

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xfer.error"), testing::ElementsAre(failure));
        EXPECT_THAT(records("nixl::releaseXferReq"), testing::ElementsAre(release));
    }

    TEST_F(recorderTracingAgentFixture, ReleaseThatCancelsRecordsACancel) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, releaseReqH).WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto cancel = testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_CANCELED));

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xfer.error"), testing::ElementsAre(cancel));
        EXPECT_TRUE(completionMarks().empty());
    }

    TEST_F(recorderTracingAgentFixture, RefusedReleaseAndItsRetryAreTwoRecords) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, releaseReqH).WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto refused = testing::AllOf(postedAs(req, 1), withStatus(NIXL_ERR_REPOST_ACTIVE));
        const auto released = testing::AllOf(postedAs(req, 1), withStatus(NIXL_SUCCESS));
        const auto completion = postedAs(req, 1);
        const auto request_span = postedAs(req, 1);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_TRUE(records("nixl::xferReq").empty());
        EXPECT_EQ(agent->getXferStatus(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xferReq"), testing::ElementsAre(request_span));
        EXPECT_THAT(records("nixl::releaseXferReq"), testing::ElementsAre(refused, released));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(completion));
        EXPECT_TRUE(records("nixl::xfer.error").empty());
    }

    TEST_F(recorderTracingAgentFixture, CompletionSeenByReleaseStaysSeparate) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto completion = postedAs(req, 1);
        const auto release = testing::AllOf(postedAs(req, 1), withStatus(NIXL_SUCCESS));

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(completionMarks(), testing::ElementsAre(completion));
        EXPECT_THAT(records("nixl::releaseXferReq"), testing::ElementsAre(release));
        EXPECT_TRUE(records("nixl::xfer.error").empty());
    }

    TEST_F(recorderTracingAgentFixture, RequestsHaveDistinctIds) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *first = nullptr;
        nixlXferReqH *second = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, first), NIXL_SUCCESS);
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, second), NIXL_SUCCESS);
        EXPECT_NE(first->traceable().requestIdentity().requestId, 0u);
        EXPECT_NE(second->traceable().requestIdentity().requestId, 0u);
        EXPECT_NE(first->traceable().requestIdentity().requestId,
                  second->traceable().requestIdentity().requestId);

        EXPECT_EQ(agent->postXferReq(second), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(first), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::createXferReq"),
                    testing::ElementsAre(identifies(first), identifies(second)));
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(postedAs(second, 1), postedAs(first, 1)));
        EXPECT_EQ(agent->releaseXferReq(first), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(second), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, PreppedRequestHasAnId) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup_.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup_.xferDlist, remote_side),
                  NIXL_SUCCESS);

        const std::vector<int> indices{0};
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(
            agent->makeXferReq(
                NIXL_WRITE, *local_side, indices, *remote_side, indices, req, &setup_.extraParams),
            NIXL_SUCCESS);
        EXPECT_NE(req->traceable().requestIdentity().requestId, 0u);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::makeXferReq"), testing::ElementsAre(identifies(req)));
        EXPECT_THAT(records("nixl::postXferReq.write"), testing::ElementsAre(postedAs(req, 1)));
        EXPECT_THAT(handedToPlugin(), testing::ElementsAre(handed(req, 0), handed(req, 1)));

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, RequestIdIsRecordedAsAnInteger) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *created = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, created), NIXL_SUCCESS);
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup_.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup_.xferDlist, remote_side),
                  NIXL_SUCCESS);
        const std::vector<int> indices{0};
        nixlXferReqH *made = nullptr;
        ASSERT_EQ(
            agent->makeXferReq(
                NIXL_WRITE, *local_side, indices, *remote_side, indices, made, &setup_.extraParams),
            NIXL_SUCCESS);

        for (const nixlXferReqH *req : {created, made}) {
            EXPECT_NE(req->traceable().requestIdentity().requestId, 0u);
            EXPECT_LE(req->traceable().requestIdentity().requestId,
                      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
        }
        EXPECT_THAT(records("nixl::createXferReq"), testing::ElementsAre(identifies(created)));
        EXPECT_THAT(records("nixl::makeXferReq"), testing::ElementsAre(identifies(made)));

        EXPECT_EQ(agent->releaseXferReq(made), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(created), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    class unsampledRecorderTracingAgentFixture : public recorderTracingAgentFixture {
    protected:
        [[nodiscard]] std::string
        sampleRatio() const override {
            return "0";
        }
    };

    TEST_F(unsampledRecorderTracingAgentFixture, UnsampledRequestRecordsNothing) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        ASSERT_FALSE(request.sampled());

        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_EQ(req->traceable().postIdentity().context, request);
        }
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        for (const std::string_view name : {"nixl::createXferReq",
                                            "nixl::postXferReq.write",
                                            "nixl::xfer.complete",
                                            "nixl::releaseXferReq",
                                            "nixl::xferReq"}) {
            EXPECT_TRUE(records(name).empty()) << name;
        }
    }

    TEST_F(unsampledRecorderTracingAgentFixture, SuppliedSampledContextIsRecorded) {
        nixlAgent *agent = agent_helper_->getAgent();
        setup_.extraParams.traceContext = nixl::trace::formatTraceparent(applicationContext("01"));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        ASSERT_TRUE(req->traceable().requestIdentity().context.sampled());

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        for (const std::string_view name : {"nixl::createXferReq",
                                            "nixl::postXferReq.write",
                                            "nixl::xfer.complete",
                                            "nixl::releaseXferReq",
                                            "nixl::xferReq"}) {
            EXPECT_EQ(records(name).size(), 1u) << name;
        }
    }

    TEST_F(unsampledRecorderTracingAgentFixture, SuppliedSampledPostContextIsRecorded) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        ASSERT_FALSE(req->traceable().requestIdentity().context.sampled());
        const auto application = applicationContext("01");
        nixl_opt_args_t with_context;
        with_context.traceContext = nixl::trace::formatTraceparent(application);

        EXPECT_EQ(agent->postXferReq(req, &with_context), NIXL_SUCCESS);
        const auto joined = req->traceable().postIdentity().context;
        const auto joined_span = testing::AllOf(isSpanOf(req, joined), childOf(application));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::postXferReq.write"), testing::ElementsAre(joined_span));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(joined)));
        for (const std::string_view name :
             {"nixl::createXferReq", "nixl::releaseXferReq", "nixl::xferReq"}) {
            EXPECT_TRUE(records(name).empty()) << name;
        }
    }

    TEST_F(recorderTracingAgentFixture, TraceInfoBeforeAnyPostNamesTheRequest) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        const auto info = traceInfo(req);
        EXPECT_THAT(records("nixl::createXferReq"),
                    testing::ElementsAre(testing::Contains("nixl.xfer_request_id=" +
                                                           std::to_string(info.requestId))));
        EXPECT_EQ(info.execution, 0u);
        EXPECT_EQ(info.traceparent,
                  nixl::trace::formatTraceparent(req->traceable().requestIdentity().context));
        EXPECT_THAT(info.traceparent, testing::EndsWith("-03"));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, TraceInfoFollowsEachAcceptedPost) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto first = traceInfo(req);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto second = traceInfo(req);
        EXPECT_EQ(first.execution, 1u);
        EXPECT_EQ(second.execution, 2u);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(first, "span="),
                                         carriesTraceInfo(second, "span=")));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, TraceInfoIsUnchangedByARefusedRepost) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        const auto posted = traceInfo(req);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_THAT(traceInfo(req),
                    testing::FieldsAre(posted.requestId, posted.execution, posted.traceparent));
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(posted, "span=")));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, TraceInfoCountsAPostTheBackendFails) {
        nixlAgent *agent = agent_helper_->getAgent();
        EXPECT_CALL(agent_helper_->getGMockEngine(), postXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_BACKEND);
        const auto info = traceInfo(req);
        EXPECT_EQ(info.execution, 1u);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(info, "span=")));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, TraceInfoNamesAPreppedRequest) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup_.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup_.xferDlist, remote_side),
                  NIXL_SUCCESS);

        const std::vector<int> indices{0};
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(
            agent->makeXferReq(
                NIXL_WRITE, *local_side, indices, *remote_side, indices, req, &setup_.extraParams),
            NIXL_SUCCESS);
        const auto prepped = traceInfo(req);
        EXPECT_THAT(records("nixl::makeXferReq"),
                    testing::ElementsAre(testing::Contains("nixl.xfer_request_id=" +
                                                           std::to_string(prepped.requestId))));
        EXPECT_EQ(prepped.execution, 0u);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto posted = traceInfo(req);
        EXPECT_EQ(posted.requestId, prepped.requestId);
        EXPECT_EQ(posted.execution, 1u);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(posted, "span=")));

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(recorderTracingAgentFixture, GetNotifsIsRecordedOnlyWithNotifications) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, getNotifs)
            .WillOnce(testing::Return(NIXL_SUCCESS))
            .WillOnce(testing::DoAll(testing::SetArgReferee<0>(notif_list_t{{"peer", "msg"}}),
                                     testing::Return(NIXL_SUCCESS)));
        nixl_notifs_t notifs;

        EXPECT_EQ(agent->getNotifs(notifs), NIXL_SUCCESS);
        EXPECT_TRUE(records("nixl::getNotifs").empty());

        EXPECT_EQ(agent->getNotifs(notifs), NIXL_SUCCESS);
        EXPECT_EQ(notifs["peer"].size(), 1u);
        EXPECT_EQ(records("nixl::getNotifs").size(), 1u);
    }

    class sampledRecorderTracingAgentFixture : public recorderTracingAgentFixture {
    protected:
        [[nodiscard]] std::string
        sampleRatio() const override {
            return "1";
        }
    };

    TEST_F(sampledRecorderTracingAgentFixture, EachPostCarriesItsOwnContext) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        ASSERT_TRUE(request.sampled());

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto first = req->traceable().postIdentity().context;
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(first));
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto second = req->traceable().postIdentity().context;
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(second));

        EXPECT_EQ(first.traceId, request.traceId);
        EXPECT_EQ(first.flags, request.flags);
        EXPECT_NE(first.spanId, request.spanId);
        EXPECT_EQ(first.parentSpanId, request.spanId);
        EXPECT_EQ(second.traceId, first.traceId);
        EXPECT_EQ(second.flags, first.flags);
        EXPECT_NE(second.spanId, first.spanId);
        EXPECT_EQ(second.parentSpanId, request.spanId);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(testing::AllOf(isSpanOf(req, first), childOf(request)),
                                         testing::AllOf(isSpanOf(req, second), childOf(request))));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(first), carries(second)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, ReleaseRecordsTheRequestSpanItsCallsDescendFrom) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        ASSERT_TRUE(request.sampled());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto request_span = testing::AllOf(isSpanOf(req, request), postedAs(req, 2));
        EXPECT_TRUE(records("nixl::xferReq").empty());

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        const auto names = recordNames();
        const auto release_at = std::ranges::find(names, "nixl::releaseXferReq");
        const auto request_at = std::ranges::find(names, "nixl::xferReq");
        ASSERT_NE(release_at, names.end());
        ASSERT_NE(request_at, names.end());
        EXPECT_LT(release_at - names.begin(), request_at - names.begin());
        EXPECT_THAT(
            records("nixl::xferReq"),
            testing::ElementsAre(testing::AllOf(
                request_span, testing::Not(testing::Contains(testing::StartsWith("parent="))))));
        EXPECT_THAT(records("nixl::createXferReq"), testing::ElementsAre(carries(request)));
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::AllOf(testing::SizeIs(2), testing::Each(childOf(request))));
        EXPECT_THAT(records("nixl::releaseXferReq"), testing::ElementsAre(carries(request)));
    }

    TEST_F(sampledRecorderTracingAgentFixture, CreationFailedInPrepEndsItsRequest) {
        EXPECT_CALL(agent_helper_->getGMockEngine(), prepXfer)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_ERR_BACKEND);

        const auto requests = records("nixl::xferReq");
        ASSERT_EQ(requests.size(), 1u);
        const auto request = contextOf(requests[0], "span=");
        EXPECT_TRUE(request.sampled());
        EXPECT_THAT(requests[0], testing::Contains("nixl.xfer_execution=0"));
        EXPECT_THAT(records("nixl::createXferReq"), testing::ElementsAre(carries(request)));
    }

    TEST_F(sampledRecorderTracingAgentFixture, SuppliedContextIsTheRequestsParent) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto application = applicationContext("01");
        setup_.extraParams.traceContext = nixl::trace::formatTraceparent(application);
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        EXPECT_EQ(request.traceId, application.traceId);
        EXPECT_NE(request.spanId, application.spanId);
        EXPECT_EQ(request.flags, application.flags);
        EXPECT_EQ(request.parentSpanId, application.spanId);

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto post = req->traceable().postIdentity().context;
        EXPECT_EQ(post.traceId, application.traceId);
        EXPECT_EQ(post.parentSpanId, request.spanId);
        const auto request_span = isSpanOf(req, request);
        const auto post_span = isSpanOf(req, post);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xferReq"),
                    testing::ElementsAre(testing::AllOf(request_span, childOf(application))));
        EXPECT_THAT(records("nixl::createXferReq"), testing::ElementsAre(carries(request)));
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(testing::AllOf(post_span, childOf(request))));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(post)));
    }

    TEST_F(sampledRecorderTracingAgentFixture, SuppliedPostContextIsThatPostsParent) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        const auto application = applicationContext("01");
        nixl_opt_args_t with_context;
        with_context.traceContext = nixl::trace::formatTraceparent(application);

        EXPECT_EQ(agent->postXferReq(req, &with_context), NIXL_SUCCESS);
        const auto joined = req->traceable().postIdentity().context;
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(joined));
        EXPECT_EQ(joined.traceId, application.traceId);
        EXPECT_EQ(joined.parentSpanId, application.spanId);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto own = req->traceable().postIdentity().context;
        EXPECT_EQ(own.traceId, request.traceId);
        EXPECT_EQ(own.parentSpanId, request.spanId);
        const auto joined_span = testing::AllOf(isSpanOf(req, joined), childOf(application));
        const auto own_span = testing::AllOf(isSpanOf(req, own), childOf(request));
        const auto request_span = testing::AllOf(isSpanOf(req, request), postedAs(req, 2));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);

        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(joined_span, own_span));
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(joined), carries(own)));
        EXPECT_THAT(records("nixl::xferReq"), testing::ElementsAre(request_span));
    }

    TEST_F(sampledRecorderTracingAgentFixture, SuppliedNotificationContextIsItsParent) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        const auto application = applicationContext("01");
        nixl_opt_args_t with_context;
        with_context.traceContext = nixl::trace::formatTraceparent(application);
        std::optional<nixl::trace::TraceIdentity> sent;
        EXPECT_CALL(engine, genNotif)
            .WillOnce(testing::Invoke([&sent](const std::string &, const std::string &) {
                if (const nixl::trace::TraceIdentity *identity = nixl::trace::currentIdentity()) {
                    sent = *identity;
                }
                return NIXL_SUCCESS;
            }));

        EXPECT_EQ(agent->genNotif(local_agent_name, "msg", &with_context), NIXL_SUCCESS);
        ASSERT_TRUE(sent.has_value());
        EXPECT_EQ(sent->context.traceId, application.traceId);
        EXPECT_NE(sent->context.spanId, application.spanId);
        EXPECT_EQ(sent->context.parentSpanId, application.spanId);
        EXPECT_EQ(sent->requestId, 0u);
        EXPECT_THAT(records("nixl::genNotif"),
                    testing::ElementsAre(testing::AllOf(
                        testing::Contains("span=" + nixl::trace::formatTraceparent(sent->context)),
                        childOf(application),
                        testing::Not(testing::Contains(testing::StartsWith("correlation="))))));
    }

    TEST_F(sampledRecorderTracingAgentFixture, SuppliedUnsampledNotificationContextRecordsNothing) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        nixl_opt_args_t with_context;
        with_context.traceContext = nixl::trace::formatTraceparent(applicationContext("00"));
        bool sampled = true;
        EXPECT_CALL(engine, genNotif)
            .WillOnce(testing::Invoke([&sampled](const std::string &, const std::string &) {
                const nixl::trace::TraceIdentity *identity = nixl::trace::currentIdentity();
                sampled = (identity != nullptr) && identity->context.sampled();
                return NIXL_SUCCESS;
            }));

        EXPECT_EQ(agent->genNotif(local_agent_name, "msg", &with_context), NIXL_SUCCESS);
        EXPECT_FALSE(sampled);
        EXPECT_TRUE(records("nixl::genNotif").empty());
    }

    TEST_F(sampledRecorderTracingAgentFixture, NotificationWithoutAContextIsAPlainSpan) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        bool scoped = true;
        EXPECT_CALL(engine, genNotif)
            .WillOnce(testing::Invoke([&scoped](const std::string &, const std::string &) {
                scoped = nixl::trace::currentIdentity() != nullptr;
                return NIXL_SUCCESS;
            }));

        EXPECT_EQ(agent->genNotif(local_agent_name, "msg"), NIXL_SUCCESS);
        EXPECT_FALSE(scoped);
        EXPECT_THAT(records("nixl::genNotif"),
                    testing::ElementsAre(testing::AllOf(
                        testing::Contains("remote_agent=" + std::string(local_agent_name)),
                        testing::Not(testing::Contains(testing::StartsWith("span="))),
                        testing::Not(testing::Contains(testing::StartsWith("context="))))));
    }

    TEST_F(sampledRecorderTracingAgentFixture, SuppliedUnsampledContextRecordsNothing) {
        nixlAgent *agent = agent_helper_->getAgent();
        setup_.extraParams.traceContext = nixl::trace::formatTraceparent(applicationContext("00"));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        ASSERT_FALSE(req->traceable().requestIdentity().context.sampled());

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        for (const std::string_view name : {"nixl::createXferReq",
                                            "nixl::postXferReq.write",
                                            "nixl::xfer.complete",
                                            "nixl::releaseXferReq",
                                            "nixl::xferReq"}) {
            EXPECT_TRUE(records(name).empty()) << name;
        }
    }

    TEST_F(sampledRecorderTracingAgentFixture, PreppedRequestJoinsASuppliedContext) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup_.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup_.xferDlist, remote_side),
                  NIXL_SUCCESS);
        const auto application = applicationContext("01");
        nixl_opt_args_t params = setup_.extraParams;
        params.traceContext = nixl::trace::formatTraceparent(application);

        const std::vector<int> indices{0};
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent->makeXferReq(
                      NIXL_WRITE, *local_side, indices, *remote_side, indices, req, &params),
                  NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;
        EXPECT_EQ(request.traceId, application.traceId);
        EXPECT_EQ(request.parentSpanId, application.spanId);
        EXPECT_EQ(traceInfo(req).traceparent, nixl::trace::formatTraceparent(request));
        EXPECT_THAT(records("nixl::makeXferReq"), testing::ElementsAre(carries(request)));

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, UnparsableSuppliedContextIsIgnoredOnce) {
        const LogMessageCounter ignored("ignoring a supplied trace context");
        nixlAgent *agent = agent_helper_->getAgent();
        setup_.extraParams.traceContext = "not a trace context";
        for (int i = 0; i < 2; ++i) {
            nixlXferReqH *req = nullptr;
            ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
            const auto request = req->traceable().requestIdentity().context;
            EXPECT_TRUE(request.sampled());
            EXPECT_EQ(request.parentSpanId, (std::array<std::uint8_t, 8>{}));
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }
        EXPECT_EQ(ignored.count(), 1u);
    }

    TEST_F(sampledRecorderTracingAgentFixture, RefusedRepostRecordsNoSpan) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);

        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        const auto first = req->traceable().postIdentity().context;
        const auto posted = traceInfo(req);
        EXPECT_EQ(agent->postXferReq(req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_EQ(req->traceable().postIdentity().context, first);
        EXPECT_THAT(traceInfo(req),
                    testing::FieldsAre(posted.requestId, posted.execution, posted.traceparent));
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto last = req->traceable().postIdentity().context;
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(last));

        const auto calls = records("nixl::postXferReq.write");
        ASSERT_EQ(calls.size(), 2u);
        EXPECT_THAT(calls[0], isSpanOf(req, first));
        EXPECT_THAT(calls[0], carriesTraceInfo(posted, "span="));
        EXPECT_THAT(calls[1], isSpanOf(req, last));
        EXPECT_EQ(last.traceId, first.traceId);
        EXPECT_NE(last.spanId, first.spanId);
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(first), carries(last)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, RefusedFirstCallRecordsNothing) {
        nixlAgent *agent = agent_helper_->getAgent();
        ON_CALL(agent_helper_->getGMockEngine(), supportsNotif())
            .WillByDefault(testing::Return(false));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;

        nixl_opt_args_t with_notif;
        with_notif.notif = "notification";
        EXPECT_EQ(agent->postXferReq(req, &with_notif), NIXL_ERR_BACKEND);
        nixl_opt_args_t without_notif;
        EXPECT_EQ(agent->postXferReq(req, &without_notif), NIXL_SUCCESS);

        const auto post = req->traceable().postIdentity().context;
        EXPECT_NE(post.spanId, request.spanId);
        EXPECT_EQ(post.parentSpanId, request.spanId);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(testing::AllOf(isSpanOf(req, post), childOf(request))));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, FailureNamesTheLatestPostAndReleaseTheRequest) {
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, postXfer)
            .WillOnce(testing::Return(NIXL_SUCCESS))
            .WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer).WillOnce(testing::Return(NIXL_ERR_BACKEND));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->postXferReq(req), NIXL_IN_PROG);
        EXPECT_EQ(agent->getXferStatus(req), NIXL_ERR_BACKEND);
        const auto post = req->traceable().postIdentity().context;
        EXPECT_NE(post.spanId, request.spanId);
        const auto posted = postedAs(req, 2);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(records("nixl::xfer.error"),
                    testing::ElementsAre(testing::AllOf(posted, carries(post))));
        EXPECT_THAT(records("nixl::releaseXferReq"),
                    testing::ElementsAre(testing::AllOf(posted, carries(request))));
    }

    TEST_F(sampledRecorderTracingAgentFixture, ReadPostIsTheSpanItsContextIdentifies) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req, NIXL_READ), NIXL_SUCCESS);
        ASSERT_TRUE(req->traceable().requestIdentity().context.sampled());

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto post = req->traceable().postIdentity().context;
        EXPECT_THAT(records("nixl::postXferReq.read"), testing::ElementsAre(isSpanOf(req, post)));
        EXPECT_TRUE(records("nixl::postXferReq.write").empty());
        EXPECT_THAT(completionMarks(), testing::ElementsAre(carries(post)));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, TraceInfoNamesTheLatestPostsSpan) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto created = traceInfo(req);
        EXPECT_EQ(created.traceparent,
                  nixl::trace::formatTraceparent(req->traceable().requestIdentity().context));
        EXPECT_THAT(created.traceparent, testing::EndsWith("-03"));

        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto first = traceInfo(req);
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        const auto second = traceInfo(req);
        EXPECT_NE(second.traceparent, first.traceparent);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(first, "span="),
                                         carriesTraceInfo(second, "span=")));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, TraceInfoAfterARefusedFirstCall) {
        nixlAgent *agent = agent_helper_->getAgent();
        ON_CALL(agent_helper_->getGMockEngine(), supportsNotif())
            .WillByDefault(testing::Return(false));
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, req), NIXL_SUCCESS);
        const auto request = req->traceable().requestIdentity().context;

        nixl_opt_args_t with_notif;
        with_notif.notif = "notification";
        EXPECT_EQ(agent->postXferReq(req, &with_notif), NIXL_ERR_BACKEND);
        const auto refused = traceInfo(req);
        EXPECT_EQ(refused.execution, 0u);
        EXPECT_EQ(refused.traceparent, nixl::trace::formatTraceparent(request));
        nixl_opt_args_t without_notif;
        EXPECT_EQ(agent->postXferReq(req, &without_notif), NIXL_SUCCESS);
        const auto posted = traceInfo(req);
        EXPECT_EQ(posted.execution, 1u);
        EXPECT_NE(posted.traceparent, refused.traceparent);
        EXPECT_THAT(records("nixl::postXferReq.write"),
                    testing::ElementsAre(carriesTraceInfo(posted, "span=")));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(sampledRecorderTracingAgentFixture, PluginIsHandedEachCallsIdentity) {
        nixlAgent *agent = agent_helper_->getAgent();
        nixlXferReqH *created = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup_, created), NIXL_SUCCESS);
        const auto created_context = created->traceable().requestIdentity().context;
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup_.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup_.xferDlist, remote_side),
                  NIXL_SUCCESS);
        const std::vector<int> indices{0};
        nixlXferReqH *made = nullptr;
        ASSERT_EQ(
            agent->makeXferReq(
                NIXL_WRITE, *local_side, indices, *remote_side, indices, made, &setup_.extraParams),
            NIXL_SUCCESS);
        const auto made_context = made->traceable().requestIdentity().context;

        EXPECT_EQ(agent->postXferReq(created), NIXL_SUCCESS);
        const auto first = created->traceable().postIdentity().context;
        EXPECT_EQ(agent->postXferReq(created), NIXL_SUCCESS);
        const auto second = created->traceable().postIdentity().context;
        EXPECT_EQ(agent->postXferReq(made), NIXL_SUCCESS);
        const auto made_post = made->traceable().postIdentity().context;
        EXPECT_NE(first.spanId, created_context.spanId);
        EXPECT_NE(second.spanId, created_context.spanId);
        EXPECT_NE(made_post.spanId, made_context.spanId);

        const auto identityOf = [](const nixl::trace::TraceContext &context,
                                   const nixlXferReqH *req,
                                   std::uint64_t execution) {
            return testing::Optional(testing::FieldsAre(
                context, req->traceable().requestIdentity().requestId, execution));
        };
        EXPECT_THAT(agent_helper_->getGMockEngine().observedTraceIdentities(),
                    testing::ElementsAre(identityOf(created_context, created, 0),
                                         identityOf(made_context, made, 0),
                                         identityOf(first, created, 1),
                                         identityOf(second, created, 2),
                                         identityOf(made_post, made, 1)));

        EXPECT_EQ(agent->releaseXferReq(made), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(created), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }
#endif

    class singleAgentSessionFixture : public testing::Test {
    protected:
        std::unique_ptr<agentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            agent_ = agent_helper_->getAgent();
        }
    };

    class dualAgentBridgeFixture : public testing::Test {
    protected:
        std::unique_ptr<agentHelper> local_agent_helper_, remote_agent_helper_;
        nixlAgent *local_agent_, *remote_agent_;

        void
        SetUp() override {
            local_agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            remote_agent_helper_ = std::make_unique<agentHelper>(remote_agent_name);
            local_agent_ = local_agent_helper_->getAgent();
            remote_agent_ = remote_agent_helper_->getAgent();
        }

        struct DualAgentSetup {
            nixlBackendH *local_backend = nullptr;
            nixlBackendH *remote_backend = nullptr;
            blob local_blob;
            blob remote_blob;
            nixl_reg_dlist_t local_reg_dlist;
            nixl_reg_dlist_t remote_reg_dlist;
            nixl_opt_args_t local_extra_params;
            nixl_opt_args_t remote_extra_params;
            nixl_b_params_t local_params;
            nixl_b_params_t remote_params;
            std::string remote_agent_name;

            explicit DualAgentSetup(nixl_mem_t mem_type)
                : local_reg_dlist(mem_type),
                  remote_reg_dlist(mem_type) {}
        };

        void
        setupDualAgent(DualAgentSetup &s, bool register_local = true, bool register_remote = true) {
            EXPECT_EQ(local_agent_helper_->createBackendWithGMock(s.local_params, s.local_backend),
                      NIXL_SUCCESS);
            EXPECT_EQ(
                remote_agent_helper_->createBackendWithGMock(s.remote_params, s.remote_backend),
                NIXL_SUCCESS);
            if (register_local) {
                EXPECT_EQ(
                    local_agent_helper_->initAndRegisterMemory(
                        s.local_blob, s.local_reg_dlist, s.local_extra_params, s.local_backend),
                    NIXL_SUCCESS);
            }
            if (register_remote) {
                EXPECT_EQ(
                    remote_agent_helper_->initAndRegisterMemory(
                        s.remote_blob, s.remote_reg_dlist, s.remote_extra_params, s.remote_backend),
                    NIXL_SUCCESS);
            }
            EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, s.remote_agent_name),
                      NIXL_SUCCESS);
        }
    };

    class singleAgentWithMemParamFixture : public testing::TestWithParam<nixl_mem_t> {
    protected:
        std::unique_ptr<agentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            agent_ = agent_helper_->getAgent();
        }
    };

    TEST_F(singleAgentSessionFixture, GetNonExistingPluginTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;

        EXPECT_NE(agent_->getPluginParams(nonexisting_plugin, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetExistingPluginTest) {
        std::vector<nixl_backend_t> plugins;
        EXPECT_EQ(agent_->getAvailPlugins(plugins), NIXL_SUCCESS);
        if (plugins.empty()) {
            GTEST_SKIP();
        }

        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(plugins.front(), mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, CreateNonExistingPluginBackendTest) {
        nixlPluginManager &plugin_manager = nixlPluginManager::getInstance();
        EXPECT_EQ(plugin_manager.loadBackendPlugin(nonexisting_plugin), nullptr);

        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_NE(agent_->createBackend(nonexisting_plugin, params, backend), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, CreateExistingPluginBackendTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(GetMockBackendName(), mem, params), NIXL_SUCCESS);

        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
    }

    // With no trace backend enabled the agent has no tracer, so a plugin is
    // handed a null sink and cannot record anything (NIX-1876).
    TEST_F(singleAgentSessionFixture, TracingOffLeavesTraceSinkNull) {
        nixl_b_params_t params;
        nixlBackendH *backend = nullptr;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        const auto observed = agent_helper_->getGMockEngine().observedTraceSink();
        ASSERT_TRUE(observed.has_value());
        EXPECT_EQ(*observed, nullptr);
    }

    TEST_F(singleAgentSessionFixture, TracingOffHandsPluginNoTraceIdentity) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        const auto &engine = agent_helper_->getGMockEngine();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_EQ(req->traceable().requestIdentity().context, nixl::trace::TraceContext{});
        EXPECT_FALSE(engine.observedTraceContext().has_value());

        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
        EXPECT_FALSE(engine.observedTraceContext().has_value());

        EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, PostGivenOnlyATraceContextKeepsTheRequestsNotification) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        const auto &engine = agent_helper_->getGMockEngine();
        ON_CALL(engine, supportsNotif()).WillByDefault(testing::Return(true));
        std::vector<std::string> sent;
        EXPECT_CALL(engine, postXfer)
            .Times(4)
            .WillRepeatedly([&sent](const nixl_xfer_op_t &,
                                    const nixl_meta_dlist_t &,
                                    const nixl_meta_dlist_t &,
                                    const std::string &,
                                    nixlBackendReqH *&,
                                    const nixl_opt_b_args_t *extra_args) {
                sent.push_back(extra_args->hasNotif ? extra_args->notifMsg : "none");
                return NIXL_SUCCESS;
            });
        setup.extraParams.notif = "created";
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);

        nixl_opt_args_t context_only;
        context_only.traceContext = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        EXPECT_EQ(agent_->postXferReq(req, &context_only), NIXL_SUCCESS);
        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
        nixl_opt_args_t context_and_param = context_only;
        context_and_param.customParam = "param";
        EXPECT_EQ(agent_->postXferReq(req, &context_and_param), NIXL_SUCCESS);
        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);

        EXPECT_THAT(sent, testing::ElementsAre("created", "created", "none", "none"));
        EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, TracingOffDrawsNoRequestId) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(req->traceable().postIdentity().requestId, 0u);
        EXPECT_EQ(req->traceable().postIdentity().execution, 0u);
        EXPECT_THAT(agent_helper_->getGMockEngine().observedTraceRequests(),
                    testing::ElementsAre(testing::Pair(0u, 0u), testing::Pair(0u, 0u)));
        const auto none =
            testing::Truly([](const std::optional<nixl::trace::TraceIdentity> &identity) {
                return !identity.has_value();
            });
        EXPECT_THAT(agent_helper_->getGMockEngine().observedTraceIdentities(),
                    testing::ElementsAre(none, none));
        EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, TraceInfoIsNotFoundWithoutTracing) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        nixl_xfer_trace_info_t info{1, 2, "untouched"};
        EXPECT_EQ(agent_->getXferTraceInfo(req, info), NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent_->getXferTraceInfo(req, info), NIXL_ERR_NOT_FOUND);
        EXPECT_THAT(info, testing::FieldsAre(1u, 2u, "untouched"));
        EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, TraceInfoRejectsANullHandle) {
        nixl_xfer_trace_info_t info{};
        EXPECT_EQ(agent_->getXferTraceInfo(nullptr, info), NIXL_ERR_INVALID_PARAM);
    }

#if !defined(NIXL_TRACE_ENABLED)
    TEST(tracingCompiledOut, AgentHooksAreEmpty) {
        static_assert(std::is_empty_v<nixl::trace::TracerPtr>);
        static_assert(std::is_empty_v<nixl::trace::BackendTraceSinkPtr>);
        static_assert(std::is_empty_v<nixl::trace::BackendTraces>);
        static_assert(std::is_empty_v<nixl::trace::Traceable>);
        static_assert(std::is_empty_v<nixlBackendPostTrace>);
    }

    TEST(tracingCompiledOut, AgentIgnoresTraceBackendsAndSuppliedContexts) {
        agentHelper helper(local_agent_name, "nvtx,recorder", "1");
        nixlAgent *agent = helper.getAgent();
        selfXferSetup setup;
        helper.setupSelfXfer(setup);
        const auto &engine = helper.getGMockEngine();
        const auto sink = engine.observedTraceSink();
        ASSERT_TRUE(sink.has_value());
        EXPECT_EQ(*sink, nullptr);
        ON_CALL(engine, supportsNotif()).WillByDefault(testing::Return(true));
        std::vector<std::string> sent;
        EXPECT_CALL(engine, postXfer)
            .WillOnce([&sent](const nixl_xfer_op_t &,
                              const nixl_meta_dlist_t &,
                              const nixl_meta_dlist_t &,
                              const std::string &,
                              nixlBackendReqH *&,
                              const nixl_opt_b_args_t *extra_args) {
                sent.push_back(extra_args->hasNotif ? extra_args->notifMsg : "none");
                return NIXL_SUCCESS;
            });

        const std::string context = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        setup.extraParams.notif = "created";
        setup.extraParams.traceContext = context;
        nixlXferReqH *req = nullptr;
        ASSERT_EQ(helper.createSelfXferReq(setup, req), NIXL_SUCCESS);
        nixl_opt_args_t context_only;
        context_only.traceContext = context;
        EXPECT_EQ(agent->postXferReq(req, &context_only), NIXL_SUCCESS);

        EXPECT_THAT(sent, testing::ElementsAre("created"));
        EXPECT_FALSE(engine.observedTraceContext().has_value());
        EXPECT_THAT(engine.observedTraceRequests(),
                    testing::ElementsAre(testing::Pair(0u, 0u), testing::Pair(0u, 0u)));
        nixl_xfer_trace_info_t info{1, 2, "untouched"};
        EXPECT_EQ(agent->getXferTraceInfo(req, info), NIXL_ERR_NOT_FOUND);
        EXPECT_THAT(info, testing::FieldsAre(1u, 2u, "untouched"));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
    }

    TEST(tracingCompiledOut, AgentWarnsOnlyWhenTracingIsRequested) {
        const auto warnings = [](const std::optional<std::string> &backends, bool under_nsys) {
            ScopedEnv env;
            if (backends) {
                env.addVar("NIXL_TRACE_BACKENDS", *backends);
            } else {
                env.unsetVar("NIXL_TRACE_BACKENDS");
            }
            if (under_nsys) {
                env.addVar("NVTX_INJECTION64_PATH", "/nonexistent/libInjectionNvtx64.so");
            } else {
                env.unsetVar("NVTX_INJECTION64_PATH");
            }
            const LogMessageCounter counter("-Dwith_trace=false");
            const nixlAgent agent(local_agent_name, nixlAgentConfig{});
            return counter.count();
        };

        EXPECT_EQ(warnings("nvtx", false), 1u);
        EXPECT_EQ(warnings(std::nullopt, true), 1u);
        EXPECT_EQ(warnings("", true), 0u);
        EXPECT_EQ(warnings(" , ", false), 0u);
        EXPECT_EQ(warnings(std::nullopt, false), 0u);
    }
#endif

    TEST_F(singleAgentSessionFixture, GetNonExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_NE(agent_->getBackendParams(nullptr, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        EXPECT_EQ(agent_->getBackendParams(backend, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetLocalMetadataTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);

        std::string metadata;
        EXPECT_EQ(agent_->getLocalMD(metadata), NIXL_SUCCESS);
        EXPECT_FALSE(metadata.empty());
    }

    TEST_P(singleAgentWithMemParamFixture, RegisterMemoryTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);

        blob blob;
        nixl_opt_args_t extra_params;
        nixl_reg_dlist_t reg_dlist(GetParam());
        EXPECT_EQ(agent_helper_->initAndRegisterMemory(blob, reg_dlist, extra_params, backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, RegisterDeregisterMemRepeatedTest) {
        constexpr int kWarmupIters = 3;
        constexpr int kTimedIters = 64;
        constexpr size_t kPoolSize = 128;

        using clock = std::chrono::steady_clock;
        using time_span = std::chrono::nanoseconds;

        nixl_opt_args_t extra_params;
        nixl_b_params_t params;
        nixlBackendH *backend;

        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        extra_params.backends.push_back(backend);

        std::vector<std::unique_ptr<blob>> pool;
        pool.resize(kPoolSize);
        for (auto &p : pool) {
            p = std::make_unique<blob>();
        }

        // Each round: registerMem once per pool entry, then deregisterMem once per entry.
        auto run_batch = [&](int rounds = 1) {
            for (int r = 0; r < rounds; ++r) {
                for (auto &bp : pool) {
                    nixl_reg_dlist_t reg_dlist{DRAM_SEG};
                    reg_dlist.addDesc(bp->getDesc());
                    EXPECT_EQ(agent_->registerMem(reg_dlist, &extra_params), NIXL_SUCCESS);
                }
                for (auto &bp : pool) {
                    nixl_reg_dlist_t reg_dlist{DRAM_SEG};
                    reg_dlist.addDesc(bp->getDesc());
                    EXPECT_EQ(agent_->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
                }
            }
        };

        // Warmup
        run_batch(kWarmupIters);

        // First measurement
        auto start = clock::now();
        run_batch();
        const int64_t timed1_ns =
            std::chrono::duration_cast<time_span>(clock::now() - start).count();

        // Many cycles
        run_batch(kTimedIters);

        // Second measurement
        start = clock::now();
        run_batch();
        const int64_t timed2_ns =
            std::chrono::duration_cast<time_span>(clock::now() - start).count();

        ASSERT_GT(timed1_ns, 0);
        ASSERT_GT(timed2_ns, 0);
        const double ratio = static_cast<double>(std::max(timed1_ns, timed2_ns)) /
            static_cast<double>(std::min(timed1_ns, timed2_ns));
        EXPECT_LE(ratio, 2.) << "timed batches differ by more than 100% "
                                "(ns1="
                             << timed1_ns << " ns2=" << timed2_ns << " ratio=" << ratio << ")";
    }

    TEST_F(singleAgentSessionFixture, PrepStrideDlistRejectsOverflowingSpan) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        const nixlBlobDesc desc = setup.localBlob.getDesc();

        nixl_stride_dlist_t valid(DRAM_SEG);
        valid.addDesc(nixlStrideDesc(desc.addr, 1, desc.devId, 2, desc.len / 2));
        nixlDlistH *dlist_hndl = nullptr;
        ASSERT_EQ(agent_->prepXferDlist(valid, dlist_hndl, &setup.extraParams), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(dlist_hndl), NIXL_SUCCESS);

        // (3 - 1) * 2^63 + 1 wraps to 1 in size_t.
        nixl_stride_dlist_t overflow(DRAM_SEG);
        overflow.addDesc(nixlStrideDesc(desc.addr, 1, desc.devId, size_t{1} << 63, 3));
        dlist_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(overflow, dlist_hndl, &setup.extraParams),
                  NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(dlist_hndl, nullptr);

        // addr + (2 - 1) * stride + 1 wraps to 0x1001 in uintptr_t.
        nixl_stride_dlist_t end_wrap(DRAM_SEG);
        end_wrap.addDesc(nixlStrideDesc(desc.addr, 1, desc.devId, 0 - desc.addr + 0x1000, 2));
        dlist_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(end_wrap, dlist_hndl, &setup.extraParams),
                  NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(dlist_hndl, nullptr);
    }

    INSTANTIATE_TEST_SUITE_P(DramRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(DRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(VramRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(VRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(BlkRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(BLK_SEG));
    INSTANTIATE_TEST_SUITE_P(ObjRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(OBJ_SEG));
    INSTANTIATE_TEST_SUITE_P(FileRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(FILE_SEG));

    TEST_F(dualAgentBridgeFixture, LoadRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_name, remote_agent_name_out);
    }

    TEST_F(dualAgentBridgeFixture, InvalidateRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->invalidateRemoteMD(remote_agent_name_out), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, XferReqTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlXferReqH *xfer_req;
        local_extra_params.notif = msg;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_WRITE,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              remote_agent_name_out,
                                              xfer_req,
                                              &local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceable().requestIdentity().requestId, 0u);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceable().postIdentity().requestId, 0u);

        nixl_opt_args_t repost_params;
        EXPECT_EQ(local_agent_->postXferReq(xfer_req, &repost_params), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceable().postIdentity().requestId, 0u);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, FailedActiveReleaseKeepsRequestPollable) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(s.local_blob.getDesc());
        remote_xfer_dlist.addDesc(s.remote_blob.getDesc());

        nixlBackendReqH backend_request;
        auto &engine = local_agent_helper_->getGMockEngine();
        EXPECT_CALL(engine, prepXfer)
            .WillOnce([&](const auto &,
                          const auto &,
                          const auto &,
                          const auto &,
                          nixlBackendReqH *&request,
                          const auto *) {
                request = &backend_request;
                return NIXL_SUCCESS;
            });
        EXPECT_CALL(engine, postXfer).WillOnce(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(engine, checkXfer)
            .WillOnce(testing::Return(NIXL_IN_PROG))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_CALL(engine, releaseReqH)
            .WillOnce(testing::Return(NIXL_ERR_BACKEND))
            .WillOnce(testing::Return(NIXL_SUCCESS));

        nixlXferReqH *xfer_req;
        EXPECT_EQ(
            local_agent_->createXferReq(
                NIXL_WRITE, local_xfer_dlist, remote_xfer_dlist, s.remote_agent_name, xfer_req),
            NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_IN_PROG);
        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_ERR_REPOST_ACTIVE);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, PrepMemViewRemoteDRAM) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s, /*register_local=*/false);

        nixl_remote_dlist_t remote_dlist(DRAM_SEG);
        remote_dlist.addDesc(nixlRemoteDesc(s.remote_blob.getDesc(), s.remote_agent_name));

        nixlMemViewH mvh = nullptr;
        EXPECT_EQ(local_agent_->prepMemView(remote_dlist, mvh), NIXL_SUCCESS);
        EXPECT_NE(mvh, nullptr);

        local_agent_->releaseMemView(mvh);
    }

    TEST_F(dualAgentBridgeFixture, PreparedTransfersPreserveBinaryBackendParameters) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);

        nixl_xfer_dlist_t local_descs(DRAM_SEG), remote_descs(DRAM_SEG);
        local_descs.addDesc(s.local_blob.getDesc());
        remote_descs.addDesc(s.remote_blob.getDesc());
        nixlDlistH *local_side = nullptr, *remote_side = nullptr;
        ASSERT_EQ(local_agent_->prepXferDlist(local_descs, local_side), NIXL_SUCCESS);
        ASSERT_EQ(local_agent_->prepXferDlist(s.remote_agent_name, remote_descs, remote_side),
                  NIXL_SUCCESS);

        s.local_extra_params.customParam = std::string("\x12\0\x34\0\x56\0\x78\0", 8);
        EXPECT_CALL(local_agent_helper_->getGMockEngine(),
                    prepXfer(testing::_,
                             testing::_,
                             testing::_,
                             testing::_,
                             testing::_,
                             testing::Pointee(testing::Field(&nixl_opt_b_args_t::customParam,
                                                             s.local_extra_params.customParam))))
            .Times(2);

        const std::vector<int> indices{0};
        nixlXferReqH *request = nullptr;
        ASSERT_EQ(local_agent_->makeXferReq(NIXL_WRITE,
                                            local_side,
                                            indices,
                                            remote_side,
                                            indices,
                                            request,
                                            &s.local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releaseXferReq(request), NIXL_SUCCESS);
        ASSERT_EQ(local_agent_->makeXferReq(NIXL_WRITE,
                                            *local_side,
                                            indices,
                                            *remote_side,
                                            indices,
                                            request,
                                            &s.local_extra_params),
                  NIXL_SUCCESS);
        nixl_opt_args_t post_params;
        for (const auto &custom_param : {std::string("\0\x87\0\x65\0\x43\0\x21", 8),
                                         std::string("\x12\0\x34\0", 4),
                                         std::string{}}) {
            post_params.customParam = custom_param;
            EXPECT_CALL(local_agent_helper_->getGMockEngine(),
                        postXfer(testing::_,
                                 testing::_,
                                 testing::_,
                                 testing::_,
                                 testing::_,
                                 testing::Pointee(testing::Field(&nixl_opt_b_args_t::customParam,
                                                                 custom_param))))
                .WillOnce(testing::Return(NIXL_SUCCESS));
            EXPECT_EQ(local_agent_->postXferReq(request, &post_params), NIXL_SUCCESS);
        }
        EXPECT_CALL(local_agent_helper_->getGMockEngine(),
                    postXfer(testing::_,
                             testing::_,
                             testing::_,
                             testing::_,
                             testing::_,
                             testing::Pointee(
                                 testing::Field(&nixl_opt_b_args_t::customParam, std::string{}))))
            .WillOnce(testing::Return(NIXL_SUCCESS));
        EXPECT_EQ(local_agent_->postXferReq(request), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releaseXferReq(request), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, XferReqSubFunctionsTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlDlistH *desc_hndl1, *desc_hndl2;
        EXPECT_EQ(local_agent_->prepXferDlist(local_xfer_dlist, desc_hndl1), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->prepXferDlist(remote_agent_name_out, remote_xfer_dlist, desc_hndl2),
                  NIXL_SUCCESS);

        std::vector<int> indices;
        for (int i = 0; i < local_xfer_dlist.descCount(); i++)
            indices.push_back(i);

        nixlXferReqH *xfer_req;
        local_extra_params.notif = msg;
        EXPECT_EQ(local_agent_->makeXferReq(NIXL_WRITE,
                                            *desc_hndl1,
                                            indices,
                                            *desc_hndl2,
                                            indices,
                                            xfer_req,
                                            &local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceable().requestIdentity().requestId, 0u);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceable().postIdentity().requestId, 0u);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl1), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl2), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, GenNotifTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->genNotif(remote_agent_name_out, msg), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);
    }

    TEST_F(dualAgentBridgeFixture, QueryXferBackendTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlXferReqH *xfer_req;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_WRITE,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              remote_agent_name_out,
                                              xfer_req,
                                              &local_extra_params),
                  NIXL_SUCCESS);

        nixlBackendH *backend_out;
        EXPECT_EQ(local_agent_->queryXferBackend(xfer_req, backend_out), NIXL_SUCCESS);
        EXPECT_EQ(backend_out, local_backend);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, MakeConnectionTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string local_agent_name_out, remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->getAndLoadRemoteMd(local_agent_, local_agent_name_out),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->makeConnection(remote_agent_name_out), NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_->makeConnection(local_agent_name_out), NIXL_SUCCESS);
    }

    TEST(agentMakeXferReq, MergeIndices) {
        constexpr size_t block = 64;
        constexpr size_t blocks = 20000;
        agentHelper helper(local_agent_name);
        nixlAgent *agent = helper.getAgent();
        nixlBackendH *backend = nullptr;
        nixl_b_params_t params;
        ASSERT_EQ(helper.createBackendWithGMock(params, backend), NIXL_SUCCESS);

        std::vector<char> memory(block * blocks);
        const auto base = reinterpret_cast<uintptr_t>(memory.data());
        nixl_reg_dlist_t reg_dlist(DRAM_SEG);
        reg_dlist.addDesc(nixlBlobDesc(base, memory.size(), 0));
        nixl_opt_args_t extra_params;
        extra_params.backends.push_back(backend);
        ASSERT_EQ(agent->registerMem(reg_dlist, &extra_params), NIXL_SUCCESS);

        nixl_stride_dlist_t strides(DRAM_SEG);
        strides.emplace(base, block, 0, block, blocks);
        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(strides, local_side, &extra_params), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, strides, remote_side, &extra_params),
                  NIXL_SUCCESS);

        struct piece {
            int local, remote, count;
        };

        struct testCase {
            std::vector<int> local, remote;
            std::vector<piece> pieces;
            bool skipMerge = false;
        };

        std::vector<int> all(blocks);
        std::iota(all.begin(), all.end(), 0);
        const std::vector<testCase> cases = {
            {{0, 1, 2, 3}, {4, 5, 6, 7}, {{0, 4, 4}}},
            {{0, 1, 5, 6, 7}, {0, 1, 2, 3, 4}, {{0, 0, 2}, {5, 2, 3}}},
            {{0, 1, 2}, {0, 1, 7}, {{0, 0, 2}, {2, 7, 1}}},
            {{0, 1, 2}, {0, 1, 2}, {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}}, true},
            {all, all, {{0, 0, static_cast<int>(blocks)}}},
        };
        for (const auto &c : cases) {
            nixl_meta_dlist_t src(DRAM_SEG), dst(DRAM_SEG);
            EXPECT_CALL(helper.getGMockEngine(), prepXfer)
                .WillOnce(testing::DoAll(testing::SaveArg<1>(&src),
                                         testing::SaveArg<2>(&dst),
                                         testing::Return(NIXL_SUCCESS)));
            extra_params.skipDescMerge = c.skipMerge;
            nixlXferReqH *req = nullptr;
            ASSERT_EQ(
                agent->makeXferReq(
                    NIXL_WRITE, *local_side, c.local, *remote_side, c.remote, req, &extra_params),
                NIXL_SUCCESS);
            ASSERT_EQ(static_cast<size_t>(src.descCount()), c.pieces.size());
            ASSERT_EQ(dst.descCount(), src.descCount());
            for (size_t i = 0; i < c.pieces.size(); ++i) {
                EXPECT_EQ(src[i].addr, base + c.pieces[i].local * block);
                EXPECT_EQ(dst[i].addr, base + c.pieces[i].remote * block);
                EXPECT_EQ(src[i].len, c.pieces[i].count * block);
                EXPECT_EQ(dst[i].len, c.pieces[i].count * block);
            }
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }

        extra_params.skipDescMerge = false;
        const std::vector<int> valid{0}, invalid{static_cast<int>(blocks)};
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(agent->makeXferReq(
                      NIXL_WRITE, *local_side, invalid, *remote_side, valid, req, &extra_params),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(agent->makeXferReq(
                      NIXL_WRITE, *local_side, valid, *remote_side, invalid, req, &extra_params),
                  NIXL_ERR_INVALID_PARAM);

        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
    }

} // namespace agent
} // namespace gtest
