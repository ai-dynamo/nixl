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
#include <absl/log/log_sink_registry.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <random>
#include <string_view>

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"
#include "transfer_request.h"
#include "tracing/trace_context.h"
#include "mocks/gmock_engine.h"

namespace {
constinit std::atomic<bool> entropy_fails{false};
} // namespace

extern "C" int
getentropy(void *buffer, size_t length) {
    if (entropy_fails.load()) {
        errno = ENOSYS;
        return -1;
    }
    static const auto next =
        reinterpret_cast<int (*)(void *, size_t)>(dlsym(RTLD_NEXT, "getentropy"));
    return next(buffer, length);
}

namespace gtest {
namespace agent {
    static constexpr const char *local_agent_name = "LocalAgent";
    static constexpr const char *remote_agent_name = "RemoteAgent";
    static constexpr const char *nonexisting_plugin = "NonExistingPlugin";
    static constexpr const char *untraced_log = "cannot carry trace contexts";
    static constexpr const char *context_failure_log = "could not generate a trace context";

    class logCounter final : public absl::LogSink {
    public:
        explicit logCounter(std::string_view needle) : needle_(needle) {
            absl::AddLogSink(this);
        }

        ~logCounter() override {
            absl::RemoveLogSink(this);
        }

        logCounter(const logCounter &) = delete;
        logCounter &
        operator=(const logCounter &) = delete;

        void
        Send(const absl::LogEntry &entry) override {
            if (entry.text_message().find(needle_) != std::string_view::npos) {
                count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        [[nodiscard]] size_t
        count() const noexcept {
            return count_.load(std::memory_order_relaxed);
        }

    private:
        const std::string needle_;
        std::atomic<size_t> count_ = 0;
    };

    class failingEntropy {
    public:
        failingEntropy() noexcept {
            entropy_fails.store(true);
        }

        ~failingEntropy() {
            entropy_fails.store(false);
        }

        failingEntropy(const failingEntropy &) = delete;
        failingEntropy &
        operator=(const failingEntropy &) = delete;
    };

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
        createSelfXferReq(selfXferSetup &setup, nixlXferReqH *&req) {
            return agent_->createXferReq(
                NIXL_WRITE, setup.xferDlist, setup.xferDlist, name_, req, &setup.extraParams);
        }
    };

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
        EXPECT_TRUE(first->traceContext().valid());
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(first->traceContext()));

        nixlXferReqH *second = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, second), NIXL_SUCCESS);
        ASSERT_NE(first->traceContext(), second->traceContext());
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(second->traceContext()));

        for (nixlXferReqH *req : {first, second, first}) {
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));
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
        EXPECT_TRUE(req->traceContext().valid());
        EXPECT_THAT(agent_helper_->getGMockEngine().observedTraceContext(),
                    testing::Optional(req->traceContext()));

        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(tracingEnabledAgentFixture, UnsampledRequestsAreNotReportedUntraced) {
        logCounter untraced(untraced_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_TRUE(req->traceContext().valid());
        EXPECT_FALSE(req->traceContext().sampled());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

    TEST_F(tracingEnabledAgentFixture, EntropyFailureLeavesRequestsUntraced) {
        logCounter failures(context_failure_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();

        nixlDlistH *local_side = nullptr;
        nixlDlistH *remote_side = nullptr;
        ASSERT_EQ(agent->prepXferDlist(setup.xferDlist, local_side), NIXL_SUCCESS);
        ASSERT_EQ(agent->prepXferDlist(local_agent_name, setup.xferDlist, remote_side),
                  NIXL_SUCCESS);
        const std::vector<int> indices{0};

        nixlXferReqH *created = nullptr;
        nixlXferReqH *made = nullptr;
        {
            const failingEntropy entropy;
            ASSERT_EQ(agent_helper_->createSelfXferReq(setup, created), NIXL_SUCCESS);
            ASSERT_EQ(agent->makeXferReq(NIXL_WRITE,
                                         *local_side,
                                         indices,
                                         *remote_side,
                                         indices,
                                         made,
                                         &setup.extraParams),
                      NIXL_SUCCESS);
        }
        EXPECT_EQ(failures.count(), 1u);

        for (nixlXferReqH *req : {created, made}) {
            EXPECT_EQ(req->traceContext(), nixl::trace::TraceContext{});
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_THAT(engine.observedTraceContext(),
                        testing::Optional(nixl::trace::TraceContext{}));
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }

        nixlXferReqH *recovered = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, recovered), NIXL_SUCCESS);
        EXPECT_TRUE(recovered->traceContext().valid());
        EXPECT_EQ(agent->releaseXferReq(recovered), NIXL_SUCCESS);
        EXPECT_EQ(failures.count(), 1u);

        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(sampledTracingAgentFixture, UnsupportedBackendIsReportedOnce) {
        logCounter untraced(untraced_log);
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();
        const auto &engine = agent_helper_->getGMockEngine();

        for (int i = 0; i < 3; ++i) {
            nixlXferReqH *req = nullptr;
            ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
            EXPECT_TRUE(req->traceContext().sampled());
            EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
            EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }
        EXPECT_EQ(untraced.count(), 1u);
    }

    TEST_F(sampledTracingAgentFixture, UnsupportedBackendIsReportedOnceForPreppedRequests) {
        logCounter untraced(untraced_log);
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
            EXPECT_TRUE(req->traceContext().sampled());
            EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        }
        EXPECT_EQ(untraced.count(), 1u);

        EXPECT_EQ(agent->releasedDlistH(local_side), NIXL_SUCCESS);
        EXPECT_EQ(agent->releasedDlistH(remote_side), NIXL_SUCCESS);
    }

    TEST_F(sampledTracingAgentFixture, SupportingBackendSeesTheSampledContextUnchanged) {
        logCounter untraced(untraced_log);
        const auto &engine = agent_helper_->getGMockEngine();
        ON_CALL(engine, supportsTraceContext()).WillByDefault(testing::Return(true));
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        ASSERT_TRUE(req->traceContext().sampled());
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

    TEST_F(sampledTracingAgentFixture, LocalOnlyBackendIsNotReportedUntraced) {
        logCounter untraced(untraced_log);
        ON_CALL(agent_helper_->getGMockEngine(), supportsRemote())
            .WillByDefault(testing::Return(false));
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        nixlAgent *agent = agent_helper_->getAgent();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_TRUE(req->traceContext().sampled());
        EXPECT_EQ(agent->postXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(agent->releaseXferReq(req), NIXL_SUCCESS);
        EXPECT_EQ(untraced.count(), 0u);
    }

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

    TEST_F(singleAgentSessionFixture, TracingOffHandsPluginTheZeroedRequestContext) {
        selfXferSetup setup;
        agent_helper_->setupSelfXfer(setup);
        const auto &engine = agent_helper_->getGMockEngine();

        nixlXferReqH *req = nullptr;
        ASSERT_EQ(agent_helper_->createSelfXferReq(setup, req), NIXL_SUCCESS);
        EXPECT_EQ(req->traceContext(), nixl::trace::TraceContext{});
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));

        EXPECT_EQ(agent_->postXferReq(req), NIXL_SUCCESS);
        EXPECT_THAT(engine.observedTraceContext(), testing::Optional(req->traceContext()));

        EXPECT_EQ(agent_->releaseXferReq(req), NIXL_SUCCESS);
    }

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
        EXPECT_EQ(xfer_req->traceCorrelationId64(), 0u);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceCorrelationId64(), 0u);

        nixl_opt_args_t repost_params;
        EXPECT_EQ(local_agent_->postXferReq(xfer_req, &repost_params), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceCorrelationId64(), 0u);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

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
        EXPECT_EQ(xfer_req->traceCorrelationId64(), 0u);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(xfer_req->traceCorrelationId64(), 0u);

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

} // namespace agent
} // namespace gtest
