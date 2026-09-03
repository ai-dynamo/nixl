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
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"
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

    class agentHelper {
    protected:
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;
        std::unique_ptr<nixlAgent> agent_;

    public:
        explicit agentHelper(const std::string &name,
                             std::chrono::microseconds xfer_stall_timeout =
                                 nixlAgentConfig::kDefaultXferStallTimeout)
            : agent_([&]() {
                  nixlAgentConfig cfg;
                  cfg.useProgThread = true;
                  cfg.xferStallTimeout = xfer_stall_timeout;
                  return std::make_unique<nixlAgent>(name, cfg);
              }()) {}

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
    };

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

        virtual std::chrono::microseconds
        getXferStallTimeout() const {
            return nixlAgentConfig::kDefaultXferStallTimeout;
        }

        void
        SetUp() override {
            local_agent_helper_ =
                std::make_unique<agentHelper>(local_agent_name, getXferStallTimeout());
            remote_agent_helper_ =
                std::make_unique<agentHelper>(remote_agent_name, getXferStallTimeout());
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

    /* Same as dualAgentBridgeFixture, but with a short transfer stall timeout so a
       transfer that never completes can be observed without a long test. */
    class dualAgentStallFixture : public dualAgentBridgeFixture {
    protected:
        static constexpr std::chrono::microseconds stall_timeout{50000};

        std::chrono::microseconds
        getXferStallTimeout() const override {
            return stall_timeout;
        }

        /* Makes the local backend accept the post but never complete it and never report an
           error, which is how a stalled peer, or a failed endpoint whose request the backend
           left outstanding, appears to NIXL. */
        void
        stallLocalBackend() {
            EXPECT_CALL(local_agent_helper_->getGMockEngine(), postXfer)
                .WillRepeatedly(testing::Return(NIXL_IN_PROG));
            EXPECT_CALL(local_agent_helper_->getGMockEngine(), checkXfer)
                .WillRepeatedly(testing::Return(NIXL_IN_PROG));
        }

        nixlXferReqH *
        postXfer(DualAgentSetup &s) {
            nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
            local_xfer_dlist.addDesc(s.local_blob.getDesc());
            remote_xfer_dlist.addDesc(s.remote_blob.getDesc());

            nixlXferReqH *xfer_req = nullptr;
            EXPECT_EQ(local_agent_->createXferReq(NIXL_READ,
                                                  local_xfer_dlist,
                                                  remote_xfer_dlist,
                                                  s.remote_agent_name,
                                                  xfer_req,
                                                  &s.local_extra_params),
                      NIXL_SUCCESS);
            EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_IN_PROG);
            return xfer_req;
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
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentStallFixture, XferStalledPastTimeoutIsReported) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);
        stallLocalBackend();

        nixlXferReqH *xfer_req = postXfer(s);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_IN_PROG);

        std::this_thread::sleep_for(stall_timeout + std::chrono::milliseconds(20));

        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_ERR_XFER_STALLED);
        /* Sticky: once reported the caller keeps seeing the stall rather than flipping back
           to a healthy-looking NIXL_IN_PROG on the next poll. */
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_ERR_XFER_STALLED);

        /* A stalled transfer is still posted, so releasing it must take the same cancel
           path as one reported in progress. */
        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentStallFixture, StalledXferCannotBeReposted) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);

        /* The backend must see exactly one post. A stalled transfer is still posted, so the
           repost below has to be rejected by the agent instead of reaching a backend request
           that can still be in flight. */
        EXPECT_CALL(local_agent_helper_->getGMockEngine(), postXfer)
            .Times(1)
            .WillRepeatedly(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(local_agent_helper_->getGMockEngine(), checkXfer)
            .WillRepeatedly(testing::Return(NIXL_IN_PROG));

        nixlXferReqH *xfer_req = postXfer(s);
        std::this_thread::sleep_for(stall_timeout + std::chrono::milliseconds(20));
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_ERR_XFER_STALLED);

        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_ERR_REPOST_ACTIVE);

        /* The rejected repost leaves the stall visible instead of reverting it. */
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_ERR_XFER_STALLED);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentStallFixture, XferWithinTimeoutStaysInProgress) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);
        stallLocalBackend();

        nixlXferReqH *xfer_req = postXfer(s);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_IN_PROG);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, XferStallTimeoutDisabledByDefault) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);
        EXPECT_CALL(local_agent_helper_->getGMockEngine(), postXfer)
            .WillRepeatedly(testing::Return(NIXL_IN_PROG));
        EXPECT_CALL(local_agent_helper_->getGMockEngine(), checkXfer)
            .WillRepeatedly(testing::Return(NIXL_IN_PROG));

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(s.local_blob.getDesc());
        remote_xfer_dlist.addDesc(s.remote_blob.getDesc());

        nixlXferReqH *xfer_req = nullptr;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_READ,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              s.remote_agent_name,
                                              xfer_req,
                                              &s.local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_IN_PROG);

        std::this_thread::sleep_for(std::chrono::milliseconds(70));

        /* Default configuration keeps the pre-existing behaviour of polling indefinitely. */
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_IN_PROG);
        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentStallFixture, CompletedXferIsNotReportedStalled) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(s.local_blob.getDesc());
        remote_xfer_dlist.addDesc(s.remote_blob.getDesc());

        nixlXferReqH *xfer_req = nullptr;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_READ,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              s.remote_agent_name,
                                              xfer_req,
                                              &s.local_extra_params),
                  NIXL_SUCCESS);
        /* Default GMock behaviour completes the post inline. */
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);

        std::this_thread::sleep_for(stall_timeout + std::chrono::milliseconds(20));

        /* The deadline only applies while a transfer is still in progress. */
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
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);

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
