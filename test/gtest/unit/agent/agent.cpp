/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include <algorithm>

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
        agentHelper(const std::string &name)
            : agent_(std::make_unique<nixlAgent>(name, nixlAgentConfig(true))) {}

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

        void
        SetUp() override {
            local_agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            remote_agent_helper_ = std::make_unique<agentHelper>(remote_agent_name);
            local_agent_ = local_agent_helper_->getAgent();
            remote_agent_ = remote_agent_helper_->getAgent();
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
        EXPECT_EQ(plugin_manager.loadPlugin(nonexisting_plugin), nullptr);

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

    TEST_F(singleAgentSessionFixture, CreateMultipleBackendsTest) {
        // Create first mock backend that supports DRAM_SEG and VRAM_SEG
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine1;
        nixl_mem_list_t mem_types1 = {DRAM_SEG, VRAM_SEG};
        ON_CALL(gmock_engine1, getSupportedMems()).WillByDefault(testing::Return(mem_types1));

        nixl_b_params_t params1;
        gmock_engine1.SetToParams(params1);
        nixlBackendH *backend1;
        EXPECT_EQ(agent_->createBackend(GetMockBackendName(), params1, backend1), NIXL_SUCCESS);
        EXPECT_NE(backend1, nullptr);

        // Create second mock backend that supports DRAM_SEG and FILE_SEG
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine2;
        nixl_mem_list_t mem_types2 = {DRAM_SEG, FILE_SEG};
        ON_CALL(gmock_engine2, getSupportedMems()).WillByDefault(testing::Return(mem_types2));

        nixl_b_params_t params2;
        gmock_engine2.SetToParams(params2);
        nixlBackendH *backend2;
        EXPECT_EQ(agent_->createBackend(GetMockBackendName2(), params2, backend2), NIXL_SUCCESS);
        EXPECT_NE(backend2, nullptr);

        // Verify they are different backend instances
        EXPECT_NE(backend1, backend2);

        // Verify we can get parameters for both backends and check supported memory types
        nixl_mem_list_t mem1, mem2;
        nixl_b_params_t params_out1, params_out2;
        EXPECT_EQ(agent_->getBackendParams(backend1, mem1, params_out1), NIXL_SUCCESS);
        EXPECT_EQ(agent_->getBackendParams(backend2, mem2, params_out2), NIXL_SUCCESS);

        // Verify backend1 supports DRAM_SEG and VRAM_SEG
        EXPECT_EQ(mem1.size(), 2u);
        EXPECT_NE(std::find(mem1.begin(), mem1.end(), DRAM_SEG), mem1.end());
        EXPECT_NE(std::find(mem1.begin(), mem1.end(), VRAM_SEG), mem1.end());

        // Verify backend2 supports DRAM_SEG and FILE_SEG
        EXPECT_EQ(mem2.size(), 2u);
        EXPECT_NE(std::find(mem2.begin(), mem2.end(), DRAM_SEG), mem2.end());
        EXPECT_NE(std::find(mem2.begin(), mem2.end(), FILE_SEG), mem2.end());
    }

    TEST_F(singleAgentSessionFixture, MultipleBackendsMemoryAndTransferTest) {
        // Create first mock backend that supports DRAM_SEG and VRAM_SEG
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine1;
        nixl_mem_list_t mem_types1 = {DRAM_SEG, VRAM_SEG};
        ON_CALL(gmock_engine1, getSupportedMems()).WillByDefault(testing::Return(mem_types1));

        nixl_b_params_t params1;
        gmock_engine1.SetToParams(params1);
        nixlBackendH *backend1;
        EXPECT_EQ(agent_->createBackend(GetMockBackendName(), params1, backend1), NIXL_SUCCESS);
        EXPECT_NE(backend1, nullptr);

        // Create second mock backend that supports DRAM_SEG and FILE_SEG
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine2;
        nixl_mem_list_t mem_types2 = {DRAM_SEG, FILE_SEG};
        ON_CALL(gmock_engine2, getSupportedMems()).WillByDefault(testing::Return(mem_types2));

        nixl_b_params_t params2;
        gmock_engine2.SetToParams(params2);
        nixlBackendH *backend2;
        EXPECT_EQ(agent_->createBackend(GetMockBackendName2(), params2, backend2), NIXL_SUCCESS);
        EXPECT_NE(backend2, nullptr);

        // Create memory blobs for each type
        blob dram_blob, vram_blob, file_blob;

        // Register DRAM memory without specifying backend - both backends support DRAM
        nixl_reg_dlist_t dram_list(DRAM_SEG);
        dram_list.addDesc(dram_blob.getDesc());
        EXPECT_EQ(agent_->registerMem(dram_list), NIXL_SUCCESS);

        // Register VRAM memory without specifying backend - only backend1 supports VRAM
        nixl_reg_dlist_t vram_list(VRAM_SEG);
        vram_list.addDesc(vram_blob.getDesc());
        EXPECT_EQ(agent_->registerMem(vram_list), NIXL_SUCCESS);

        // Register FILE memory without specifying backend - only backend2 supports FILE
        nixl_reg_dlist_t file_list(FILE_SEG);
        file_list.addDesc(file_blob.getDesc());
        EXPECT_EQ(agent_->registerMem(file_list), NIXL_SUCCESS);

        // Verify memory was registered with correct backends by preparing transfer descriptors
        // and attempting to create transfer requests for each combination

        // Prepare LOCAL transfer descriptor lists for each memory type
        nixl_xfer_dlist_t dram_xfer_list(DRAM_SEG);
        dram_xfer_list.addDesc(nixlBasicDesc(dram_blob.getDesc()));
        nixlDlistH *dram_local_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(NIXL_INIT_AGENT, dram_xfer_list, dram_local_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(dram_local_hndl, nullptr);

        nixl_xfer_dlist_t vram_xfer_list(VRAM_SEG);
        vram_xfer_list.addDesc(nixlBasicDesc(vram_blob.getDesc()));
        nixlDlistH *vram_local_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(NIXL_INIT_AGENT, vram_xfer_list, vram_local_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(vram_local_hndl, nullptr);

        nixl_xfer_dlist_t file_xfer_list(FILE_SEG);
        file_xfer_list.addDesc(nixlBasicDesc(file_blob.getDesc()));
        nixlDlistH *file_local_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(NIXL_INIT_AGENT, file_xfer_list, file_local_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(file_local_hndl, nullptr);

        // Prepare REMOTE transfer descriptor lists for loopback transfers (using agent's own name)
        nixlDlistH *dram_remote_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(local_agent_name, dram_xfer_list, dram_remote_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(dram_remote_hndl, nullptr);

        nixlDlistH *vram_remote_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(local_agent_name, vram_xfer_list, vram_remote_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(vram_remote_hndl, nullptr);

        nixlDlistH *file_remote_hndl = nullptr;
        EXPECT_EQ(agent_->prepXferDlist(local_agent_name, file_xfer_list, file_remote_hndl),
                  NIXL_SUCCESS);
        EXPECT_NE(file_remote_hndl, nullptr);

        // Verify DRAM was registered with backend1 by creating a loopback transfer request
        std::vector<int> indices = {0};
        nixlXferReqH *xfer_req1 = nullptr;
        nixl_opt_args_t extra_params1;
        extra_params1.backends.push_back(backend1);
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      dram_local_hndl,
                                      indices,
                                      dram_remote_hndl,
                                      indices,
                                      xfer_req1,
                                      &extra_params1),
                  NIXL_SUCCESS)
            << "DRAM should be registered with backend1";
        EXPECT_NE(xfer_req1, nullptr);
        EXPECT_EQ(agent_->releaseXferReq(xfer_req1), NIXL_SUCCESS);

        // Verify DRAM was registered with backend2 by creating a loopback transfer request
        nixlXferReqH *xfer_req2 = nullptr;
        nixl_opt_args_t extra_params2;
        extra_params2.backends.push_back(backend2);
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      dram_local_hndl,
                                      indices,
                                      dram_remote_hndl,
                                      indices,
                                      xfer_req2,
                                      &extra_params2),
                  NIXL_SUCCESS)
            << "DRAM should be registered with backend2";
        EXPECT_NE(xfer_req2, nullptr);
        EXPECT_EQ(agent_->releaseXferReq(xfer_req2), NIXL_SUCCESS);

        // Verify VRAM was registered with backend1 only
        nixlXferReqH *xfer_req3 = nullptr;
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      vram_local_hndl,
                                      indices,
                                      vram_remote_hndl,
                                      indices,
                                      xfer_req3,
                                      &extra_params1),
                  NIXL_SUCCESS)
            << "VRAM should be registered with backend1";
        EXPECT_NE(xfer_req3, nullptr);
        EXPECT_EQ(agent_->releaseXferReq(xfer_req3), NIXL_SUCCESS);

        // Verify VRAM was NOT registered with backend2
        nixlXferReqH *xfer_req4 = nullptr;
        EXPECT_NE(agent_->makeXferReq(NIXL_WRITE,
                                      vram_local_hndl,
                                      indices,
                                      vram_remote_hndl,
                                      indices,
                                      xfer_req4,
                                      &extra_params2),
                  NIXL_SUCCESS)
            << "VRAM should NOT be registered with backend2";

        // Verify FILE was registered with backend2 only
        nixlXferReqH *xfer_req5 = nullptr;
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      file_local_hndl,
                                      indices,
                                      file_remote_hndl,
                                      indices,
                                      xfer_req5,
                                      &extra_params2),
                  NIXL_SUCCESS)
            << "FILE should be registered with backend2";
        EXPECT_NE(xfer_req5, nullptr);
        EXPECT_EQ(agent_->releaseXferReq(xfer_req5), NIXL_SUCCESS);

        // Verify FILE was NOT registered with backend1
        nixlXferReqH *xfer_req6 = nullptr;
        EXPECT_NE(agent_->makeXferReq(NIXL_WRITE,
                                      file_local_hndl,
                                      indices,
                                      file_remote_hndl,
                                      indices,
                                      xfer_req6,
                                      &extra_params1),
                  NIXL_SUCCESS)
            << "FILE should NOT be registered with backend1";

        // Test cross-memory-type transfers

        // 1) FILE to VRAM should fail (no common backend: FILE on backend2, VRAM on backend1)
        nixlXferReqH *xfer_req7 = nullptr;
        EXPECT_NE(agent_->makeXferReq(NIXL_WRITE,
                                      file_local_hndl,
                                      indices,
                                      vram_remote_hndl,
                                      indices,
                                      xfer_req7,
                                      nullptr),
                  NIXL_SUCCESS)
            << "FILE to VRAM should fail - no common backend";

        // 2) DRAM to VRAM without backend specified should succeed (both on backend1)
        nixlXferReqH *xfer_req8 = nullptr;
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      dram_local_hndl,
                                      indices,
                                      vram_remote_hndl,
                                      indices,
                                      xfer_req8,
                                      nullptr),
                  NIXL_SUCCESS)
            << "DRAM to VRAM should succeed - both on backend1";
        EXPECT_NE(xfer_req8, nullptr);

        // Verify the transfer uses backend1
        nixlBackendH *backend_used = nullptr;
        EXPECT_EQ(agent_->queryXferBackend(xfer_req8, backend_used), NIXL_SUCCESS);
        EXPECT_EQ(backend_used, backend1) << "DRAM to VRAM should use backend1";
        EXPECT_EQ(agent_->releaseXferReq(xfer_req8), NIXL_SUCCESS);

        // 3) FILE to DRAM without backend specified should succeed (both on backend2)
        nixlXferReqH *xfer_req9 = nullptr;
        EXPECT_EQ(agent_->makeXferReq(NIXL_WRITE,
                                      file_local_hndl,
                                      indices,
                                      dram_remote_hndl,
                                      indices,
                                      xfer_req9,
                                      nullptr),
                  NIXL_SUCCESS)
            << "FILE to DRAM should succeed - both on backend2";
        EXPECT_NE(xfer_req9, nullptr);

        // Verify the transfer uses backend2
        backend_used = nullptr;
        EXPECT_EQ(agent_->queryXferBackend(xfer_req9, backend_used), NIXL_SUCCESS);
        EXPECT_EQ(backend_used, backend2) << "FILE to DRAM should use backend2";
        EXPECT_EQ(agent_->releaseXferReq(xfer_req9), NIXL_SUCCESS);

        // Release all descriptor list handles
        EXPECT_EQ(agent_->releasedDlistH(dram_local_hndl), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(dram_remote_hndl), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(vram_local_hndl), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(vram_remote_hndl), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(file_local_hndl), NIXL_SUCCESS);
        EXPECT_EQ(agent_->releasedDlistH(file_remote_hndl), NIXL_SUCCESS);

        // Deregister all memory
        EXPECT_EQ(agent_->deregisterMem(dram_list), NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(vram_list), NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(file_list), NIXL_SUCCESS);
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
        local_extra_params.notifMsg = msg;
        local_extra_params.hasNotif = true;
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
        EXPECT_EQ(local_agent_->prepXferDlist(NIXL_INIT_AGENT, local_xfer_dlist, desc_hndl1),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->prepXferDlist(remote_agent_name_out, remote_xfer_dlist, desc_hndl2),
                  NIXL_SUCCESS);

        std::vector<int> indices;
        for (int i = 0; i < local_xfer_dlist.descCount(); i++)
            indices.push_back(i);

        nixlXferReqH *xfer_req;
        local_extra_params.notifMsg = msg;
        local_extra_params.hasNotif = true;
        EXPECT_EQ(local_agent_->makeXferReq(NIXL_WRITE,
                                            desc_hndl1,
                                            indices,
                                            desc_hndl2,
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
