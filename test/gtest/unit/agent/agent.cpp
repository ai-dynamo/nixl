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

    class Blob {
    protected:
        static constexpr size_t buf_len = 256;
        static constexpr uint32_t dev_id = 0;

        std::unique_ptr<char[]> buf_;
        const nixlBlobDesc desc_;
        const char buf_pattern_;

    public:
        Blob()
            : buf_(std::make_unique<char[]>(buf_len)),
              desc_(reinterpret_cast<uintptr_t>(buf_.get()), buf_len, dev_id),
              buf_pattern_(GetRandomByte()) {
            memset(buf_.get(), buf_pattern_, buf_len);
        }

        nixlBlobDesc
        GetDesc() const {
            return desc_;
        }
    };

    class AgentHelper {
    protected:
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;
        std::unique_ptr<nixlAgent> agent_;

    public:
        AgentHelper(const std::string &name)
            : agent_(std::make_unique<nixlAgent>(name, nixlAgentConfig(true))) {}

        ~AgentHelper() {
            /* We must release nixlAgent first (i.e. explicitly in the destructor), as it calls
               cleanup functions in gmock_engine, which must stay alive during the process. */
            agent_.reset();
        }

        nixlAgent *
        GetAgent() const {
            return agent_.get();
        }

        const mocks::GMockBackendEngine &
        GetGMockEngine() const {
            return gmock_engine_;
        }

        nixl_status_t
        CreateBackendWithGMock(nixl_b_params_t &params, nixlBackendH *&backend) {
            gmock_engine_.SetToParams(params);
            return agent_->createBackend(GetMockBackendName(), params, backend);
        }

        nixl_status_t
        GetAndLoadRemoteMD(nixlAgent *remote_agent, std::string &remote_agent_name_out) {
            std::string remote_metadata;
            EXPECT_EQ(remote_agent->getLocalMD(remote_metadata), NIXL_SUCCESS);
            return agent_->loadRemoteMD(remote_metadata, remote_agent_name_out);
        }

        nixl_status_t
        InitAndRegisterMemory(Blob &blob,
                              nixl_reg_dlist_t &reg_dlist,
                              nixl_opt_args_t &extra_params,
                              nixlBackendH *backend) {
            reg_dlist.addDesc(blob.GetDesc());
            extra_params.backends.push_back(backend);
            return agent_->registerMem(reg_dlist, &extra_params);
        }
    };

    class SingleAgentSessionFixture : public testing::Test {
    protected:
        std::unique_ptr<AgentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<AgentHelper>(local_agent_name);
            agent_ = agent_helper_->GetAgent();
        }
    };

    class DualAgentBridgeFixture : public testing::Test {
    protected:
        std::unique_ptr<AgentHelper> local_agent_helper_, remote_agent_helper_;
        nixlAgent *local_agent_, *remote_agent_;

        void
        SetUp() override {
            local_agent_helper_ = std::make_unique<AgentHelper>(local_agent_name);
            remote_agent_helper_ = std::make_unique<AgentHelper>(remote_agent_name);
            local_agent_ = local_agent_helper_->GetAgent();
            remote_agent_ = remote_agent_helper_->GetAgent();
        }
    };

    class SingleAgentWithMemParamFixture : public testing::TestWithParam<nixl_mem_t> {
    protected:
        std::unique_ptr<AgentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<AgentHelper>(local_agent_name);
            agent_ = agent_helper_->GetAgent();
        }
    };

    TEST_F(SingleAgentSessionFixture, GetNonExistingPluginTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;

        EXPECT_NE(agent_->getPluginParams(nonexisting_plugin, mem, params), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, GetExistingPluginTest) {
        std::vector<nixl_backend_t> plugins;
        EXPECT_EQ(agent_->getAvailPlugins(plugins), NIXL_SUCCESS);
        if (plugins.empty()) {
            GTEST_SKIP();
        }

        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(plugins.front(), mem, params), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, CreateNonExistingPluginBackendTest) {
        nixlPluginManager &plugin_manager = nixlPluginManager::getInstance();
        EXPECT_EQ(plugin_manager.loadPlugin(nonexisting_plugin), nullptr);

        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_NE(agent_->createBackend(nonexisting_plugin, params, backend), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, CreateExistingPluginBackendTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(GetMockBackendName(), mem, params), NIXL_SUCCESS);

        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->CreateBackendWithGMock(params, backend), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, GetNonExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_NE(agent_->getBackendParams(nullptr, mem, params), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, GetExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->CreateBackendWithGMock(params, backend), NIXL_SUCCESS);
        EXPECT_EQ(agent_->getBackendParams(backend, mem, params), NIXL_SUCCESS);
    }

    TEST_F(SingleAgentSessionFixture, GetLocalMetadataTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->CreateBackendWithGMock(params, backend), NIXL_SUCCESS);

        std::string metadata;
        EXPECT_EQ(agent_->getLocalMD(metadata), NIXL_SUCCESS);
        EXPECT_FALSE(metadata.empty());
    }

    TEST_P(SingleAgentWithMemParamFixture, RegisterMemoryTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->CreateBackendWithGMock(params, backend), NIXL_SUCCESS);

        Blob blob;
        nixl_opt_args_t extra_params;
        nixl_reg_dlist_t reg_dlist(GetParam());
        EXPECT_EQ(agent_helper_->InitAndRegisterMemory(blob, reg_dlist, extra_params, backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
    }

    INSTANTIATE_TEST_SUITE_P(DramRegisterMemoryInstantiation,
                             SingleAgentWithMemParamFixture,
                             testing::Values(DRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(VramRegisterMemoryInstantiation,
                             SingleAgentWithMemParamFixture,
                             testing::Values(VRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(BlkRegisterMemoryInstantiation,
                             SingleAgentWithMemParamFixture,
                             testing::Values(BLK_SEG));
    INSTANTIATE_TEST_SUITE_P(ObjRegisterMemoryInstantiation,
                             SingleAgentWithMemParamFixture,
                             testing::Values(OBJ_SEG));
    INSTANTIATE_TEST_SUITE_P(FileRegisterMemoryInstantiation,
                             SingleAgentWithMemParamFixture,
                             testing::Values(FILE_SEG));

    TEST_F(DualAgentBridgeFixture, LoadRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_name, remote_agent_name_out);
    }

    TEST_F(DualAgentBridgeFixture, InvalidateRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->invalidateRemoteMD(remote_agent_name_out), NIXL_SUCCESS);
    }

    TEST_F(DualAgentBridgeFixture, XferReqTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->GetGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        Blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->InitAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->InitAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.GetDesc());
        remote_xfer_dlist.addDesc(remote_blob.GetDesc());

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
        EXPECT_EQ(notif_map.size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(DualAgentBridgeFixture, XferReqSubFunctionsTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->GetGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        Blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->InitAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->InitAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.GetDesc());
        remote_xfer_dlist.addDesc(remote_blob.GetDesc());

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
        EXPECT_EQ(notif_map.size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl1), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl2), NIXL_SUCCESS);
    }

    TEST_F(DualAgentBridgeFixture, GenNotifTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->GetGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->genNotif(remote_agent_name_out, msg), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);
    }

    TEST_F(DualAgentBridgeFixture, QueryXferBackendTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        Blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->InitAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->InitAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.GetDesc());
        remote_xfer_dlist.addDesc(remote_blob.GetDesc());

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

    TEST_F(DualAgentBridgeFixture, MakeConnectionTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->CreateBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->CreateBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string local_agent_name_out, remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->GetAndLoadRemoteMD(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->GetAndLoadRemoteMD(local_agent_, local_agent_name_out),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->makeConnection(remote_agent_name_out), NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_->makeConnection(local_agent_name_out), NIXL_SUCCESS);
    }

} // namespace agent
} // namespace gtest
