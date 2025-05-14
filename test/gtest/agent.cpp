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
unsigned char GetRandomByte() {
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

  nixlBlobDesc GetDesc() const { return desc_; }
};

class AgentHelper {
protected:
  NiceMock<GMockBackendEngine> gmock_engine_;
  std::shared_ptr<nixlAgent> agent_;
  nixl_mem_list_t mem_;
  nixl_b_params_t params_;
  nixlBackendH* backend_;
  std::string metadata_;
  nixl_opt_args_t extra_params_;
  Blob blob_;
  nixl_reg_dlist_t reg_dlist_;
  nixl_xfer_dlist_t xfer_dlist_;

public:
  AgentHelper(const std::string& name, nixl_mem_t mem_type) :
    agent_(std::make_unique<nixlAgent>(name, nixlAgentConfig(true))),
    reg_dlist_(mem_type),
    xfer_dlist_(mem_type) {}

  nixl_status_t GetAvailablePlugins(std::vector<nixl_backend_t>& plugins) {
    return agent_->getAvailPlugins(plugins);
  }

  void SetBackend(nixlBackendH* backend) {
    backend_ = backend;
  }

  const nixlBackendH* GetBackend() const {
    return backend_;
  }

  nixl_status_t GetPluginParams(const nixl_backend_t& type) {
    return agent_->getPluginParams(type, mem_, params_);
  }

  nixl_status_t CreateBackend(const nixl_backend_t& type) {
    params_[gmock_engine_key] = std::to_string(reinterpret_cast<uintptr_t>(&gmock_engine_));

    nixl_status_t status = agent_->createBackend(type, params_, backend_);
    extra_params_.backends.push_back(backend_);
    return status;
  }

  nixl_status_t GetBackendParams() {
    return agent_->getBackendParams(backend_, mem_, params_);
  }

  nixl_status_t RegisterMemory() {
    reg_dlist_.addDesc(blob_.GetDesc());
    return agent_->registerMem(reg_dlist_, &extra_params_);
  }

  nixl_status_t DeregisterMemory() {
    return agent_->deregisterMem(reg_dlist_, &extra_params_);
  }

  nixl_status_t GetLocalMetadata(std::string& metadata) {
    nixl_status_t status = agent_->getLocalMD(metadata_);
    metadata = metadata_;
    return status;
  }

  nixl_status_t LoadRemoteMetadata(const std::string& remote_metadata,
                                   std::string& remote_agent_name_out) {
    return agent_->loadRemoteMD(remote_metadata, remote_agent_name_out);
  }

  nixl_status_t InvalidateRemoteMD(const std::string& remote_agent_name) {
    return agent_->invalidateRemoteMD(remote_agent_name);
  }

  nixl_xfer_dlist_t& PopulateXferDlist() {
    xfer_dlist_.addDesc(blob_.GetDesc());
    return xfer_dlist_;
  }

  nixl_status_t CreateXferReq(nixl_xfer_op_t op, nixl_xfer_dlist_t& src,
                              nixl_xfer_dlist_t& dst, const std::string& remote_agent_name,
                              nixlXferReqH*& req_hndl, const std::string& msg = "") {
    if (!msg.empty()) {
      extra_params_.notifMsg = msg;
      extra_params_.hasNotif = true;
    }
    return agent_->createXferReq(op, src, dst, remote_agent_name, req_hndl, &extra_params_);
  }

  nixl_status_t PostXferReq(nixlXferReqH* xfer_req) {
    return agent_->postXferReq(xfer_req);
  }

  nixl_status_t GetXferStatus(nixlXferReqH* xfer_req) {
    return agent_->getXferStatus(xfer_req);
  }

  nixl_status_t ReleaseXferReq(nixlXferReqH* xfer_req) {
    return agent_->releaseXferReq(xfer_req);
  }

  nixl_status_t GetNotifs(nixl_notifs_t& notif_map) {
    return agent_->getNotifs(notif_map);
  }

  nixl_status_t GenNotif(const std::string& remote_agent_name, const std::string& msg) {
    return agent_->genNotif(remote_agent_name, msg);
  }

  nixl_status_t PrepXferDlist(const std::string& remote_agent_name,
                              nixl_xfer_dlist_t& dlist, nixlDlistH*& dlist_hndl) {
    return agent_->prepXferDlist(remote_agent_name, dlist, dlist_hndl);
  }

  nixl_status_t MakeXferReq(nixl_xfer_op_t op, nixlDlistH* desc_hndl_src,
                            std::vector<int>& indices_src, nixlDlistH* desc_hndl_dst,
                            std::vector<int>& indices_dst, nixlXferReqH*& req_hndl,
                            const std::string& msg = "") {
    if (!msg.empty()) {
      extra_params_.notifMsg = msg;
      extra_params_.hasNotif = true;
    }
    return agent_->makeXferReq(op, desc_hndl_src, indices_src, desc_hndl_dst,
                               indices_dst, req_hndl, &extra_params_);
  }

  nixl_status_t ReleasedDlistH(nixlDlistH* dlist_hndl) {
    return agent_->releasedDlistH(dlist_hndl);
  }

  nixl_status_t QueryXferBackend(nixlXferReqH* xfer_req, nixlBackendH*& backend_out) {
    return agent_->queryXferBackend(xfer_req, backend_out);
  }

  nixl_status_t MakeConnection(const std::string& remote_agent_name) {
    return agent_->makeConnection(remote_agent_name);
  }

  const GMockBackendEngine& GetGMockEngine() const {
    return gmock_engine_;
  }
};

class CreateAgentFixture : public testing::Test {
protected:
  std::shared_ptr<AgentHelper> agent;

  void SetUp() override {
    agent = std::make_unique<AgentHelper>(local_agent_name, DRAM_SEG);
  }
};

class CreateTwoAgentsFixture : public testing::Test {
protected:
  std::shared_ptr<AgentHelper> l_agent, r_agent;

  void SetUp() override {
    l_agent = std::make_unique<AgentHelper>(local_agent_name, DRAM_SEG);
    r_agent = std::make_unique<AgentHelper>(remote_agent_name, DRAM_SEG);
  }
};

class CreateAgentWithParamsFixture : public testing::TestWithParam<nixl_mem_t> {
protected:
  std::shared_ptr<AgentHelper> agent;

  void SetUp() override {
    agent = std::make_unique<AgentHelper>(local_agent_name, GetParam());
  }
};

TEST_F(CreateAgentFixture, GetNonExistingPluginTest) {
  EXPECT_NE(agent->GetPluginParams(nixl_backend_t(nonexisting_plugin)), NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, GetExistingPluginTest) {
  std::vector<nixl_backend_t> plugins;

  EXPECT_EQ(agent->GetAvailablePlugins(plugins), NIXL_SUCCESS);
  if (plugins.empty()) {
    GTEST_SKIP();
  }

  EXPECT_EQ(agent->GetPluginParams(plugins.front()), NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, CreateNonExistingPluginBackendTest) {
  EXPECT_NE(agent->CreateBackend(nixl_backend_t(nonexisting_plugin)), NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, CreateExistingPluginBackendTest) {
  EXPECT_EQ(agent->GetPluginParams(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, GetNonExistingBackendParamsTest) {
  agent->SetBackend(nullptr);
  EXPECT_NE(agent->GetBackendParams(), NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, GetExistingBackendParamsTest) {
  EXPECT_EQ(agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(agent->GetBackendParams(), NIXL_SUCCESS);
}

TEST_P(CreateAgentWithParamsFixture, RegisterMemoryTest) {
  EXPECT_EQ(agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);

  EXPECT_EQ(agent->RegisterMemory(), NIXL_SUCCESS);
  EXPECT_EQ(agent->DeregisterMemory(), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(DramRegisterMemoryInstantiation,
                         CreateAgentWithParamsFixture,
                         testing::Values(DRAM_SEG));
INSTANTIATE_TEST_SUITE_P(VramRegisterMemoryInstantiation,
                         CreateAgentWithParamsFixture,
                         testing::Values(VRAM_SEG));
INSTANTIATE_TEST_SUITE_P(BlkRegisterMemoryInstantiation,
                         CreateAgentWithParamsFixture,
                         testing::Values(BLK_SEG));
INSTANTIATE_TEST_SUITE_P(ObjRegisterMemoryInstantiation,
                         CreateAgentWithParamsFixture,
                         testing::Values(OBJ_SEG));
INSTANTIATE_TEST_SUITE_P(FileRegisterMemoryInstantiation,
                         CreateAgentWithParamsFixture,
                         testing::Values(FILE_SEG));

TEST_F(CreateAgentFixture, GetLocalMetadataTest) {
  std::string metadata;
  EXPECT_EQ(agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(agent->GetLocalMetadata(metadata), NIXL_SUCCESS);
  EXPECT_FALSE(metadata.empty());
}

TEST_F(CreateTwoAgentsFixture, LoadRemoteMetadataTest) {
  std::string remote_agent_name_out;
  std::string remote_metadata;

  l_agent->CreateBackend(mock_backend_plugin_name);
  r_agent->CreateBackend(mock_backend_plugin_name);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);

  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);
  EXPECT_EQ(remote_agent_name, remote_agent_name_out);
}

TEST_F(CreateTwoAgentsFixture, InvalidateRemoteMetadataTest) {
  std::string remote_agent_name_out;
  std::string remote_metadata;

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

  EXPECT_EQ(l_agent->InvalidateRemoteMD(remote_agent_name_out), NIXL_SUCCESS);
}

TEST_F(CreateTwoAgentsFixture, XferReqTest) {
  std::string remote_agent_name_out;
  std::string msg = "notification";
  std::string remote_metadata;
  nixl_notifs_t notif_map;
  nixlXferReqH* xfer_req;

  EXPECT_CALL(r_agent->GetGMockEngine(), getNotifs).WillOnce([&](notif_list_t &notif_list) {
    notif_list.push_back(std::make_pair(local_agent_name, msg));
    return NIXL_SUCCESS;
  });

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->RegisterMemory(), NIXL_SUCCESS);

  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->RegisterMemory(), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

  nixl_xfer_dlist_t l_xfer_dlist = l_agent->PopulateXferDlist();
  nixl_xfer_dlist_t r_xfer_dlist = r_agent->PopulateXferDlist();
  EXPECT_EQ(l_agent->CreateXferReq(NIXL_WRITE, l_xfer_dlist, r_xfer_dlist,
            remote_agent_name, xfer_req, msg), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->PostXferReq(xfer_req), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->GetXferStatus(xfer_req), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetNotifs(notif_map), NIXL_SUCCESS);
  EXPECT_EQ(notif_map.size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].front(), msg);

  l_agent->ReleaseXferReq(xfer_req);
}

TEST_F(CreateTwoAgentsFixture, XferReqSubFunctionsTest) {
  nixlDlistH* desc_hndl1, *desc_hndl2;
  std::string remote_agent_name_out;
  std::string msg = "notification";
  std::string remote_metadata;
  std::vector<int> indices;
  nixl_notifs_t notif_map;
  nixlXferReqH* xfer_req;

  EXPECT_CALL(r_agent->GetGMockEngine(), getNotifs).WillOnce([&](notif_list_t &notif_list) {
    notif_list.push_back(std::make_pair(local_agent_name, msg));
    return NIXL_SUCCESS;
  });

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->RegisterMemory(), NIXL_SUCCESS);

  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->RegisterMemory(), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

  nixl_xfer_dlist_t l_xfer_dlist = l_agent->PopulateXferDlist();
  nixl_xfer_dlist_t r_xfer_dlist = r_agent->PopulateXferDlist();
  EXPECT_EQ(l_agent->PrepXferDlist(NIXL_INIT_AGENT, l_xfer_dlist, desc_hndl1), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->PrepXferDlist(remote_agent_name_out, r_xfer_dlist, desc_hndl2), NIXL_SUCCESS);

  for (int i = 0; i < l_xfer_dlist.descCount(); i++)
    indices.push_back(i);

  EXPECT_EQ(l_agent->MakeXferReq(NIXL_WRITE, desc_hndl1, indices, desc_hndl2, indices,
            xfer_req, msg), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->PostXferReq(xfer_req), NIXL_SUCCESS);

  EXPECT_EQ(l_agent->GetXferStatus(xfer_req), NIXL_SUCCESS);

  EXPECT_EQ(r_agent->GetNotifs(notif_map), NIXL_SUCCESS);
  EXPECT_EQ(notif_map.size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].front(), msg);

  EXPECT_EQ(l_agent->ReleaseXferReq(xfer_req), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->ReleasedDlistH(desc_hndl1), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->ReleasedDlistH(desc_hndl2), NIXL_SUCCESS);
}

TEST_F(CreateTwoAgentsFixture, GenNotifTest) {
  std::string remote_agent_name_out;
  std::string msg = "notification";
  std::string remote_metadata;
  nixl_notifs_t notif_map;

  EXPECT_CALL(r_agent->GetGMockEngine(), getNotifs).WillOnce([&](notif_list_t &notif_list) {
    notif_list.push_back(std::make_pair(local_agent_name, msg));
    return NIXL_SUCCESS;
  });

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

  EXPECT_EQ(l_agent->GenNotif(remote_agent_name_out, msg), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetNotifs(notif_map), NIXL_SUCCESS);

  EXPECT_EQ(notif_map.size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].size(), 1);
  EXPECT_EQ(notif_map[local_agent_name].front(), msg);
}

TEST_F(CreateTwoAgentsFixture, QueryXferBackendTest) {
  std::string remote_agent_name_out;
  std::string remote_metadata;
  nixlBackendH* backend_out;
  nixlXferReqH* xfer_req;

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->RegisterMemory(), NIXL_SUCCESS);

  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->RegisterMemory(), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(remote_metadata), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

  nixl_xfer_dlist_t l_xfer_dlist = l_agent->PopulateXferDlist();
  nixl_xfer_dlist_t r_xfer_dlist = r_agent->PopulateXferDlist();
  EXPECT_EQ(l_agent->CreateXferReq(NIXL_WRITE, l_xfer_dlist, r_xfer_dlist,
            remote_agent_name_out, xfer_req), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->QueryXferBackend(xfer_req, backend_out), NIXL_SUCCESS);
  EXPECT_EQ(backend_out, l_agent->GetBackend());

  EXPECT_EQ(l_agent->ReleaseXferReq(xfer_req), NIXL_SUCCESS);
}

TEST_F(CreateTwoAgentsFixture, MakeConnectionTest) {
  std::string remote_agent_name_out;
  std::string local_agent_name_out;
  std::string metadata1, metadata2;

  EXPECT_EQ(l_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->CreateBackend(mock_backend_plugin_name), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->GetLocalMetadata(metadata2), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->GetLocalMetadata(metadata1), NIXL_SUCCESS);
  EXPECT_EQ(l_agent->LoadRemoteMetadata(metadata2, remote_agent_name_out), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->LoadRemoteMetadata(metadata1, local_agent_name_out), NIXL_SUCCESS);

  EXPECT_EQ(l_agent->MakeConnection(remote_agent_name_out), NIXL_SUCCESS);
  EXPECT_EQ(r_agent->MakeConnection(local_agent_name_out), NIXL_SUCCESS);
}

} // namespace agent
} // namespace gtest
