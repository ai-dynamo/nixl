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

#include "nixl.h"

using namespace testing;

nixl_b_params_t custom_params;
const nixlBackendInitParams init_params{.customParams = &custom_params};
const std::string gmock_engine_key = "gmock_engine_key";

class GMockBackendEngine : public nixlBackendEngine {
public:
  GMockBackendEngine() : nixlBackendEngine(&init_params) {
    ON_CALL(*this, supportsRemote()).WillByDefault(Return(true));
    ON_CALL(*this, supportsLocal()).WillByDefault(Return(true));
    ON_CALL(*this, supportsNotif()).WillByDefault(Return(true));
    ON_CALL(*this, supportsProgTh()).WillByDefault(Return(false));
    ON_CALL(*this, getSupportedMems()).WillByDefault(Return(nixl_mem_list_t{DRAM_SEG}));
    ON_CALL(*this, registerMem(_, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, deregisterMem(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, connect(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, disconnect(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, unloadMD(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, prepXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, postXfer(_, _, _, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, checkXfer(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, releaseReqH(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, getPublicData(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, getConnInfo(_)).WillByDefault([&](std::string &str) {
      str = "mock_dram_plugin_conn_info";
      return NIXL_SUCCESS;
    });
    ON_CALL(*this, loadRemoteConnInfo(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, loadRemoteMD(_, _, _, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, loadLocalMD(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, getNotifs(_)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, genNotif(_, _)).WillByDefault(Return(NIXL_SUCCESS));
    ON_CALL(*this, progress()).WillByDefault(Return(0));
  }

  MOCK_METHOD(bool, supportsRemote, (), (const, override));
  MOCK_METHOD(bool, supportsLocal, (), (const, override));
  MOCK_METHOD(bool, supportsNotif, (), (const, override));
  MOCK_METHOD(bool, supportsProgTh, (), (const, override));
  MOCK_METHOD(nixl_mem_list_t, getSupportedMems, (), (const, override));
  MOCK_METHOD(nixl_status_t, registerMem, (const nixlBlobDesc& desc,
              const nixl_mem_t& mem, nixlBackendMD*& out), (override));
  MOCK_METHOD(nixl_status_t, deregisterMem, (nixlBackendMD* meta), (override));
  MOCK_METHOD(nixl_status_t, connect, (const std::string& remote_agent), (override));
  MOCK_METHOD(nixl_status_t, disconnect, (const std::string& remote_agent), (override));
  MOCK_METHOD(nixl_status_t, unloadMD, (nixlBackendMD* input), (override));
  MOCK_METHOD(nixl_status_t, prepXfer, (const nixl_xfer_op_t& op,
              const nixl_meta_dlist_t& src, const nixl_meta_dlist_t& dst,
              const std::string& remote_agent, nixlBackendReqH*& req,
              const nixl_opt_b_args_t* extra_args), (const, override));
  MOCK_METHOD(nixl_status_t, postXfer, (const nixl_xfer_op_t& op,
              const nixl_meta_dlist_t& src, const nixl_meta_dlist_t& dst,
              const std::string& remote_agent, nixlBackendReqH*& req,
              const nixl_opt_b_args_t* extra_args), (const, override));
  MOCK_METHOD(nixl_status_t, checkXfer, (nixlBackendReqH* req), (const, override));
  MOCK_METHOD(nixl_status_t, releaseReqH, (nixlBackendReqH* req), (const, override));
  MOCK_METHOD(nixl_status_t, getPublicData, (const nixlBackendMD* input,
              std::string& str), (const, override));
  MOCK_METHOD(nixl_status_t, getConnInfo, (std::string& str), (const, override));
  MOCK_METHOD(nixl_status_t, loadRemoteConnInfo, (const std::string& remote_agent,
              const std::string& remote_conn_info), (override));
  MOCK_METHOD(nixl_status_t, loadRemoteMD, (const nixlBlobDesc& input,
              const nixl_mem_t& nixl_mem, const std::string& remote_agent,
              nixlBackendMD*& output), (override));
  MOCK_METHOD(nixl_status_t, loadLocalMD, (nixlBackendMD* input,
              nixlBackendMD*& output), (override));
  MOCK_METHOD(nixl_status_t, getNotifs, (notif_list_t& notif_list), (override));
  MOCK_METHOD(nixl_status_t, genNotif, (const std::string& remote_agent,
              const std::string& msg), (const, override));
  MOCK_METHOD(int, progress, (), (override));
};
