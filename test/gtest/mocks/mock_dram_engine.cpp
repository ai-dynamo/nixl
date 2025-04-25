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
#include "mock_dram_engine.h"

namespace mocks {

MockDramBackendEngine::~MockDramBackendEngine() {}

nixl_status_t MockDramBackendEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                                                nixlBackendMD *&out) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::deregisterMem(nixlBackendMD *meta) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::connect(const std::string &remote_agent) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::disconnect(const std::string &remote_agent) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::unloadMD(nixlBackendMD *input) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::prepXfer(const nixl_xfer_op_t &operation,
                                             const nixl_meta_dlist_t &local,
                                             const nixl_meta_dlist_t &remote,
                                             const std::string &remote_agent,
                                             nixlBackendReqH *&handle,
                                             const nixl_opt_b_args_t *opt_args) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::postXfer(const nixl_xfer_op_t &operation,
                                             const nixl_meta_dlist_t &local,
                                             const nixl_meta_dlist_t &remote,
                                             const std::string &remote_agent,
                                             nixlBackendReqH *&handle,
                                             const nixl_opt_b_args_t *opt_args) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::checkXfer(nixlBackendReqH *handle) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::releaseReqH(nixlBackendReqH *handle) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                                       const std::string &remote_conn_info) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::loadRemoteMD(const nixlBlobDesc &input,
                                                 const nixl_mem_t &nixl_mem,
                                                 const std::string &remote_agent,
                                                 nixlBackendMD *&output) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::loadLocalMD(nixlBackendMD *input,
                                                nixlBackendMD *&output) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::getNotifs(notif_list_t &notif_list) {
  sharedState++;
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::genNotif(const std::string &remote_agent,
                                             const std::string &msg) {
  sharedState++;
  return NIXL_SUCCESS;
}

int MockDramBackendEngine::progress() {
  sharedState++;
  return 0;
}
} // namespace mocks
