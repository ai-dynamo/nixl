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
#include "gmock_engine.h"

namespace mocks {

MockDramBackendEngine::~MockDramBackendEngine() {}

MockDramBackendEngine::MockDramBackendEngine(
    const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      sharedState(1) {
      if (init_params->customParams->count(gmock_engine_key) <= 0) {
        throw std::runtime_error("gmock_engine not found");
      } else {
        gmock_backend_engine = reinterpret_cast<nixlBackendEngine*>(
            std::stoul(init_params->customParams->at(gmock_engine_key)));
      }
    }

nixl_status_t MockDramBackendEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                                                nixlBackendMD *&out) {
  sharedState++;
  return gmock_backend_engine->registerMem(mem, nixl_mem, out);
}

nixl_status_t MockDramBackendEngine::deregisterMem(nixlBackendMD *meta) {
  sharedState++;
  return gmock_backend_engine->deregisterMem(meta);
}

nixl_status_t MockDramBackendEngine::connect(const std::string &remote_agent) {
  sharedState++;
  return gmock_backend_engine->connect(remote_agent);
}

nixl_status_t MockDramBackendEngine::disconnect(const std::string &remote_agent) {
  sharedState++;
  return gmock_backend_engine->disconnect(remote_agent);
}

nixl_status_t MockDramBackendEngine::unloadMD(nixlBackendMD *input) {
  sharedState++;
  return gmock_backend_engine->unloadMD(input);
}

nixl_status_t MockDramBackendEngine::prepXfer(const nixl_xfer_op_t &operation,
                                             const nixl_meta_dlist_t &local,
                                             const nixl_meta_dlist_t &remote,
                                             const std::string &remote_agent,
                                             nixlBackendReqH *&handle,
                                             const nixl_opt_b_args_t *opt_args) const {
  assert(sharedState > 0);
  return gmock_backend_engine->prepXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t MockDramBackendEngine::postXfer(const nixl_xfer_op_t &operation,
                                             const nixl_meta_dlist_t &local,
                                             const nixl_meta_dlist_t &remote,
                                             const std::string &remote_agent,
                                             nixlBackendReqH *&handle,
                                             const nixl_opt_b_args_t *opt_args) const {
  assert(sharedState > 0);
  return gmock_backend_engine->postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t MockDramBackendEngine::checkXfer(nixlBackendReqH *handle) const {
  assert(sharedState > 0);
  return gmock_backend_engine->checkXfer(handle);
}

nixl_status_t MockDramBackendEngine::releaseReqH(nixlBackendReqH *handle) const {
  assert(sharedState > 0);
  return gmock_backend_engine->releaseReqH(handle);
}

nixl_status_t MockDramBackendEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                                       const std::string &remote_conn_info) {
  sharedState++;
  return gmock_backend_engine->loadRemoteConnInfo(remote_agent, remote_conn_info);
}

nixl_status_t MockDramBackendEngine::loadRemoteMD(const nixlBlobDesc &input,
                                                 const nixl_mem_t &nixl_mem,
                                                 const std::string &remote_agent,
                                                 nixlBackendMD *&output) {
  sharedState++;
  return gmock_backend_engine->loadRemoteMD(input, nixl_mem, remote_agent, output);
}

nixl_status_t MockDramBackendEngine::loadLocalMD(nixlBackendMD *input,
                                                nixlBackendMD *&output) {
  sharedState++;
  return gmock_backend_engine->loadLocalMD(input, output);
}

nixl_status_t MockDramBackendEngine::getNotifs(notif_list_t &notif_list) {
  sharedState++;
  return gmock_backend_engine->getNotifs(notif_list);
}

nixl_status_t MockDramBackendEngine::genNotif(const std::string &remote_agent,
                                             const std::string &msg) const {
  assert(sharedState > 0);
  return gmock_backend_engine->genNotif(remote_agent, msg);
}

int MockDramBackendEngine::progress() {
  sharedState++;
  return gmock_backend_engine->progress();
}
} // namespace mocks
