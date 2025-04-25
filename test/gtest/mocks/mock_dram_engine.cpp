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
MockDramBackendEngine::MockDramBackendEngine(
    const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {}

MockDramBackendEngine::~MockDramBackendEngine() {}

bool MockDramBackendEngine::supportsRemote() const { return true; }
bool MockDramBackendEngine::supportsLocal() const { return true; }
bool MockDramBackendEngine::supportsNotif() const { return false; }
bool MockDramBackendEngine::supportsProgTh() const { return false; };

nixl_mem_list_t MockDramBackendEngine::getSupportedMems() const {
  return nixl_mem_list_t{DRAM_SEG};
}

nixl_status_t MockDramBackendEngine::registerMem(const nixlBlobDesc &,
                                                 const nixl_mem_t &,
                                                 nixlBackendMD *&) {
  return NIXL_SUCCESS;
}

nixl_status_t MockDramBackendEngine::deregisterMem(nixlBackendMD *) {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::connect(const std::string &) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::disconnect(const std::string &) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::unloadMD(nixlBackendMD *) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::prepXfer(const nixl_xfer_op_t &,
                                              const nixl_meta_dlist_t &,
                                              const nixl_meta_dlist_t &,
                                              const std::string &,
                                              nixlBackendReqH *&,
                                              const nixl_opt_b_args_t *) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::postXfer(const nixl_xfer_op_t &,
                                              const nixl_meta_dlist_t &,
                                              const nixl_meta_dlist_t &,
                                              const std::string &,
                                              nixlBackendReqH *&,
                                              const nixl_opt_b_args_t *) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::checkXfer(nixlBackendReqH *) {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::releaseReqH(nixlBackendReqH *) {
  return NIXL_SUCCESS;
};

nixl_status_t MockDramBackendEngine::getPublicData(const nixlBackendMD *,
                                                   std::string &) const {
  return NIXL_SUCCESS;
};
nixl_status_t MockDramBackendEngine::getConnInfo(std::string &) const {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::loadRemoteConnInfo(const std::string &,
                                                        const std::string &) {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::loadRemoteMD(const nixlBlobDesc &,
                                                  const nixl_mem_t &,
                                                  const std::string &,
                                                  nixlBackendMD *&) {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::loadLocalMD(nixlBackendMD *,
                                                 nixlBackendMD *&) {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::getNotifs(notif_list_t &) {
  return NIXL_SUCCESS;
}
nixl_status_t MockDramBackendEngine::genNotif(const std::string &,
                                              const std::string &) {
  return NIXL_SUCCESS;
}
int MockDramBackendEngine::progress() { return 0; }
} // namespace mocks
