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
#ifndef TEST_GTEST_MOCKS_MOCK_BACKEND_ENGINE_H
#define TEST_GTEST_MOCKS_MOCK_BACKEND_ENGINE_H

#include "backend/backend_engine.h"
#include "backend/backend_plugin.h"
#include <cassert>

namespace mocks {

class MockBackendEngine : public nixlBackendEngine {
private:
    nixlBackendEngine *gmock_backend_engine;

public:
    MockBackendEngine(const nixlBackendInitParams *init_params);
    ~MockBackendEngine() = default;

    bool
    supportsRemote() const override {
        assert(sharedState > 0);
        return gmock_backend_engine->supportsRemote();
    }
  bool supportsLocal() const override {
    assert(sharedState > 0);
    return gmock_backend_engine->supportsLocal();
  }
  bool supportsNotif() const override {
    assert(sharedState > 0);
    return gmock_backend_engine->supportsNotif();
  }
  nixl_mem_list_t getSupportedMems() const override {
    assert(sharedState > 0);
    return gmock_backend_engine->getSupportedMems();
  }
  nixl_status_t registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) override;
  nixl_status_t deregisterMem(nixlBackendMD *meta) override;
  nixl_status_t connect(const std::string &remote_agent) override;
  nixl_status_t disconnect(const std::string &remote_agent) override;
  nixl_status_t unloadMD(nixlBackendMD *input) override;
  nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const override;
  nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const override;
  nixl_status_t checkXfer(nixlBackendReqH *handle) const override;
  nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;
  nixl_status_t getPublicData(const nixlBackendMD *meta, std::string &str) const override {
    assert(sharedState > 0);
    return gmock_backend_engine->getPublicData(meta, str);
  }
  nixl_status_t getConnInfo(std::string &str) const override {
    assert(sharedState > 0);
    return gmock_backend_engine->getConnInfo(str);
  }
  nixl_status_t loadRemoteConnInfo(const std::string &remote_agent,
                                   const std::string &remote_conn_info);
  nixl_status_t loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) override;
  nixl_status_t loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output);
  nixl_status_t getNotifs(notif_list_t &notif_list) override;
  nixl_status_t genNotif(const std::string &remote_agent,
                         const std::string &msg) const override;

private:
  // This represents an engine shared state that is read in every const method and modified in non-cost ones
  // The purpose is to trigger thread sanitizer in multi-threading tests
  int sharedState;
};
} // namespace mocks

#endif
