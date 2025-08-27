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
#ifndef TEST_GTEST_GMOCK_ENGINE_H
#define TEST_GTEST_GMOCK_ENGINE_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "backend/backend_engine.h"

namespace mocks {

/**
 * @class GMockBackendEngine
 * @brief A GMock implementation of nixlBackendEngine for GTest testing purposes.
 *
 * This class provides a Google Mock (GMock) implementation of the nixlBackendEngine
 * interface, enabling flexible and test-specific behavior.
 * Unlike the standalone mock plugin (MockBackendEngine), which is loaded as an external
 * executable and cannot be customized per test - this GMock-based approach allows
 * defining mock behavior directly in the test. These behaviors are passed to the
 * backend during creation, and the mock engine delegates calls to the GMock
 * implementation accordingly.
 *
 * Usage:
 * 1. Create an instance (use NiceMock to suppress warnings about uninteresting calls
 *    that occur when invoking methods with only default, but no explicit, implementations):
 *    NiceMock<mocks::GMockBackendEngine> gmock_engine;
 *
 * 2. Set up expectations for method calls:
 *    EXPECT_CALL(gmock_engine, someMethod())...
 *
 * 3. Pass it to the backend via the custom input parameters:
 *    gmock_engine.SetToParams(params);
 *
 * Note: If no explicit expectation is set for a method, the default behavior defined
 * with ON_CALL(...).WillByDefault() will be used. These defaults are designed to provide
 * reasonable behavior for testing, such as returning NIXL_SUCCESS for most operations.
 *
 */
class GMockBackendEngine : public nixlBackendEngine {
public:
    GMockBackendEngine();

    GMockBackendEngine(const nixlBackendInitParams *init_params) : nixlBackendEngine(init_params) {}


    void
    SetToParams(nixl_b_params_t &params) const;
    static GMockBackendEngine *
    GetFromParams(nixl_b_params_t *params);

    MOCK_METHOD(bool, supportsRemote, (), (const, override));
    MOCK_METHOD(bool, supportsLocal, (), (const, override));
    MOCK_METHOD(bool, supportsNotif, (), (const, override));
    MOCK_METHOD(nixl_mem_list_t, getSupportedMems, (), (const, override));
    MOCK_METHOD(nixl_status_t,
                registerMem,
                (const nixlBlobDesc &desc, const nixl_mem_t &mem, nixlBackendMD *&out),
                (override));
    MOCK_METHOD(nixl_status_t, deregisterMem, (nixlBackendMD * meta), (override));
    MOCK_METHOD(nixl_status_t, connect, (const std::string &remote_agent), (override));
    MOCK_METHOD(nixl_status_t, disconnect, (const std::string &remote_agent), (override));
    MOCK_METHOD(nixl_status_t, unloadMD, (nixlBackendMD * input), (override));
    MOCK_METHOD(nixl_status_t,
                prepXfer,
                (const nixl_xfer_op_t &op,
                 const nixl_meta_dlist_t &src,
                 const nixl_meta_dlist_t &dst,
                 const std::string &remote_agent,
                 nixlBackendReqH *&req,
                 const nixl_opt_b_args_t *extra_args),
                (const, override));
    MOCK_METHOD(nixl_status_t,
                postXfer,
                (const nixl_xfer_op_t &op,
                 const nixl_meta_dlist_t &src,
                 const nixl_meta_dlist_t &dst,
                 const std::string &remote_agent,
                 nixlBackendReqH *&req,
                 const nixl_opt_b_args_t *extra_args),
                (const, override));
    MOCK_METHOD(nixl_status_t, checkXfer, (nixlBackendReqH * req), (const, override));
    MOCK_METHOD(nixl_status_t, releaseReqH, (nixlBackendReqH * req), (const, override));
    MOCK_METHOD(nixl_status_t,
                getPublicData,
                (const nixlBackendMD *input, std::string &str),
                (const, override));
    MOCK_METHOD(nixl_status_t, getConnInfo, (std::string & str), (const, override));
    MOCK_METHOD(nixl_status_t,
                loadRemoteConnInfo,
                (const std::string &remote_agent, const std::string &remote_conn_info),
                (override));
    MOCK_METHOD(nixl_status_t,
                loadRemoteMD,
                (const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output),
                (override));
    MOCK_METHOD(nixl_status_t,
                loadLocalMD,
                (nixlBackendMD * input, nixlBackendMD *&output),
                (override));
    MOCK_METHOD(nixl_status_t, getNotifs, (notif_list_t & notif_list), (override));
    MOCK_METHOD(nixl_status_t,
                genNotif,
                (const std::string &remote_agent, const std::string &msg),
                (const, override));
};

} // namespace mocks

#endif // TEST_GTEST_GMOCK_ENGINE_H
