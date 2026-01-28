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
#include <vector>

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"
#include "mocks/gmock_engine.h"

namespace gtest {
namespace agent {

class getXferStatusListTest : public ::testing::Test {
protected:
    static constexpr const char *local_agent_name = "LocalAgent";
    static constexpr const char *remote_agent_name = "RemoteAgent";
    static constexpr size_t bufLen = 256;

    std::unique_ptr<nixlAgent> local_agent_;
    std::unique_ptr<nixlAgent> remote_agent_;
    testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;

    void SetUp() override {
        local_agent_ = std::make_unique<nixlAgent>(local_agent_name, nixlAgentConfig(true));
        remote_agent_ = std::make_unique<nixlAgent>(remote_agent_name, nixlAgentConfig(true));
    }

    void TearDown() override {
        local_agent_.reset();
        remote_agent_.reset();
    }
};

// Test that getXferStatus with entry_status returns NOT_SUPPORTED for backends without support
TEST_F(getXferStatusListTest, ReturnsNotSupportedForDefaultBackend) {
    // Create backend with mock
    nixl_b_params_t params;
    gmock_engine_.SetToParams(params);
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(local_agent_->createBackend(GetMockBackendName(), params, backend), NIXL_SUCCESS);
    ASSERT_NE(backend, nullptr);

    // Exchange metadata
    std::string remote_metadata;
    ASSERT_EQ(remote_agent_->getLocalMD(remote_metadata), NIXL_SUCCESS);
    std::string remote_agent_name_out;
    ASSERT_EQ(local_agent_->loadRemoteMD(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

    // Register memory
    std::vector<char> local_buf(bufLen);
    std::vector<char> remote_buf(bufLen);
    nixlBlobDesc local_desc(reinterpret_cast<uintptr_t>(local_buf.data()), bufLen, 0);
    nixlBlobDesc remote_desc(reinterpret_cast<uintptr_t>(remote_buf.data()), bufLen, 0);

    ASSERT_EQ(local_agent_->registerMem({{local_desc, NIXL_CPU}}), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->registerMem({{remote_desc, NIXL_CPU}}), NIXL_SUCCESS);

    // Create transfer request
    nixlXferReqH *req_hndl = nullptr;
    ASSERT_EQ(local_agent_->createXferReq(
        NIXL_WRITE,
        {{local_desc, NIXL_CPU}},
        {{remote_desc, NIXL_CPU}},
        remote_agent_name,
        req_hndl), NIXL_SUCCESS);
    ASSERT_NE(req_hndl, nullptr);

    // Post transfer
    ASSERT_EQ(local_agent_->postXferReq(req_hndl), NIXL_SUCCESS);

    // Try to get per-entry status - should return NOT_SUPPORTED
    std::vector<nixl_status_t> entry_status;
    nixl_status_t status = local_agent_->getXferStatus(req_hndl, entry_status);

    EXPECT_EQ(status, NIXL_ERR_NOT_SUPPORTED);
    EXPECT_TRUE(entry_status.empty());  // Should not be populated

    // Cleanup
    ASSERT_EQ(local_agent_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    ASSERT_EQ(local_agent_->deregisterMem({{local_desc, NIXL_CPU}}), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->deregisterMem({{remote_desc, NIXL_CPU}}), NIXL_SUCCESS);
}

// Test that standard getXferStatus still works correctly
TEST_F(getXferStatusListTest, StandardGetXferStatusStillWorks) {
    // Create backend with mock
    nixl_b_params_t params;
    gmock_engine_.SetToParams(params);
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(local_agent_->createBackend(GetMockBackendName(), params, backend), NIXL_SUCCESS);
    ASSERT_NE(backend, nullptr);

    // Exchange metadata
    std::string remote_metadata;
    ASSERT_EQ(remote_agent_->getLocalMD(remote_metadata), NIXL_SUCCESS);
    std::string remote_agent_name_out;
    ASSERT_EQ(local_agent_->loadRemoteMD(remote_metadata, remote_agent_name_out), NIXL_SUCCESS);

    // Register memory
    std::vector<char> local_buf(bufLen);
    std::vector<char> remote_buf(bufLen);
    nixlBlobDesc local_desc(reinterpret_cast<uintptr_t>(local_buf.data()), bufLen, 0);
    nixlBlobDesc remote_desc(reinterpret_cast<uintptr_t>(remote_buf.data()), bufLen, 0);

    ASSERT_EQ(local_agent_->registerMem({{local_desc, NIXL_CPU}}), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->registerMem({{remote_desc, NIXL_CPU}}), NIXL_SUCCESS);

    // Create transfer request
    nixlXferReqH *req_hndl = nullptr;
    ASSERT_EQ(local_agent_->createXferReq(
        NIXL_WRITE,
        {{local_desc, NIXL_CPU}},
        {{remote_desc, NIXL_CPU}},
        remote_agent_name,
        req_hndl), NIXL_SUCCESS);
    ASSERT_NE(req_hndl, nullptr);

    // Post transfer
    ASSERT_EQ(local_agent_->postXferReq(req_hndl), NIXL_SUCCESS);

    // Standard getXferStatus should work as before
    nixl_status_t status = local_agent_->getXferStatus(req_hndl);
    EXPECT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);

    // Cleanup
    ASSERT_EQ(local_agent_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    ASSERT_EQ(local_agent_->deregisterMem({{local_desc, NIXL_CPU}}), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->deregisterMem({{remote_desc, NIXL_CPU}}), NIXL_SUCCESS);
}

}  // namespace agent
}  // namespace gtest
