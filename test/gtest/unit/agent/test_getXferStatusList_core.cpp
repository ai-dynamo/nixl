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
#include "nixl_types.h"
#include "mocks/gmock_engine.h"

namespace gtest {
namespace agent {

static constexpr const char *local_agent_name = "LocalAgent";
static constexpr const char *remote_agent_name = "RemoteAgent";
static constexpr size_t bufLen = 256;

class agentHelperForXferStatus {
protected:
    testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;
    std::unique_ptr<nixlAgent> agent_;

public:
    agentHelperForXferStatus(const std::string &name)
        : agent_(std::make_unique<nixlAgent>(name, nixlAgentConfig(true))) {}

    ~agentHelperForXferStatus() {
        agent_.reset();
    }

    nixlAgent *getAgent() const {
        return agent_.get();
    }

    const mocks::GMockBackendEngine &getGMockEngine() const {
        return gmock_engine_;
    }

    nixl_status_t createBackendWithGMock(nixl_b_params_t &params, nixlBackendH *&backend) {
        gmock_engine_.SetToParams(params);
        return agent_->createBackend(GetMockBackendName(), params, backend);
    }

    nixl_status_t getAndLoadRemoteMd(nixlAgent *remote_agent, std::string &remote_agent_name_out) {
        std::string remote_metadata;
        nixl_status_t status = remote_agent->getLocalMD(remote_metadata);
        if (status != NIXL_SUCCESS) return status;
        return agent_->loadRemoteMD(remote_metadata, remote_agent_name_out);
    }
};

class getXferStatusListTest : public ::testing::Test {
protected:
    std::unique_ptr<agentHelperForXferStatus> local_agent_helper_;
    std::unique_ptr<agentHelperForXferStatus> remote_agent_helper_;
    nixlAgent *local_agent_;
    nixlAgent *remote_agent_;

    void SetUp() override {
        local_agent_helper_ = std::make_unique<agentHelperForXferStatus>(local_agent_name);
        remote_agent_helper_ = std::make_unique<agentHelperForXferStatus>(remote_agent_name);
        local_agent_ = local_agent_helper_->getAgent();
        remote_agent_ = remote_agent_helper_->getAgent();
    }

    void TearDown() override {
        local_agent_helper_.reset();
        remote_agent_helper_.reset();
    }
};

// Test that getXferStatus with entry_status returns NOT_SUPPORTED for backends without support
TEST_F(getXferStatusListTest, ReturnsNotSupportedForDefaultBackend) {
    // Create backends on BOTH agents
    nixl_b_params_t local_params, remote_params;
    nixlBackendH *local_backend, *remote_backend;
    ASSERT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend), NIXL_SUCCESS);

    // Register memory on both agents
    std::vector<char> local_buf(bufLen);
    std::vector<char> remote_buf(bufLen);
    nixlBlobDesc local_desc(reinterpret_cast<uintptr_t>(local_buf.data()), bufLen, 0);
    nixlBlobDesc remote_desc(reinterpret_cast<uintptr_t>(remote_buf.data()), bufLen, 0);

    nixl_reg_dlist_t local_reg_dlist(DRAM_SEG);
    local_reg_dlist.addDesc(local_desc);
    nixl_reg_dlist_t remote_reg_dlist(DRAM_SEG);
    remote_reg_dlist.addDesc(remote_desc);

    nixl_opt_args_t local_extra_params, remote_extra_params;
    local_extra_params.backends.push_back(local_backend);
    remote_extra_params.backends.push_back(remote_backend);

    ASSERT_EQ(local_agent_->registerMem(local_reg_dlist, &local_extra_params), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->registerMem(remote_reg_dlist, &remote_extra_params), NIXL_SUCCESS);

    // Exchange metadata
    std::string remote_agent_name_out;
    ASSERT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out), NIXL_SUCCESS);

    // Create transfer request
    nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG);
    local_xfer_dlist.addDesc(local_desc);
    nixl_xfer_dlist_t remote_xfer_dlist(DRAM_SEG);
    remote_xfer_dlist.addDesc(remote_desc);

    nixlXferReqH *req_hndl = nullptr;
    ASSERT_EQ(local_agent_->createXferReq(
        NIXL_WRITE,
        local_xfer_dlist,
        remote_xfer_dlist,
        remote_agent_name_out,
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
    ASSERT_EQ(local_agent_->deregisterMem(local_reg_dlist, &local_extra_params), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->deregisterMem(remote_reg_dlist, &remote_extra_params), NIXL_SUCCESS);
}

// Test that standard getXferStatus still works correctly
TEST_F(getXferStatusListTest, StandardGetXferStatusStillWorks) {
    // Create backends on BOTH agents
    nixl_b_params_t local_params, remote_params;
    nixlBackendH *local_backend, *remote_backend;
    ASSERT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend), NIXL_SUCCESS);

    // Register memory on both agents
    std::vector<char> local_buf(bufLen);
    std::vector<char> remote_buf(bufLen);
    nixlBlobDesc local_desc(reinterpret_cast<uintptr_t>(local_buf.data()), bufLen, 0);
    nixlBlobDesc remote_desc(reinterpret_cast<uintptr_t>(remote_buf.data()), bufLen, 0);

    nixl_reg_dlist_t local_reg_dlist(DRAM_SEG);
    local_reg_dlist.addDesc(local_desc);
    nixl_reg_dlist_t remote_reg_dlist(DRAM_SEG);
    remote_reg_dlist.addDesc(remote_desc);

    nixl_opt_args_t local_extra_params, remote_extra_params;
    local_extra_params.backends.push_back(local_backend);
    remote_extra_params.backends.push_back(remote_backend);

    ASSERT_EQ(local_agent_->registerMem(local_reg_dlist, &local_extra_params), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->registerMem(remote_reg_dlist, &remote_extra_params), NIXL_SUCCESS);

    // Exchange metadata
    std::string remote_agent_name_out;
    ASSERT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out), NIXL_SUCCESS);

    // Create transfer request
    nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG);
    local_xfer_dlist.addDesc(local_desc);
    nixl_xfer_dlist_t remote_xfer_dlist(DRAM_SEG);
    remote_xfer_dlist.addDesc(remote_desc);

    nixlXferReqH *req_hndl = nullptr;
    ASSERT_EQ(local_agent_->createXferReq(
        NIXL_WRITE,
        local_xfer_dlist,
        remote_xfer_dlist,
        remote_agent_name_out,
        req_hndl), NIXL_SUCCESS);
    ASSERT_NE(req_hndl, nullptr);

    // Post transfer
    ASSERT_EQ(local_agent_->postXferReq(req_hndl), NIXL_SUCCESS);

    // Standard getXferStatus should work as before
    nixl_status_t status = local_agent_->getXferStatus(req_hndl);
    EXPECT_TRUE(status == NIXL_SUCCESS || status == NIXL_IN_PROG);

    // Cleanup
    ASSERT_EQ(local_agent_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    ASSERT_EQ(local_agent_->deregisterMem(local_reg_dlist, &local_extra_params), NIXL_SUCCESS);
    ASSERT_EQ(remote_agent_->deregisterMem(remote_reg_dlist, &remote_extra_params), NIXL_SUCCESS);
}

}  // namespace agent
}  // namespace gtest
