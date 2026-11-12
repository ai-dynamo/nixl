/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common.h"
#include "mocks/gmock_engine.h"
#include "nixl.h"

namespace gtest {

class LocalConnectionTest : public testing::Test {
protected:
    [[nodiscard]] nixl_status_t
    createBackend(const std::string &allow_local_connection) {
        nixl_b_params_t params;
        engine_.SetToParams(params);
        params["allow_local_connection"] = allow_local_connection;
        return agent_.createBackend(GetMockBackendName(), params, backend_);
    }

    const std::string name_ = "agent";
    testing::NiceMock<mocks::GMockBackendEngine> engine_;
    nixlAgent agent_{name_, nixlAgentConfig()};
    nixlBackendH *backend_ = nullptr;
};

TEST_F(LocalConnectionTest, NotAllowed) {
    EXPECT_CALL(engine_, connect(testing::_)).Times(0);
    EXPECT_CALL(engine_, loadLocalMD(testing::_, testing::_)).Times(0);
    // The memory is still published for the other agents
    EXPECT_CALL(engine_, getPublicData(testing::_, testing::_)).Times(1);
    ASSERT_EQ(createBackend("false"), NIXL_SUCCESS);

    char buffer[64];
    const nixlBasicDesc desc(reinterpret_cast<uintptr_t>(buffer), sizeof(buffer), 0);
    nixl_reg_dlist_t reg_dlist(DRAM_SEG);
    reg_dlist.addDesc(nixlBlobDesc(desc, ""));
    const nixl_xfer_dlist_t xfer_dlist = reg_dlist.trim();
    ASSERT_EQ(agent_.registerMem(reg_dlist), NIXL_SUCCESS);

    // The backend does not serve its own agent
    {
        const LogIgnoreGuard lig("metadata for remote agent 'agent' not found");
        nixlXferReqH *req = nullptr;
        EXPECT_EQ(agent_.createXferReq(NIXL_WRITE, xfer_dlist, xfer_dlist, name_, req),
                  NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }
    {
        const LogIgnoreGuard lig("no specified or potential backend can send intra-agent");
        EXPECT_EQ(agent_.genNotif(name_, "notification"), NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(lig.getIgnoredCount(), 1);
    }

    EXPECT_EQ(agent_.deregisterMem(reg_dlist), NIXL_SUCCESS);
}

TEST_F(LocalConnectionTest, InvalidValue) {
    const LogIgnoreGuard lig("Conversion to bool failed");
    EXPECT_EQ(createBackend("maybe"), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

TEST_F(LocalConnectionTest, LocalOnlyBackend) {
    ON_CALL(engine_, supportsRemote()).WillByDefault(testing::Return(false));
    const LogIgnoreGuard lig("serves only its own agent");
    EXPECT_EQ(createBackend("false"), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(lig.getIgnoredCount(), 1);
}

} // namespace gtest
