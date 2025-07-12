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

#include "plugins_common.h"
#include "obj/obj_backend.h"

namespace gtest {
namespace obj_plugin {

nixl_b_params_t obj_params{
    {"region", "eu-central-1"},
    {"bucket", "nixl-ci-test"},
};

const nixlBackendInitParams obj_test_params = {
    .localAgent = "Agent1",
    .type = "OBJ",
    .customParams = &obj_params,
    .enableProgTh = false,
    .pthrDelay = 0,
    .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW
};

class SetupObjTestFixture : public plugins_common::SetupBackendTestFixture {
protected:
    SetupObjTestFixture() {
        backend_engine_ = std::make_unique<nixlObjEngine>(&GetParam());
    }
};

TEST_P(SetupObjTestFixture, SimpleLifeCycleTest) {
    EXPECT_TRUE(IsLoaded());
}

TEST_P(SetupObjTestFixture, XferTest) {
    EXPECT_TRUE(IsLoaded());
    EXPECT_TRUE((SetupLocalXfer<DRAM_SEG, OBJ_SEG>()));
    EXPECT_TRUE(TestLocalXfer(NIXL_WRITE));
    ResetLocalBuf();
    EXPECT_TRUE(TestLocalXfer(NIXL_READ));
    EXPECT_TRUE(CheckLocalBuf());
    EXPECT_TRUE(TeardownXfer());
}

INSTANTIATE_TEST_SUITE_P(ObjTests, SetupObjTestFixture, 
                        testing::Values(obj_test_params));

} // namespace obj_plugin
} // namespace gtest 