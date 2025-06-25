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

#include "plugin_test.h"
#include "ucx_backend.h"

namespace gtest {
namespace ucx_plugin {

    nixl_b_params_t ucx_custom_params{};

    const nixlBackendInitParams ucx_params = {.localAgent = "Agent1",
                                              .type = "UCX",
                                              .customParams = &ucx_custom_params,
                                              .enableProgTh = true,
                                              .pthrDelay = 100};

    class SetupUcxTestFixture : public plugin_test::SetupBackendTestFixture {
    protected:
        SetupUcxTestFixture() {
            backend_engine_ = std::make_unique<nixlUcxEngine>(&GetParam());
            remote_backend_engine_ = std::make_unique<nixlUcxEngine>(&GetParam());
        }
    };

    TEST_P(SetupUcxTestFixture, SimpleLifeCycleTest) {
        EXPECT_TRUE(IsLoaded());
    }

    TEST_P(SetupUcxTestFixture, XferTest) {
        EXPECT_TRUE(IsLoaded());
        EXPECT_TRUE(TestLocalXfer(DRAM_SEG, NIXL_WRITE));
        EXPECT_TRUE(TestLocalXfer(DRAM_SEG, NIXL_READ));
        EXPECT_TRUE(TestRemoteXfer(DRAM_SEG, NIXL_WRITE));
        EXPECT_TRUE(TestRemoteXfer(DRAM_SEG, NIXL_READ));
    }

    TEST_P(SetupUcxTestFixture, NotifTest) {
        EXPECT_TRUE(IsLoaded());
        EXPECT_TRUE(SetupRemoteXfer(DRAM_SEG));
        EXPECT_TRUE(TestGenNotif("Test"));
        EXPECT_TRUE(TeardownXfer(DRAM_SEG));
    }

    INSTANTIATE_TEST_SUITE_P(UcxBackendTest, SetupUcxTestFixture, testing::Values(ucx_params));

} // namespace ucx_plugin
} // namespace gtest
