/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

#include <cstdlib>

#include "nixl.h"

namespace gtest::rados {

namespace {

    constexpr const char *localAgent = "rados_discovery_test";

} // namespace

class radosDiscoveryFixture : public testing::Test {
protected:
    void
    SetUp() override {
        if (std::getenv("NIXL_PLUGIN_DIR") == nullptr) {
            ASSERT_EQ(setenv("NIXL_PLUGIN_DIR", RADOS_TEST_PLUGIN_DIR, 1), 0)
                << "cannot export NIXL_PLUGIN_DIR";
        }
    }
};

TEST_F(radosDiscoveryFixture, CreateBackendRejectsUnprefixedParameter) {
    nixl_b_params_t parameters{{"conf", "missing-ceph.conf"},
                               {"pool", "qa-pool"},
                               {"name", "qa-client"},
                               {"qa_unprefixed_option", "1"}};
    const nixlAgentConfig agent_config;
    nixlAgent agent(localAgent, agent_config);

    nixlBackendH *backend = nullptr;
    EXPECT_EQ(agent.createBackend("RADOS", parameters, backend), NIXL_ERR_BACKEND);
    EXPECT_EQ(backend, nullptr);
}

TEST_F(radosDiscoveryFixture, LoadedPluginDeclaresExactlyDramAndObjSegments) {
    const nixlAgentConfig agent_config;
    nixlAgent agent(localAgent, agent_config);

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    ASSERT_EQ(agent.getPluginParams("RADOS", mems, params), NIXL_SUCCESS);
    EXPECT_EQ(mems, (nixl_mem_list_t{DRAM_SEG, OBJ_SEG}));
    EXPECT_TRUE(params.empty());
}

} // namespace gtest::rados
