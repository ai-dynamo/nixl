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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include <absl/cleanup/cleanup.h>
#include <gtest/gtest.h>

#include "agent_data.h"
#include "common.h"
#include "mocks/gmock_engine.h"
#include "plugin_manager.h"

namespace gtest {
namespace {

    TEST(BackendNameResolution, AgentUsesOneCanonicalEngine) {
        auto &manager = nixlPluginManager::getInstance();
        const auto *canonical = GetMockBackendName();
        const bool was_loaded = manager.getBackendPlugin(canonical) != nullptr;
        const absl::Cleanup restore = [&] {
            if (!was_loaded) {
                manager.unloadBackendPluginForUnitTest(canonical);
            }
        };

        testing::NiceMock<mocks::GMockBackendEngine> engine;
        nixlAgent agent("backend_name_test", nixlAgentConfig{});
        nixl_b_params_t params;
        nixl_mem_list_t mems;
        nixlBackendH *backend = nullptr;
        const std::string alias = " \tmock_Backend\r\n";
        ASSERT_EQ(agent.getPluginParams(alias, mems, params), NIXL_SUCCESS);
        engine.SetToParams(params);
        ASSERT_EQ(agent.createBackend(alias, params, backend), NIXL_SUCCESS);
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->getType(), canonical);
        EXPECT_EQ(manager.getBackendPlugin(canonical), manager.loadBackendPlugin(alias));
        const LogIgnoreGuard duplicate("backend already created");
        EXPECT_EQ(agent.createBackend(canonical, params, backend), NIXL_ERR_INVALID_PARAM);
    }

    TEST(BackendNameResolution, RejectsEmptyNames) {
        nixlAgent agent("empty_name_test", nixlAgentConfig{});
        nixl_b_params_t params;
        nixl_mem_list_t mems;
        nixlBackendH *backend = nullptr;
        const LogIgnoreGuard empty("Empty backend name");
        for (const auto *name : {"", " \t\r\n\f\v"}) {
            EXPECT_EQ(agent.getPluginParams(name, mems, params), NIXL_ERR_INVALID_PARAM);
            EXPECT_EQ(agent.createBackend(name, params, backend), NIXL_ERR_INVALID_PARAM);
        }
    }

    void
    checkAmbiguousPluginList() {
        auto &manager = nixlPluginManager::getInstance();
        std::string list = testing::TempDir() + "/nixl-pluginlist-XXXXXX";
        const int fd = mkstemp(list.data());
        ASSERT_NE(fd, -1);
        close(fd);
        const absl::Cleanup cleanup = [&] { std::filesystem::remove(list); };
        ASSERT_TRUE(std::ofstream(list) << "mock_backend=\n");
        manager.discoverPluginsFromList(list);

        nixl_backend_t name;
        ASSERT_EQ(manager.canonicalBackendName(" mock_backend ", name), NIXL_SUCCESS);
        EXPECT_EQ(name, "mock_backend");
        const LogIgnoreGuard ambiguous(
            "Ambiguous backend name 'Mock_Backend': MOCK_BACKEND, mock_backend");
        EXPECT_EQ(manager.canonicalBackendName("Mock_Backend", name), NIXL_ERR_NOT_FOUND);
        ASSERT_NE(manager.loadBackendPlugin(GetMockBackendName()), nullptr);
        EXPECT_EQ(manager.loadBackendPlugin("Mock_Backend"), nullptr);
        EXPECT_EQ(manager.getBackendPlugin("Mock_Backend"), nullptr);
        nixl_mem_list_t mems;
        nixl_b_params_t params;
        EXPECT_EQ(manager.getBackendParams("Mock_Backend", mems, params), NIXL_ERR_NOT_FOUND);
        EXPECT_EQ(ambiguous.getIgnoredCount(), 4u);
        EXPECT_EQ(manager.getBackendPlugin("mock_backend"), nullptr);
    }

    TEST(BackendNameResolutionDeathTest, LoadedAndListedSpellingsMustBothBeConsidered) {
        // The plugin list belongs to the process-wide manager and has no unregister operation.
        EXPECT_EXIT(
            {
                checkAmbiguousPluginList();
                std::_Exit(testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
            },
            testing::ExitedWithCode(EXIT_SUCCESS),
            "");
    }

} // namespace
} // namespace gtest
