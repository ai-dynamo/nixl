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
#include <random>

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"

namespace gtest {
namespace agent {
namespace utils {
static constexpr const char *agent_name = "Agent";
static constexpr const char *nonexisting_plugin = "NonExistingPlugin";

/* Generates a random number in [0,255] (byte range). */
char GetRandomByte() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<char> distr(0, 255);
  return distr(gen);
}

class Blob {
protected:
  static constexpr size_t buf_len = 256;
  static constexpr uint32_t dev_id = 0;

  std::unique_ptr<char> buf_;
  const nixlBlobDesc desc_;
  const char buf_pattern_;

public:
  Blob()
      : buf_(std::make_unique<char>(buf_len)),
        desc_({.addr = buf_.get(), .len = buf_len, .devId = dev_id}),
        buf_pattern_(GetRandomByte()) {
    memset(buf_.get(), buf_pattern_, buf_len);
  }

  nixlBlobDesc GetDesc() const { return desc_; }
};
} // namespace utils

class CreateAgentFixture : public testing::Test {
protected:
  std::shared_ptr<nixlAgent> agent_;
  nixl_mem_list_t mem_;
  nixl_b_params_t params_;
  std::shared_ptr<nixlPluginHandle> plugin_handle_;

  void SetUp() override {
    nixlAgentConfig cfg(true);
    agent_ = std::make_unique<nixlAgent>(utils::agent_name, cfg);

    /* Non tested API can be asserted. */
    std::vector<nixl_backend_t> plugins;
    ASSERT_TRUE(agent_->getAvailPlugins(plugins) == NIXL_SUCCESS);
    ASSERT_TRUE(std::find(plugins.begin(), plugins.end(),
                          nixl_backend_t(utils::nonexisting_plugin)) ==
                plugins.end());
    plugin_handle_ =
        nixlPluginManager::getInstance().loadPlugin(mock_basic_plugin_name);
    ASSERT_TRUE(plugin_handle_ != nullptr);
  }

  void TearDown() override {
    nixlPluginManager::getInstance().unloadPlugin(mock_basic_plugin_name);
  }
};

TEST_F(CreateAgentFixture, GetNonExistingPluginTest) {
  EXPECT_NE(agent_->getPluginParams(nixl_backend_t(utils::nonexisting_plugin),
                                    mem_, params_),
            NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, GetExistingPluginTest) {
  std::vector<nixl_backend_t> plugins;

  EXPECT_EQ(agent_->getAvailPlugins(plugins), NIXL_SUCCESS);
  if (plugins.empty()) {
    GTEST_SKIP();
  }

  EXPECT_EQ(agent_->getPluginParams(plugins.front(), mem_, params_),
            NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, CreateNonExistingPluginBackendTest) {
  nixlBackendH* backend;

  EXPECT_NE(agent_->createBackend(utils::nonexisting_plugin, params_, backend),
            NIXL_SUCCESS);
}

TEST_F(CreateAgentFixture, CreateExistingPluginBackendTest) {
  nixlBackendH* backend;

  EXPECT_EQ(agent_->getPluginParams(mock_basic_plugin_name, mem_, params_),
            NIXL_SUCCESS);
  EXPECT_EQ(agent_->createBackend(mock_basic_plugin_name, params_, backend),
            NIXL_SUCCESS);
}
} // namespace agent
} // namespace gtest
