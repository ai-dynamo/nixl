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

#include "nixl.h"
#include "plugin_manager.h"

namespace gtest {
namespace plugin_manager {

class LoadSinglePluginTestFixture : public testing::TestWithParam<std::string> {
protected:
  /* Plugin Manager. */
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  /* Added plugin. */
  std::shared_ptr<nixlPluginHandle> plugin_handle_;

  void SetUp() override {
    plugin_handle_ = plugin_manager_.loadPlugin(GetParam());
  }

  void TearDown() override { plugin_manager_.unloadPlugin(GetParam()); }

  /* Returns true if the plugin was succesfully loaded, otherwise false. */
  bool IsLoaded() { return plugin_handle_ != nullptr; }
};

class LoadMultiplePluginsTestFixture
    : public testing::TestWithParam<std::vector<std::string>> {
protected:
  /* Plugin Manager. */
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  /* Added plugin. */
  std::vector<std::shared_ptr<nixlPluginHandle>> plugin_handles_;

  void SetUp() override {
    for (const auto &plugin : GetParam()) {
      plugin_handles_.push_back(plugin_manager_.loadPlugin(plugin));
    }
  }

  void TearDown() override {
    for (const auto &plugin : GetParam()) {
      plugin_manager_.unloadPlugin(plugin);
    }
  }

  /*
   * Returns true if all the plugins were succesfully loaded, otherwise false.
   */
  bool AreAllLoaded() {
    return all_of(
        plugin_handles_.begin(), plugin_handles_.end(),
        [](std::shared_ptr<nixlPluginHandle> ptr) { return ptr != nullptr; });
  }
};

TEST_P(LoadSinglePluginTestFixture, SimlpeLifeCycleTest) {
  EXPECT_TRUE(IsLoaded());
}

TEST_P(LoadMultiplePluginsTestFixture, SimlpeLifeCycleTest) {
  EXPECT_TRUE(AreAllLoaded());
}

/* Load single plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(UcxLoadPluginInstantiation,
                         LoadSinglePluginTestFixture, testing::Values("UCX"));
INSTANTIATE_TEST_SUITE_P(GdsLoadPluginInstantiation,
                         LoadSinglePluginTestFixture, testing::Values("GDS"));
INSTANTIATE_TEST_SUITE_P(UcxMoLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values("UCX_MO"));

/* Load single plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(UcxGdsLoadMultiplePluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values("UCX, GDS"));
INSTANTIATE_TEST_SUITE_P(UcxUcxMoLoadMultiplePluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values("UCX", "UCX_MO"));
INSTANTIATE_TEST_SUITE_P(GdsUcxMoLoadMultiplePluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values("GDS", "UCX_MO"));
INSTANTIATE_TEST_SUITE_P(UcxGdsUcxMoLoadMultiplePluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values("UCX", "GDS", "UCX_MO"));

} // namespace plugin_manager
} // namespace gtest

