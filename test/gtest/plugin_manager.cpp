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

class LoadedPluginTestFixture : public testing::Test {
protected:
  /* Plugin Manager. */
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  /* Static plugins. */
  std::set<std::string> static_plugins_;
  /* Dynamically loaded plugins. */
  std::set<std::string> loaded_plugins_;

  void SetUp() override {
    for (const auto &plugin : nixlPluginManager::getStaticPlugins()) {
      static_plugins_.insert(plugin.name);
    }
  }

  void TearDown() override {
    for (const auto &plugin : loaded_plugins_) {
      plugin_manager_.unloadPlugin(plugin);
    }
  }

  /*
   * Load a plugin to plugin manager.
   *
   * Returns true if the plugin loaded succesfully, otherwise false.
   */
  bool LoadPlugin(std::string name) {
    loaded_plugins_.insert(name);
    return plugin_manager_.loadPlugin(name) != nullptr;
  }

  /* Unload a plugin from plugin manager. */
  void UnloadPlugin(std::string name) {
    plugin_manager_.unloadPlugin(name);
    loaded_plugins_.erase(name);
  }

  /*
   * Returns true if the only non static plugins are similar to the loaded ones,
   * otherwise false.
   */
  bool HasOnlyLoadedPlugins() {
    const auto &pm_loaded = plugin_manager_.getLoadedPluginNames();
    std::set<std::string> pm_loaded_set(pm_loaded.begin(), pm_loaded.end());
    std::set<std::string> expected;

    std::set_union(loaded_plugins_.begin(), loaded_plugins_.end(),
                   static_plugins_.begin(), static_plugins_.end(),
                   inserter(expected, expected.begin()));

    return expected == pm_loaded_set;
  }
};

TEST_P(LoadSinglePluginTestFixture, SimlpeLifeCycleTest) {
  EXPECT_TRUE(IsLoaded());
}

TEST_P(LoadMultiplePluginsTestFixture, SimlpeLifeCycleTest) {
  EXPECT_TRUE(AreAllLoaded());
}

TEST_F(LoadedPluginTestFixture, NoLoadedPluginsTest) {
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadSinglePluginTest) {
  EXPECT_TRUE(LoadPlugin("UCX"));
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadMultiplePluginsTest) {
  EXPECT_TRUE(LoadPlugin("UCX"));
  EXPECT_TRUE(LoadPlugin("GDS"));
  EXPECT_TRUE(LoadPlugin("UCX_MO"));
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadUnloadSimplePluginTest) {
  const std::string &plugin = "UCX";

  EXPECT_TRUE(LoadPlugin(plugin));
  UnloadPlugin(plugin);
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadUnloadComplexPluginTest) {
  const std::string &plugin0 = "UCX";
  const std::string &plugin1 = "GDS";

  EXPECT_TRUE(LoadPlugin(plugin0));
  EXPECT_TRUE(LoadPlugin(plugin1));
  UnloadPlugin(plugin0);
  EXPECT_TRUE(HasOnlyLoadedPlugins());

  EXPECT_TRUE(LoadPlugin(plugin0));
  EXPECT_TRUE(HasOnlyLoadedPlugins());

  UnloadPlugin(plugin0);
  UnloadPlugin(plugin1);
  EXPECT_TRUE(HasOnlyLoadedPlugins());
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

