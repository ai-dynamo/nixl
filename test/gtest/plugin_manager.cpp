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

#include "common.h"
#include "nixl.h"
#include "plugin_manager.h"

namespace gtest {
namespace plugin_manager {
struct PluginDesc {
  enum PluginType { Real, Mock };
  const char *name;
  const PluginType type;
};

/* Utililty print function for readable log. */
void PrintTo(const PluginDesc &plugin_desc, ::std::ostream *os) {
  *os << plugin_desc.name;
}

const PluginDesc ucx_plugin_desc{.name = "UCX",
                                 .type = PluginDesc::PluginType::Real};
const PluginDesc gds_plugin_desc{.name = "GDS",
                                 .type = PluginDesc::PluginType::Real};

class LoadSinglePluginTestFixture
    : public testing::TestWithParam<PluginDesc> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::shared_ptr<const nixlPluginHandle> plugin_handle_;

  void SetUp() override {
#if !TEST_ALL_PLUGINS
    if (GetParam().type == PluginDesc::PluginType::Real)
      GTEST_SKIP();
#endif
    plugin_handle_ = plugin_manager_.loadPlugin(GetParam().name);
  }

  void TearDown() override {
#if !TEST_ALL_PLUGINS
    if (GetParam().type == PluginDesc::PluginType::Real)
      return;
#endif
    plugin_manager_.unloadPlugin(GetParam().name);
  }

  /* Returns true if the plugin was successfully loaded, otherwise false. */
  bool IsLoaded() { return plugin_handle_ != nullptr; }
};

class LoadMultiplePluginsTestFixture
    : public testing::TestWithParam<std::vector<PluginDesc>> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::vector<std::shared_ptr<const nixlPluginHandle>> plugin_handles_;

  void SetUp() override {
    for (const auto &plugin : GetParam()) {
#if !TEST_ALL_PLUGINS
      if (plugin.type == PluginDesc::PluginType::Real)
        continue;
#endif
      plugin_handles_.push_back(plugin_manager_.loadPlugin(plugin.name));
    }
  }

  void TearDown() override {
    for (const auto &plugin : GetParam()) {
#if !TEST_ALL_PLUGINS
      if (plugin.type == PluginDesc::PluginType::Real)
        continue;
#endif
      plugin_manager_.unloadPlugin(plugin.name);
    }
  }

  /*
   * Returns true if all the plugins were successfully loaded, otherwise false.
   */
  bool AreAllLoaded() {
    return all_of(
        plugin_handles_.begin(), plugin_handles_.end(),
        [](std::shared_ptr<const nixlPluginHandle> ptr) { return ptr != nullptr; });
  }
};

class LoadedPluginTestFixture : public testing::Test {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::set<std::string> prev_plugins_;
  std::set<std::string> loaded_plugins_;

  void SetUp() override {
    for (const auto &plugin : plugin_manager_.getLoadedPluginNames())
      prev_plugins_.insert(plugin);
  }

  void TearDown() override {
    for (const auto &plugin : loaded_plugins_)
      plugin_manager_.unloadPlugin(plugin);
  }

  /*
   * Load a plugin to plugin manager if isn't already loaded.
   *
   * Returns true if the plugin loaded successfully, otherwise false.
   */
  bool LoadPlugin(std::string name) {
    if (prev_plugins_.find(name) != prev_plugins_.end())
      return true;
    if (loaded_plugins_.find(name) != loaded_plugins_.end())
      return false;
    loaded_plugins_.insert(name);
    return plugin_manager_.loadPlugin(name) != nullptr;
  }

  /* Unload a plugin from plugin manager if it was loaded during this test. */
  void UnloadPlugin(std::string name) {
    if (loaded_plugins_.find(name) == loaded_plugins_.end())
      return;
    plugin_manager_.unloadPlugin(name);
    loaded_plugins_.erase(name);
  }

  /*
   * Returns true if the only non static plugins are similar to the loaded ones,
   * otherwise false.
   */
  bool HasOnlyLoadedPlugins() {
    const auto &pm_loaded = plugin_manager_.getLoadedPluginNames();
    for (const auto &pm_loaded_plugin : pm_loaded) {
      if ((prev_plugins_.find(pm_loaded_plugin) == prev_plugins_.end()) &&
          (loaded_plugins_.find(pm_loaded_plugin) == loaded_plugins_.end()))
        return false;
    }

    return true;
  }
};

TEST_P(LoadSinglePluginTestFixture, SimpleLifeCycleTest) {
  EXPECT_TRUE(IsLoaded());
}

TEST_P(LoadMultiplePluginsTestFixture, SimpleLifeCycleTest) {
  EXPECT_TRUE(AreAllLoaded());
}

TEST_F(LoadedPluginTestFixture, NoLoadedPluginsTest) {
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadSinglePluginTest) {
    EXPECT_TRUE(LoadPlugin(GetMockBackendName()));
    EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadUnloadSimplePluginTest) {
    EXPECT_TRUE(LoadPlugin(GetMockBackendName()));
    UnloadPlugin(GetMockBackendName());
    EXPECT_TRUE(HasOnlyLoadedPlugins());
}

/* Load single plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(MockLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(PluginDesc{.name = GetMockBackendName(),
                                                    .type = PluginDesc::PluginType::Mock}));
INSTANTIATE_TEST_SUITE_P(UcxLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(ucx_plugin_desc));
INSTANTIATE_TEST_SUITE_P(GdsLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(gds_plugin_desc));

/* Load multiple plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(UcxGdsLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<PluginDesc>{
                             ucx_plugin_desc, gds_plugin_desc}));
} // namespace plugin_manager
} // namespace gtest
