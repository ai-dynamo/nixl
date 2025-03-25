/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
namespace utils {
struct PluginDesc {
  enum PluginType { Native, Mock };
  const char *name;
  const PluginType type;
};

/* Utililty print function for readable log. */
void PrintTo(const PluginDesc &plugin_desc, ::std::ostream *os) {
  *os << plugin_desc.name;
}

static constexpr const char *mock_basic_plugin_name = "MOCK_BASIC_PLUGIN";
static constexpr const char *mock_dram_plugin_name = "MOCK_DRAM_PLUGIN";

const PluginDesc ucx_plugin_desc{.name = "UCX",
                                 .type = utils::PluginDesc::PluginType::Native};
const PluginDesc gds_plugin_desc{.name = "GDS",
                                 .type = utils::PluginDesc::PluginType::Native};
const PluginDesc ucx_mo_plugin_desc{
    .name = "UCX_MO", .type = utils::PluginDesc::PluginType::Native};
} // namespace utils

class LoadSinglePluginTestFixture
    : public testing::TestWithParam<utils::PluginDesc> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::shared_ptr<nixlPluginHandle> plugin_handle_;

  void SetUp() override {
#if SKIP_NATIVE_PLUGINS
    if (GetParam().type == utils::PluginDesc::PluginType::Native) {
      GTEST_SKIP();
    }
#endif
    plugin_handle_ = plugin_manager_.loadPlugin(GetParam().name);
  }

  void TearDown() override {
#if SKIP_NATIVE_PLUGINS
    if (GetParam().type == utils::PluginDesc::PluginType::Native) {
      return;
    }
#endif
    plugin_manager_.unloadPlugin(GetParam().name);
  }

  /* Returns true if the plugin was succesfully loaded, otherwise false. */
  bool IsLoaded() { return plugin_handle_ != nullptr; }
};

class LoadMultiplePluginsTestFixture
    : public testing::TestWithParam<std::vector<utils::PluginDesc>> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::vector<std::shared_ptr<nixlPluginHandle>> plugin_handles_;

  void SetUp() override {
    for (const auto &plugin : GetParam()) {
#if SKIP_NATIVE_PLUGINS
      if (plugin.type == utils::PluginDesc::PluginType::Native) {
        continue;
      }
#endif
      plugin_handles_.push_back(plugin_manager_.loadPlugin(plugin.name));
    }
  }

  void TearDown() override {
    for (const auto &plugin : GetParam()) {
#if SKIP_NATIVE_PLUGINS
      if (plugin.type == utils::PluginDesc::PluginType::Native) {
        continue;
      }
#endif
      plugin_manager_.unloadPlugin(plugin.name);
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
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::set<std::string> prev_plugins_;
  std::set<std::string> loaded_plugins_;

  void SetUp() override {
    for (const auto &plugin : plugin_manager_.getLoadedPluginNames()) {
      prev_plugins_.insert(plugin);
    }
  }

  void TearDown() override {
    for (const auto &plugin : loaded_plugins_) {
      plugin_manager_.unloadPlugin(plugin);
    }
  }

  /*
   * Load a plugin to plugin manager if isn't already loaded.
   *
   * Returns true if the plugin loaded succesfully, otherwise false.
   */
  bool LoadPlugin(std::string name) {
    if (prev_plugins_.find(name) != prev_plugins_.end()) {
      return true;
    }
    if (loaded_plugins_.find(name) != loaded_plugins_.end()) {
      return false;
    }
    loaded_plugins_.insert(name);
    return plugin_manager_.loadPlugin(name) != nullptr;
  }

  /* Unload a plugin from plugin manager if it was loaded during this test. */
  void UnloadPlugin(std::string name) {
    if (loaded_plugins_.find(name) == loaded_plugins_.end()) {
      return;
    }
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
          (loaded_plugins_.find(pm_loaded_plugin) == loaded_plugins_.end())) {
        return false;
      }
    }

    return true;
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
  EXPECT_TRUE(LoadPlugin(utils::mock_basic_plugin_name));
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadMultiplePluginsTest) {
  EXPECT_TRUE(LoadPlugin(utils::mock_basic_plugin_name));
  EXPECT_TRUE(LoadPlugin(utils::mock_dram_plugin_name));
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadUnloadSimplePluginTest) {
  EXPECT_TRUE(LoadPlugin(utils::mock_basic_plugin_name));
  UnloadPlugin(utils::mock_basic_plugin_name);
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

TEST_F(LoadedPluginTestFixture, LoadUnloadComplexPluginTest) {
  EXPECT_TRUE(LoadPlugin(utils::mock_basic_plugin_name));
  EXPECT_TRUE(LoadPlugin(utils::mock_dram_plugin_name));
  UnloadPlugin(utils::mock_basic_plugin_name);
  EXPECT_TRUE(HasOnlyLoadedPlugins());

  EXPECT_TRUE(LoadPlugin(utils::mock_basic_plugin_name));
  EXPECT_TRUE(HasOnlyLoadedPlugins());

  UnloadPlugin(utils::mock_basic_plugin_name);
  UnloadPlugin(utils::mock_dram_plugin_name);
  EXPECT_TRUE(HasOnlyLoadedPlugins());
}

/* Load single plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(MockLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(utils::PluginDesc{
                             .name = utils::mock_basic_plugin_name,
                             .type = utils::PluginDesc::PluginType::Mock}));
INSTANTIATE_TEST_SUITE_P(UcxLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(utils::ucx_plugin_desc));
INSTANTIATE_TEST_SUITE_P(GdsLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(utils::gds_plugin_desc));
INSTANTIATE_TEST_SUITE_P(UcxMoLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(utils::ucx_mo_plugin_desc));

/* Load single plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(UcxGdsLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<utils::PluginDesc>{
                             utils::ucx_plugin_desc, utils::gds_plugin_desc}));
INSTANTIATE_TEST_SUITE_P(UcxUcxMoLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<utils::PluginDesc>{
                             utils::ucx_plugin_desc,
                             utils::ucx_mo_plugin_desc}));
INSTANTIATE_TEST_SUITE_P(GdsUcxMoLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<utils::PluginDesc>{
                             utils::gds_plugin_desc,
                             utils::ucx_mo_plugin_desc}));
INSTANTIATE_TEST_SUITE_P(UcxGdsUcxMoLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<utils::PluginDesc>{
                             utils::ucx_plugin_desc, utils::gds_plugin_desc,
                             utils::ucx_mo_plugin_desc}));

} // namespace plugin_manager
} // namespace gtest

