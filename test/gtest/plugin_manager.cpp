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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "agent_data.h"
#include "mocks/gmock_engine.h"

#include "backend/backend_plugin.h"
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

/* Unloads a plugin at scope exit unless it was already loaded when the test
 * started; runs even when an assertion fails. */
struct CacheRestore {
    std::string name;
    bool unload;

    ~CacheRestore() {
        if (unload) {
            nixlPluginManager::getInstance().unloadBackendPluginForUnitTest(name);
        }
    }
};

/* Removes a directory tree at scope exit, even when an assertion fails. */
struct DirRemove {
    std::filesystem::path dir;

    ~DirRemove() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

/* A plugin directory registered with the plugin manager for the duration of
 * a scope: deregisters it (even when an assertion fails) and removes the
 * tree. */
struct TempPluginDir {
    std::filesystem::path dir;
    bool registered = false;

    ~TempPluginDir() {
        if (registered) {
            nixlPluginManager::getInstance().removePluginDirectoryForUnitTest(dir.string());
        }
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};
const PluginDesc gds_plugin_desc{.name = "GDS",
                                 .type = PluginDesc::PluginType::Real};
#ifdef HAVE_LIBFABRIC
const PluginDesc libfabric_plugin_desc{.name = "LIBFABRIC", .type = PluginDesc::PluginType::Real};
#endif

class LoadSinglePluginTestFixture
    : public testing::TestWithParam<PluginDesc> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::shared_ptr<const nixlBackendPluginHandle> plugin_handle_;

  void SetUp() override {
#if !TEST_ALL_PLUGINS
    if (GetParam().type == PluginDesc::PluginType::Real)
      GTEST_SKIP();
#endif
    plugin_handle_ = plugin_manager_.loadBackendPlugin(GetParam().name);
  }

  void TearDown() override {
#if !TEST_ALL_PLUGINS
    if (GetParam().type == PluginDesc::PluginType::Real)
      return;
#endif
    plugin_manager_.unloadBackendPluginForUnitTest(GetParam().name);
  }

  /* Returns true if the plugin was successfully loaded, otherwise false. */
  bool IsLoaded() { return plugin_handle_ != nullptr; }
};

class LoadMultiplePluginsTestFixture
    : public testing::TestWithParam<std::vector<PluginDesc>> {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::vector<std::shared_ptr<const nixlBackendPluginHandle>> plugin_handles_;

  void SetUp() override {
    for (const auto &plugin : GetParam()) {
#if !TEST_ALL_PLUGINS
      if (plugin.type == PluginDesc::PluginType::Real)
        continue;
#endif
      plugin_handles_.push_back(plugin_manager_.loadBackendPlugin(plugin.name));
    }
  }

  void TearDown() override {
    for (const auto &plugin : GetParam()) {
#if !TEST_ALL_PLUGINS
      if (plugin.type == PluginDesc::PluginType::Real)
        continue;
#endif
      plugin_manager_.unloadBackendPluginForUnitTest(plugin.name);
    }
  }

  /*
   * Returns true if all the plugins were successfully loaded, otherwise false.
   */
  bool AreAllLoaded() {
      return all_of(
          plugin_handles_.begin(),
          plugin_handles_.end(),
          [](std::shared_ptr<const nixlBackendPluginHandle> ptr) { return ptr != nullptr; });
  }
};

class LoadedPluginTestFixture : public testing::Test {
protected:
  nixlPluginManager &plugin_manager_ = nixlPluginManager::getInstance();
  std::set<std::string> prev_plugins_;
  std::set<std::string> loaded_plugins_;

  void SetUp() override {
      for (const auto &plugin : plugin_manager_.getLoadedBackendPluginNames())
          prev_plugins_.insert(plugin);
  }

  void TearDown() override {
    for (const auto &plugin : loaded_plugins_)
        plugin_manager_.unloadBackendPluginForUnitTest(plugin);
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
    return plugin_manager_.loadBackendPlugin(name) != nullptr;
  }

  /* Unload a plugin from plugin manager if it was loaded during this test. */
  void UnloadPlugin(std::string name) {
    if (loaded_plugins_.find(name) == loaded_plugins_.end())
      return;
    plugin_manager_.unloadBackendPluginForUnitTest(name);
    loaded_plugins_.erase(name);
  }

  /*
   * Returns true if the only non static plugins are similar to the loaded ones,
   * otherwise false.
   */
  bool
  HasOnlyLoadedPlugins() {
      const auto &pm_loaded = plugin_manager_.getLoadedBackendPluginNames();
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

TEST_F(LoadedPluginTestFixture, DeferredDiscoveryTest) {
    auto mock = GetMockBackendName();
    auto avail = plugin_manager_.getAvailBackendPluginNames();
    auto loaded = plugin_manager_.getLoadedBackendPluginNames();

    // Available plugins should include discovered-but-not-loaded ones
    EXPECT_GT(avail.size(), loaded.size());
    EXPECT_NE(std::find(avail.begin(), avail.end(), mock), avail.end());
    EXPECT_EQ(std::find(loaded.begin(), loaded.end(), mock), loaded.end());

    // After loading, it should appear in both
    EXPECT_TRUE(LoadPlugin(mock));
    loaded = plugin_manager_.getLoadedBackendPluginNames();
    EXPECT_NE(std::find(loaded.begin(), loaded.end(), mock), loaded.end());

    // After unloading, still available (discovered on disk) but not loaded
    UnloadPlugin(mock);
    loaded = plugin_manager_.getLoadedBackendPluginNames();
    EXPECT_EQ(std::find(loaded.begin(), loaded.end(), mock), loaded.end());
    avail = plugin_manager_.getAvailBackendPluginNames();
    EXPECT_NE(std::find(avail.begin(), avail.end(), mock), avail.end());
}

TEST_F(LoadedPluginTestFixture, StaleTelemetryPluginApiVersionIsRejected) {
    const LogIgnoreGuard lig_mismatch("Plugin API version mismatch");
    const LogIgnoreGuard lig_missing("Plugin file does not exist");

    EXPECT_EQ(plugin_manager_.loadTelemetryPlugin("mock_stale"), nullptr);
    EXPECT_GE(lig_mismatch.getIgnoredCount(), 1u);
}

TEST_F(LoadedPluginTestFixture, StaleTracePluginApiVersionIsRejected) {
    const LogIgnoreGuard lig_mismatch("Plugin API version mismatch");

    EXPECT_EQ(plugin_manager_.loadTracePlugin("mock_stale"), nullptr);
    EXPECT_GE(lig_mismatch.getIgnoredCount(), 1u);
}

TEST_F(LoadedPluginTestFixture, StalePluginApiVersionIsRejected) {
    const LogIgnoreGuard lig_mismatch("Plugin API version mismatch");
    const LogIgnoreGuard lig_missing("Plugin file does not exist");

    const auto avail = plugin_manager_.getAvailBackendPluginNames();
    ASSERT_NE(std::find(avail.begin(), avail.end(), "MOCK_STALE_BACKEND"), avail.end());

    EXPECT_EQ(plugin_manager_.loadBackendPlugin("MOCK_STALE_BACKEND"), nullptr);

    const auto loaded = plugin_manager_.getLoadedBackendPluginNames();
    EXPECT_EQ(std::find(loaded.begin(), loaded.end(), "MOCK_STALE_BACKEND"), loaded.end());
}

TEST_F(LoadedPluginTestFixture, CurrentPluginApiVersionIsAccepted) {
    EXPECT_TRUE(LoadPlugin(GetMockBackendName()));
    EXPECT_EQ(NIXL_PLUGIN_API_VERSION, 4);
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

TEST_F(LoadedPluginTestFixture, LibfabricPluginAdvertisesPostThreadOptions) {
#if defined(HAVE_LIBFABRIC) && TEST_ALL_PLUGINS
    auto plugin_handle = plugin_manager_.getBackendPlugin("LIBFABRIC");
    if (!plugin_handle) {
        plugin_handle = plugin_manager_.loadBackendPlugin("LIBFABRIC");
        if (plugin_handle) {
            loaded_plugins_.insert("LIBFABRIC");
        }
    }
    ASSERT_NE(plugin_handle, nullptr);

    const auto backend_options = plugin_handle->getBackendOptions();
    ASSERT_NE(backend_options.find("num_threads"), backend_options.end());
    ASSERT_NE(backend_options.find("split_batch_size"), backend_options.end());
    EXPECT_EQ(backend_options.at("num_threads"), "0");
    EXPECT_EQ(backend_options.at("split_batch_size"), "1024");
#else
    GTEST_SKIP();
#endif
}

/* Backend names coming from users may differ in case and carry surrounding
 * whitespace; every string-keyed entry point should resolve them to the
 * canonical plugin name. */
TEST_F(LoadedPluginTestFixture, BackendNamesAreCaseAndSpaceInsensitive) {
    const std::string canonical = GetMockBackendName();
    std::string lower(canonical);
    std::transform(
        lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    const std::string padded = "  " + lower + " \t";

    // Resolution maps any spelling onto the canonical plugin name, rejects
    // whitespace-only names, and returns unknown names trimmed.
    EXPECT_EQ(plugin_manager_.canonicalBackendName(padded), canonical);
    EXPECT_EQ(plugin_manager_.canonicalBackendName("   "), "");
    EXPECT_EQ(plugin_manager_.canonicalBackendName(" no_such_plugin "), "no_such_plugin");
    EXPECT_EQ(plugin_manager_.loadBackendPlugin("   "), nullptr);

    // Snapshot the cache keys so the alias load below can be shown to add
    // at most the canonical entry; restore the cache even if an assertion
    // below fails.
    auto before = plugin_manager_.getLoadedBackendPluginNames();
    std::sort(before.begin(), before.end());
    const bool was_loaded = std::find(before.begin(), before.end(), canonical) != before.end();
    CacheRestore cache_restore{canonical, !was_loaded};

    // Loading through an alias reports the canonical name it resolved to.
    nixl_backend_t resolved;
    const auto alias_handle = plugin_manager_.loadBackendPlugin(padded, &resolved);
    ASSERT_NE(alias_handle, nullptr);
    EXPECT_EQ(resolved, canonical);
    EXPECT_EQ(alias_handle.get(), plugin_manager_.loadBackendPlugin(canonical).get());
    EXPECT_NE(plugin_manager_.getBackendPlugin(lower), nullptr);
    nixl_mem_list_t mems;
    nixl_b_params_t params;
    EXPECT_EQ(plugin_manager_.getBackendParams(padded, mems, params), NIXL_SUCCESS);

    auto after = plugin_manager_.getLoadedBackendPluginNames();
    std::sort(after.begin(), after.end());
    auto expected = before;
    expected.push_back(canonical);
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    EXPECT_EQ(after, expected);
}

/* An agent must key backends by the canonical name, so creating the same
 * backend through two spellings is rejected as a duplicate. */
TEST(BackendNameCanonicalization, AgentRejectsAliasDuplicateBackend) {
    const std::string canonical = GetMockBackendName();
    std::string lower(canonical);
    std::transform(
        lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });

    auto &plugin_manager = nixlPluginManager::getInstance();
    const auto loaded_before = plugin_manager.getLoadedBackendPluginNames();
    const bool was_loaded =
        std::find(loaded_before.begin(), loaded_before.end(), canonical) != loaded_before.end();
    // Restore the plugin cache even if an assertion below fails.
    CacheRestore cache_restore{canonical, !was_loaded};

    // The mock backend wires this engine in through the init parameters; it
    // must outlive the agent.
    mocks::GMockBackendEngine gmock_engine;

    {
        nixlAgent agent("canonical_names_agent", nixlAgentConfig(true));
        nixl_b_params_t params;
        gmock_engine.SetToParams(params);
        nixlBackendH *first = nullptr;
        ASSERT_EQ(agent.createBackend("  " + lower + " ", params, first), NIXL_SUCCESS);
        // The engine is keyed by the canonical name the manager resolved to.
        ASSERT_NE(first, nullptr);
        EXPECT_EQ(first->getType(), canonical);

        // The duplicate rejection logs an error on purpose.
        LogIgnoreGuard ignore_dup("backend already created");
        nixlBackendH *second = nullptr;
        EXPECT_EQ(agent.createBackend(canonical, params, second), NIXL_ERR_INVALID_PARAM);
    }
}

/* A plugin file with a non-canonical name ("libplugin_ Foo .so") must be
 * advertised under its canonical name, exactly once, before any load. */
TEST(BackendNameCanonicalization, NonCanonicalPluginFileIsAdvertisedOnce) {
    namespace fs = std::filesystem;
    // A name and directory exclusive to this test invocation, so repeated
    // runs and parallel tests cannot collide.
    static std::atomic<int> run_counter{0};
    const int run = run_counter++;
    const std::string plugin =
        "SPACED_TEST_" + std::to_string(::getpid()) + "_" + std::to_string(run);
    const fs::path dir = fs::path(testing::TempDir()) /
        ("nixl_spaced_plugin_" + std::to_string(::getpid()) + "_" + std::to_string(run));
    ASSERT_TRUE(fs::create_directories(dir)) << dir;
    TempPluginDir guard{dir};
    const fs::path file = dir / ("libplugin_ " + plugin + " .so");
    {
        std::ofstream out(file);
        ASSERT_TRUE(out.good()) << file;
    }

    auto &plugin_manager = nixlPluginManager::getInstance();
    // The name must not exist before the scan; a leftover from an earlier
    // run would silently satisfy the count below.
    const auto before = plugin_manager.getAvailBackendPluginNames();
    ASSERT_EQ(std::count(before.begin(), before.end(), plugin), 0) << plugin;

    plugin_manager.addPluginDirectory(dir.string());
    guard.registered = true;

    const auto avail = plugin_manager.getAvailBackendPluginNames();
    EXPECT_EQ(std::count(avail.begin(), avail.end(), plugin), 1);
}

namespace {
    /* Locate the real mock plugin: prefer a registered plugin directory (works
     * for installed test binaries), fall back to the build tree. */
    std::optional<std::filesystem::path>
    findMockPlugin() {
        const std::string name = "libplugin_" + std::string(GetMockBackendName()) + ".so";
        for (const auto &dir : nixlPluginManager::getInstance().getPluginDirectories()) {
            const std::filesystem::path candidate = std::filesystem::path(dir) / name;
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
#ifdef BUILD_DIR
        const std::filesystem::path build_candidate =
            std::filesystem::path(BUILD_DIR) / "test/gtest/mocks" / name;
        if (std::filesystem::exists(build_candidate)) {
            return build_candidate;
        }
#endif
        return std::nullopt;
    }

    /* Copy the mock plugin, optionally renaming its embedded name string so two
     * copies can be told apart after loading. Returns false on failure. */
    bool
    writeMockCopy(const std::filesystem::path &mock,
                  const std::filesystem::path &dst,
                  bool rename_plugin) {
        static_assert(sizeof("MOCK_BACK2ND") == sizeof("MOCK_BACKEND"),
                      "marker must keep the embedded string length");
        std::ifstream in(mock, std::ios::binary);
        if (!in.good()) {
            return false;
        }
        std::string buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (rename_plugin) {
            const std::string from = GetMockBackendName();
            const std::string to = "MOCK_BACK2ND";
            if (from.size() != to.size()) {
                return false;
            }
            if (buf.find(from) == std::string::npos) {
                return false;
            }
            size_t pos = 0;
            while ((pos = buf.find(from, pos)) != std::string::npos) {
                buf.replace(pos, from.size(), to);
                pos += to.size();
            }
        }
        std::ofstream out(dst, std::ios::binary);
        out.write(buf.data(), buf.size());
        return out.good();
    }
} // namespace

/* A plugin behind a non-canonical file name is taken from the directory
 * with the highest registration priority, and falls back to lower-priority
 * directories when a copy disappeared. */
TEST(BackendNameCanonicalization, NonCanonicalFileFollowsDirectoryPriority) {
    namespace fs = std::filesystem;
    const auto mock = findMockPlugin();
    if (!mock.has_value()) {
        GTEST_SKIP() << "no mock plugin file available";
    }

    static std::atomic<int> run_counter{0};
    const int run = run_counter++;
    const auto make_dir = [&](int i) {
        const fs::path dir = fs::path(testing::TempDir()) /
            ("nixl_spaced_load_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_" +
             std::to_string(i));
        if (!fs::create_directories(dir)) {
            return fs::path();
        }
        return dir;
    };

    auto &plugin_manager = nixlPluginManager::getInstance();

    // Case 1: both copies alive; the newer directory's copy must win. Its
    // copy is renamed so the loaded plugin can be identified by name.
    {
        const std::string plugin =
            "MOCK_SPACED_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_1";
        const fs::path dir_a = make_dir(1);
        ASSERT_FALSE(dir_a.empty()) << dir_a;
        TempPluginDir guard_a{dir_a};
        const fs::path dir_b = make_dir(2);
        ASSERT_FALSE(dir_b.empty()) << dir_b;
        TempPluginDir guard_b{dir_b};
        ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so"), false));
        ASSERT_TRUE(writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so"), true));

        plugin_manager.addPluginDirectory(dir_a.string());
        guard_a.registered = true;
        plugin_manager.addPluginDirectory(dir_b.string());
        guard_b.registered = true;

        const auto handle = plugin_manager.loadBackendPlugin(plugin);
        ASSERT_NE(handle, nullptr);
        EXPECT_STREQ(handle->getName(), "MOCK_BACK2ND");
        plugin_manager.unloadBackendPluginForUnitTest(plugin);
    }

    // Case 2: the newer directory's copy disappears; the load must fall back
    // to the older directory's unmodified copy.
    {
        const std::string plugin =
            "MOCK_SPACED_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_2";
        const fs::path dir_a = make_dir(3);
        ASSERT_FALSE(dir_a.empty()) << dir_a;
        TempPluginDir guard_a{dir_a};
        const fs::path dir_b = make_dir(4);
        ASSERT_FALSE(dir_b.empty()) << dir_b;
        TempPluginDir guard_b{dir_b};
        ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so"), false));
        ASSERT_TRUE(writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so"), false));

        plugin_manager.addPluginDirectory(dir_a.string());
        guard_a.registered = true;
        plugin_manager.addPluginDirectory(dir_b.string());
        guard_b.registered = true;

        std::error_code ec;
        ASSERT_TRUE(fs::remove(dir_b / ("libplugin_ " + plugin + " .so"), ec)) << ec.message();

        const auto handle = plugin_manager.loadBackendPlugin(plugin);
        ASSERT_NE(handle, nullptr);
        EXPECT_STREQ(handle->getName(), GetMockBackendName());
        plugin_manager.unloadBackendPluginForUnitTest(plugin);
    }

    // Case 3: a plugin loaded from A stays reloadable from B after A is
    // removed, because B's source is recorded even while the plugin is
    // loaded.
    {
        const std::string plugin =
            "MOCK_SPACED_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_3";
        const fs::path dir_a = make_dir(5);
        ASSERT_FALSE(dir_a.empty()) << dir_a;
        TempPluginDir guard_a{dir_a};
        const fs::path dir_b = make_dir(6);
        ASSERT_FALSE(dir_b.empty()) << dir_b;
        TempPluginDir guard_b{dir_b};
        ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so"), false));
        ASSERT_TRUE(writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so"), true));

        plugin_manager.addPluginDirectory(dir_a.string());
        guard_a.registered = true;

        auto handle = plugin_manager.loadBackendPlugin(plugin);
        ASSERT_NE(handle, nullptr);
        EXPECT_STREQ(handle->getName(), GetMockBackendName());

        plugin_manager.addPluginDirectory(dir_b.string());
        guard_b.registered = true;
        plugin_manager.unloadBackendPluginForUnitTest(plugin);
        plugin_manager.removePluginDirectoryForUnitTest(dir_a.string());
        guard_a.registered = false;

        handle = plugin_manager.loadBackendPlugin(plugin);
        ASSERT_NE(handle, nullptr);
        EXPECT_STREQ(handle->getName(), "MOCK_BACK2ND");
        plugin_manager.unloadBackendPluginForUnitTest(plugin);
    }
}

/* A scan carrying a superseded registration generation of the same directory
 * path is dropped, while the fresh registration's scan takes effect. The
 * generations of two registrations of one path must differ, so a report with
 * the first generation is stale after the second registration; reusing
 * numbers after a deregistration would publish it anyway. */
TEST(BackendNameCanonicalization, StaleScanOfReRegisteredDirectoryIsDropped) {
    namespace fs = std::filesystem;
    static std::atomic<int> run_counter{0};
    const int run = run_counter++;
    const std::string suffix = std::to_string(::getpid()) + "_" + std::to_string(run) + "_stale";
    const std::string discovered = "MOCK_SPACED_" + suffix + "_fresh";
    const std::string reported = "MOCK_SPACED_" + suffix + "_report";
    const fs::path dir = fs::path(testing::TempDir()) / ("nixl_spaced_stale_" + suffix);
    ASSERT_TRUE(fs::create_directories(dir)) << dir;
    TempPluginDir guard{dir};

    auto &plugin_manager = nixlPluginManager::getInstance();
    const auto count_avail = [&plugin_manager](const std::string &name) {
        const auto avail = plugin_manager.getAvailBackendPluginNames();
        return std::count(avail.begin(), avail.end(), name);
    };

    // Register with one plugin already in place: the scan publishes it once.
    const fs::path discovered_file = dir / ("libplugin_ " + discovered + " .so");
    {
        std::ofstream out(discovered_file);
        ASSERT_TRUE(out.good()) << discovered_file;
    }
    plugin_manager.addPluginDirectory(dir.string());
    guard.registered = true;
    const uint64_t first_generation =
        plugin_manager.pluginDirectoryScanGenerationForUnitTest(dir.string());
    ASSERT_NE(first_generation, 0);
    EXPECT_EQ(count_avail(discovered), 1);

    // Deregistering and re-registering the same path must allocate a fresh
    // generation and rediscover the plugin exactly once with it.
    plugin_manager.removePluginDirectoryForUnitTest(dir.string());
    guard.registered = false;
    plugin_manager.addPluginDirectory(dir.string());
    guard.registered = true;
    const uint64_t second_generation =
        plugin_manager.pluginDirectoryScanGenerationForUnitTest(dir.string());
    ASSERT_NE(second_generation, first_generation);
    EXPECT_EQ(count_avail(discovered), 1);

    // A report of a second plugin carrying the superseded generation must be
    // discarded...
    const fs::path reported_file = dir / ("libplugin_ " + reported + " .so");
    {
        std::ofstream out(reported_file);
        ASSERT_TRUE(out.good()) << reported_file;
    }
    plugin_manager.discoverBackendPluginWithGenerationForUnitTest(
        reported_file, dir.string(), first_generation);
    EXPECT_EQ(count_avail(reported), 0);
    // ...while the same report with the current generation publishes it.
    plugin_manager.discoverBackendPluginWithGenerationForUnitTest(
        reported_file, dir.string(), second_generation);
    EXPECT_EQ(count_avail(reported), 1);
}

/* Removing a directory literally named "pluginlist" must not unregister
 * plugins whose source is the pluginlist file. */
TEST(BackendNameCanonicalization, RemovingPluginlistNamedDirectoryKeepsListedNames) {
    namespace fs = std::filesystem;
    static std::atomic<int> run_counter{0};
    const int run = run_counter++;
    // An exclusive working directory; the guard is installed before anything
    // else is created inside it.
    const fs::path work = fs::path(testing::TempDir()) /
        ("nixl_pluginlist_dir_" + std::to_string(::getpid()) + "_" + std::to_string(run));
    ASSERT_TRUE(fs::create_directories(work)) << work;
    DirRemove cleanup{work};

    // Register a unique pluginlist-only name through a temporary list file.
    const std::string listed =
        "LISTED_ONLY_" + std::to_string(::getpid()) + "_" + std::to_string(run);
    const fs::path list_file = work / "pluginlist.txt";
    {
        std::ofstream out(list_file);
        ASSERT_TRUE(out.good()) << list_file;
        out << listed << "=\n";
    }
    auto &plugin_manager = nixlPluginManager::getInstance();

    // Restore the singleton on every exit path: forget the list-only name,
    // and only remove the "pluginlist" directory when this test added it.
    struct RegistryRestore {
        nixlPluginManager &manager;
        std::string name;
        std::string dir;
        bool remove_dir;

        ~RegistryRestore() {
            manager.forgetDiscoveredPluginForUnitTest(name);
            if (remove_dir) {
                manager.removePluginDirectoryForUnitTest(dir);
            }
        }
    } registry_restore{plugin_manager, listed, "pluginlist", false};

    plugin_manager.discoverPluginsFromList(list_file.string());

    const auto count_listed = [&plugin_manager, &listed]() {
        const auto avail = plugin_manager.getAvailBackendPluginNames();
        return std::count(avail.begin(), avail.end(), listed);
    };
    ASSERT_EQ(count_listed(), 1) << listed << " must come from the list file";

    // Register and remove a directory under the literal key "pluginlist",
    // relative to an isolated working directory.
    std::error_code ec;
    const fs::path old_cwd = fs::current_path(ec);
    ASSERT_FALSE(ec) << ec.message();
    fs::current_path(work, ec);
    ASSERT_FALSE(ec) << ec.message();

    struct CwdRestore {
        fs::path path;

        ~CwdRestore() {
            std::error_code restore_ec;
            fs::current_path(path, restore_ec);
        }
    } cwd_restore{old_cwd};

    // The exercise below registers and removes a directory under the literal
    // key "pluginlist"; with such a key pre-registered by the environment the
    // add would only warn (failing the log check) and the remove would delete
    // the pre-existing registration, so skip instead.
    const auto registered_dirs = plugin_manager.getPluginDirectories();
    if (std::find(registered_dirs.begin(), registered_dirs.end(), "pluginlist") !=
        registered_dirs.end()) {
        GTEST_SKIP() << "a 'pluginlist' plugin directory is pre-registered";
    }
    ASSERT_TRUE(fs::create_directories("pluginlist"));
    plugin_manager.addPluginDirectory("pluginlist");
    registry_restore.remove_dir = true;
    plugin_manager.removePluginDirectoryForUnitTest("pluginlist");
    registry_restore.remove_dir = false;

    // The list-file source must be untouched by the directory removal.
    EXPECT_EQ(count_listed(), 1);
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
#ifdef HAVE_LIBFABRIC
INSTANTIATE_TEST_SUITE_P(LibfabricLoadPluginInstantiation,
                         LoadSinglePluginTestFixture,
                         testing::Values(libfabric_plugin_desc));
#endif

/* Load multiple plugins tests instantiations. */
INSTANTIATE_TEST_SUITE_P(UcxGdsLoadMultiplePluginInstantiation,
                         LoadMultiplePluginsTestFixture,
                         testing::Values(std::vector<PluginDesc>{
                             ucx_plugin_desc, gds_plugin_desc}));
} // namespace plugin_manager
} // namespace gtest
