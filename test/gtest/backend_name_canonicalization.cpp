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

#include <gtest/gtest.h>

#include "agent_data.h"
#include "common.h"
#include "mocks/gmock_engine.h"
#include "plugin_manager.h"
#include "plugin_manager_test_helpers.h"

namespace gtest {
namespace plugin_manager {

    /* An agent must key backends by the canonical name, so creating the same
     * backend through two spellings is rejected as a duplicate. */
    TEST(BackendNameCanonicalization, AgentRejectsAliasDuplicateBackend) {
        const std::string canonical = GetMockBackendName();
        std::string lower(canonical);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return std::tolower(c);
        });

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
        /* ASCII fold of a name, matching the plugin manager's comparison rule. */
        std::string
        upperFold(std::string s) {
            std::transform(
                s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
            return s;
        }

        /* Locate a built mock plugin by its registered name: prefer a registered
         * plugin directory (works for installed test binaries), fall back to the
         * build tree. */
        std::optional<std::filesystem::path>
        findMockPlugin(const std::string &plugin_name) {
            const std::string name = "libplugin_" + plugin_name + ".so";
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

        /* Copy the mock plugin, optionally renaming its embedded name string so
         * copies can be told apart after loading. The marker may be shorter than
         * the original; it is NUL-padded to keep the buffer length. Returns false
         * on failure. */
        bool
        writeMockCopy(const std::filesystem::path &mock,
                      const std::filesystem::path &dst,
                      const std::string &rename_to = "") {
            std::ifstream in(mock, std::ios::binary);
            if (!in.good()) {
                return false;
            }
            std::string buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!rename_to.empty()) {
                const std::string from = GetMockBackendName();
                if (rename_to.size() > from.size()) {
                    return false;
                }
                if (buf.find(from) == std::string::npos) {
                    return false;
                }
                const std::string to =
                    rename_to + std::string(from.size() - rename_to.size(), '\0');
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
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        const auto make_dir = [&](int i) {
            const fs::path dir = fs::path(testing::TempDir()) /
                ("nixl_spaced_load_" + std::to_string(::getpid()) + "_" + std::to_string(run) +
                 "_" + std::to_string(i));
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
            ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so")));
            ASSERT_TRUE(
                writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so"), "MOCK_BACK2ND"));

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
            ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so")));
            ASSERT_TRUE(writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so")));

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
            ASSERT_TRUE(writeMockCopy(*mock, dir_a / ("libplugin_ " + plugin + " .so")));
            ASSERT_TRUE(
                writeMockCopy(*mock, dir_b / ("libplugin_ " + plugin + " .so"), "MOCK_BACK2ND"));

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

            // B's recorded source keeps the name advertised after A is gone.
            {
                const auto avail = plugin_manager.getAvailBackendPluginNames();
                EXPECT_EQ(std::count(avail.begin(), avail.end(), plugin), 1);
            }

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
        const std::string suffix =
            std::to_string(::getpid()) + "_" + std::to_string(run) + "_stale";
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

    /* The fold between two same-fold spellings from different sources must be
     * stable: whichever case variant another agent loads, one agent's repeated
     * alias request keeps resolving to the same plugin, so its duplicate check
     * still rejects the repeat. */
    TEST(BackendNameCanonicalization, AliasResolutionIsStableAcrossAgents) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        // Run-unique same-fold spellings so no pre-existing registration of the
        // same name can be hit; getType() (from init_params.type) distinguishes
        // the engines, so the copies keep the embedded mock name.
        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        const std::string listed = "dual_" + std::to_string(::getpid()) + "_" + std::to_string(run);
        std::string dir_name = listed;
        std::transform(dir_name.begin(), dir_name.end(), dir_name.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        std::string alias = listed;
        alias[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(alias[0])));
        const fs::path work = fs::path(testing::TempDir()) /
            ("nixl_dual_" + std::to_string(::getpid()) + "_" + std::to_string(run));
        ASSERT_TRUE(fs::create_directories(work)) << work;
        DirRemove cleanup{work};

        // The pluginlist names the lowercase copy (kept outside any registered
        // directory); a registered directory provides the uppercase copy under a
        // standard file name.
        const fs::path listed_file = work / ("libplugin_" + listed + ".so");
        ASSERT_TRUE(writeMockCopy(*mock, listed_file));
        const fs::path list_file = work / "pluginlist.txt";
        {
            std::ofstream out(list_file);
            ASSERT_TRUE(out.good()) << list_file;
            out << listed << "=" << listed_file.string() << "\n";
        }
        const fs::path dir = work / "plugindir";
        ASSERT_TRUE(fs::create_directories(dir)) << dir;
        TempPluginDir dir_guard{dir};
        ASSERT_TRUE(writeMockCopy(*mock, dir / ("libplugin_" + dir_name + ".so")));

        auto &plugin_manager = nixlPluginManager::getInstance();
        // A pre-existing registration of any same-fold spelling would be
        // overwritten or unloaded by the guard's alias-resolving cleanup; skip
        // instead.
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(listed);
            for (const auto &name : avail) {
                if (upperFold(name) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << name << "' is already registered";
                }
            }
        }

        // Restore the singleton even when an assertion fails: unload both plugins
        // (unique names, never loaded before this test) and forget every source.
        struct RegistryRestore {
            nixlPluginManager &manager;
            std::string a;
            std::string b;

            ~RegistryRestore() {
                manager.unloadBackendPluginForUnitTest(a);
                manager.unloadBackendPluginForUnitTest(b);
                manager.forgetDiscoveredPluginForUnitTest(a);
                manager.forgetDiscoveredPluginForUnitTest(b);
            }
        } registry_restore{plugin_manager, listed, dir_name};

        plugin_manager.discoverPluginsFromList(list_file.string());
        plugin_manager.addPluginDirectory(dir.string());
        dir_guard.registered = true;

        // Agent A resolves the mixed-case alias to the pluginlist spelling.
        mocks::GMockBackendEngine engine_a;
        nixlAgent agent_a("dual_agent_a", nixlAgentConfig(true));
        {
            nixl_b_params_t params;
            engine_a.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_a.createBackend(" " + alias + " ", params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), listed);
        }

        // Agent B exact-creates the directory spelling, so both spellings are now
        // loaded and the uppercase one sorts first in the plugin cache.
        {
            mocks::GMockBackendEngine engine_b;
            nixlAgent agent_b("dual_agent_b", nixlAgentConfig(true));
            nixl_b_params_t params;
            engine_b.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_b.createBackend(dir_name, params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), dir_name);
        }

        // The manager itself must keep resolving the alias to the pluginlist
        // spelling even though the directory spelling is now loaded and sorts
        // first in the plugin cache.
        EXPECT_EQ(plugin_manager.canonicalBackendName(" " + alias + " "), listed);

        // The same alias request in agent A must resolve to the same spelling
        // again and be rejected as a duplicate, not create a second, case-variant
        // engine. Empty params on purpose: reaching engine creation would throw
        // in the mock and fail the test.
        {
            LogIgnoreGuard ignore_dup("backend already created");
            nixl_b_params_t params;
            nixlBackendH *second = nullptr;
            EXPECT_EQ(agent_a.createBackend(" " + alias + " ", params, second),
                      NIXL_ERR_INVALID_PARAM);
        }
    }

    /* A standard plugin file that a registered directory's scan has not seen
     * (written after registration, before any rescan) still wins over folding to
     * an already-loaded case variant: the requested spelling loads its own
     * library, exactly as it would once discovered. */
    TEST(BackendNameCanonicalization, UndiscoveredStandardFileWinsOverLoadedVariant) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        // Short markers so the copies can be renamed inside the mock binary.
        const std::string tag = std::to_string(::getpid() % 1000000) + "_" + std::to_string(run);
        const std::string upper = "W" + tag;
        const std::string lower = "w" + tag;
        ASSERT_LE(upper.size(), std::string(GetMockBackendName()).size());

        // One directory per spelling: on a case-insensitive file system two
        // case-variant names in one directory are one file, and writing the
        // second copy would truncate the first.
        const fs::path dir_upper = fs::path(testing::TempDir()) /
            ("nixl_window_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_u");
        ASSERT_TRUE(fs::create_directories(dir_upper)) << dir_upper;
        TempPluginDir guard_upper{dir_upper};
        const fs::path dir_lower = fs::path(testing::TempDir()) /
            ("nixl_window_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_l");
        ASSERT_TRUE(fs::create_directories(dir_lower)) << dir_lower;
        TempPluginDir guard_lower{dir_lower};

        auto &plugin_manager = nixlPluginManager::getInstance();
        // Unload only what this test loads: no same-fold spelling is loaded or
        // registered anywhere (an environment registration would satisfy the
        // exact-name check below through the plugin tables instead of the disk
        // probe this test exercises).
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(lower);
            for (const auto &name : avail) {
                if (upperFold(name) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << name << "' is already registered";
                }
            }
        }
        CacheRestore upper_restore{upper, true};
        CacheRestore lower_restore{lower, true};

        // Register the uppercase directory with its copy in place, and load it.
        ASSERT_TRUE(writeMockCopy(*mock, dir_upper / ("libplugin_" + upper + ".so"), upper));
        plugin_manager.addPluginDirectory(dir_upper.string());
        guard_upper.registered = true;
        const auto upper_handle = plugin_manager.loadBackendPlugin(upper);
        ASSERT_NE(upper_handle, nullptr);
        EXPECT_STREQ(upper_handle->getName(), upper.c_str());

        // Register the lowercase directory while empty, then drop the lowercase
        // copy in without rescanning. A params query for the lowercase spelling
        // must succeed even though it is not loaded and not discovered yet, and
        // loading it must return the lowercase library, not fold to the loaded
        // uppercase one.
        plugin_manager.addPluginDirectory(dir_lower.string());
        guard_lower.registered = true;
        ASSERT_TRUE(writeMockCopy(*mock, dir_lower / ("libplugin_" + lower + ".so"), lower));
        nixl_mem_list_t mems;
        nixl_b_params_t params;
        EXPECT_EQ(plugin_manager.getBackendParams(lower, mems, params), NIXL_SUCCESS);
        const auto lower_handle = plugin_manager.loadBackendPlugin(lower);
        ASSERT_NE(lower_handle, nullptr);
        EXPECT_STREQ(lower_handle->getName(), lower.c_str());
        EXPECT_NE(lower_handle.get(), upper_handle.get());
        const auto loaded = plugin_manager.getLoadedBackendPluginNames();
        EXPECT_EQ(std::count(loaded.begin(), loaded.end(), lower), 1);
        EXPECT_EQ(std::count(loaded.begin(), loaded.end(), upper), 1);
    }

    /* A standard file name a directory scan really saw stays loadable when the
     * directory later cannot be enumerated at all (read permission removed,
     * search permission kept): the loader falls back to verifying the recorded
     * path. When enumeration still works (e.g. running as root), the plain
     * exact-entry match serves the same load. */
    TEST(BackendNameCanonicalization, ScanSeenStandardFileLoadsFromUnlistableDir) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        // Short marker so the copy can be renamed inside the mock binary.
        const std::string tag = std::to_string(::getpid() % 1000000) + "_" + std::to_string(run);
        const std::string name = "x" + tag;
        ASSERT_LE(name.size(), std::string(GetMockBackendName()).size());

        const fs::path dir = fs::path(testing::TempDir()) /
            ("nixl_unlistable_" + std::to_string(::getpid()) + "_" + std::to_string(run));
        ASSERT_TRUE(fs::create_directories(dir)) << dir;
        TempPluginDir guard{dir};

        auto &plugin_manager = nixlPluginManager::getInstance();
        // Unload only what this test loads: no same-fold spelling exists yet.
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(name);
            for (const auto &candidate : avail) {
                if (upperFold(candidate) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << candidate
                                 << "' is already registered";
                }
            }
        }
        CacheRestore restore{name, true};

        // The scan must see the standard file name while the directory is still
        // readable; only that record keeps the name loadable without enumeration.
        ASSERT_TRUE(writeMockCopy(*mock, dir / ("libplugin_" + name + ".so"), name));
        plugin_manager.addPluginDirectory(dir.string());
        guard.registered = true;
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            ASSERT_EQ(std::count(avail.begin(), avail.end(), name), 1);
        }

        // Drop read permission (keep search) and load without re-scanning.
        std::error_code ec;
        fs::permissions(dir, fs::perms::owner_read, fs::perm_options::remove, ec);
        ASSERT_FALSE(ec) << ec.message();

        struct PermsRestore {
            fs::path dir;

            ~PermsRestore() {
                std::error_code restore_ec;
                fs::permissions(dir, fs::perms::owner_read, fs::perm_options::add, restore_ec);
            }
        } perms_restore{dir};

        // The fallback only engages when enumeration actually fails; a root
        // process can still list the directory.
        {
            std::error_code probe_ec;
            (void)fs::directory_iterator{dir, probe_ec};
            if (!probe_ec) {
                GTEST_SKIP() << "directory remained listable (running as root?)";
            }
        }

        const auto handle = plugin_manager.loadBackendPlugin(name);
        ASSERT_NE(handle, nullptr);
        EXPECT_STREQ(handle->getName(), name.c_str());
    }

    /* The loader's standard-path compatibility gate, exercised directly: an
     * enumeration that completed always wins, and only a scan that saw the
     * standard file name keeps an unlistable directory loadable. */
    TEST(BackendNameCanonicalization, StandardPathFallbackGate) {
        using Lookup = nixlPluginManager::EntryLookup;
        EXPECT_FALSE(
            nixlPluginManager::standardPathFallbackForUnitTest(Lookup::NO_MATCH, true, true));
        EXPECT_FALSE(
            nixlPluginManager::standardPathFallbackForUnitTest(Lookup::NO_MATCH, false, true));
        EXPECT_TRUE(
            nixlPluginManager::standardPathFallbackForUnitTest(Lookup::CANNOT_LIST, true, true));
        EXPECT_FALSE(
            nixlPluginManager::standardPathFallbackForUnitTest(Lookup::CANNOT_LIST, true, false));
        EXPECT_FALSE(
            nixlPluginManager::standardPathFallbackForUnitTest(Lookup::CANNOT_LIST, false, true));
    }

    /* The alias duplicate check is re-verified against the load's own
     * resolution: when the exact file the first resolution saw disappears before
     * the load resolves again, the request must be rejected instead of silently
     * creating a second, case-variant engine. The one-shot pre-load hook makes
     * that state change deterministic inside a single createBackend call. */
    TEST(BackendNameCanonicalization, AliasDuplicateRecheckedAfterResolutionShift) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        const std::string lower = "hk_" + std::to_string(::getpid()) + "_" + std::to_string(run);
        std::string upper = lower;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        std::string mixed = lower;
        mixed[1] = static_cast<char>(std::toupper(static_cast<unsigned char>(mixed[1])));

        const fs::path work = fs::path(testing::TempDir()) /
            ("nixl_shift_" + std::to_string(::getpid()) + "_" + std::to_string(run));
        ASSERT_TRUE(fs::create_directories(work)) << work;
        DirRemove cleanup{work};
        const fs::path dir = work / "plugindir";
        ASSERT_TRUE(fs::create_directories(dir)) << dir;
        TempPluginDir dir_guard{dir};
        const fs::path mixed_dir = work / "mixeddir";
        ASSERT_TRUE(fs::create_directories(mixed_dir)) << mixed_dir;
        TempPluginDir mixed_dir_guard{mixed_dir};

        auto &plugin_manager = nixlPluginManager::getInstance();
        // Unload only what this test loads: no same-fold spelling exists yet.
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(lower);
            for (const auto &name : avail) {
                if (upperFold(name) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << name << "' is already registered";
                }
            }
        }
        CacheRestore lower_restore{lower, true};
        CacheRestore upper_restore{upper, true};

        struct ListForget {
            nixlPluginManager &manager;
            std::string name;

            ~ListForget() {
                manager.forgetDiscoveredPluginForUnitTest(name);
            }
        } list_forget{plugin_manager, upper};

        // The pluginlist carries a valid library for the uppercase spelling; a
        // registered directory carries the lowercase one. getType() distinguishes
        // the engines, so the copies keep the embedded mock name.
        const fs::path upper_file = work / ("libplugin_" + upper + ".so");
        ASSERT_TRUE(writeMockCopy(*mock, upper_file));
        const fs::path list_file = work / "pluginlist.txt";
        {
            std::ofstream out(list_file);
            ASSERT_TRUE(out.good()) << list_file;
            out << upper << "=" << upper_file.string() << "\n";
        }
        ASSERT_TRUE(writeMockCopy(*mock, dir / ("libplugin_" + lower + ".so")));
        plugin_manager.discoverPluginsFromList(list_file.string());
        plugin_manager.addPluginDirectory(dir.string());
        dir_guard.registered = true;

        // A exact-creates the lowercase spelling.
        mocks::GMockBackendEngine engine_a;
        nixlAgent agent_a("shift_agent_a", nixlAgentConfig(true));
        {
            nixl_b_params_t params;
            engine_a.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_a.createBackend(lower, params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), lower);
        }

        // A mixed-case request resolves exactly while libplugin_<mixed>.so is on
        // disk: its own directory is registered only now (so the lowercase load
        // above never probed it), and on a case-insensitive file system the two
        // case-variant libraries stay separate files in separate directories.
        plugin_manager.addPluginDirectory(mixed_dir.string());
        mixed_dir_guard.registered = true;
        const fs::path mixed_file = mixed_dir / ("libplugin_" + mixed + ".so");
        ASSERT_TRUE(writeMockCopy(*mock, mixed_file));
        EXPECT_EQ(plugin_manager.canonicalBackendName(" " + mixed + " "), mixed);

        // The one-shot hook removes the file between that resolution and the
        // load's own resolution, which must then fold to the listed uppercase
        // spelling -- and the duplicate check must still reject the request. The
        // hook captures by value and a guard disarms it on every exit path, so
        // an unconsumed hook cannot outlive this test; the flag proves the call
        // reached the load. Should the post-load re-check disappear, this call
        // would proceed to engine creation and succeed.
        const auto fired = std::make_shared<bool>(false);
        plugin_manager.setPreLoadHookForUnitTest([mixed_file, fired]() {
            std::error_code ec;
            fs::remove(mixed_file, ec);
            *fired = true;
        });

        struct HookGuard {
            nixlPluginManager &manager;

            ~HookGuard() {
                manager.setPreLoadHookForUnitTest(nullptr);
            }
        } hook_guard{plugin_manager};

        {
            LogIgnoreGuard ignore_dup("backend already created");
            nixl_b_params_t params;
            engine_a.SetToParams(params);
            nixlBackendH *second = nullptr;
            EXPECT_EQ(agent_a.createBackend(" " + mixed + " ", params, second),
                      NIXL_ERR_INVALID_PARAM);
        }
        EXPECT_TRUE(*fired);
    }

    /* A params query for an undiscovered standard file consults that file, not
     * the already-loaded case variant: with a stale-ABI mock behind the lower
     * spelling and a valid upper library cached, the query must fail on the
     * stale file rather than succeed off the cached one. */
    TEST(BackendNameCanonicalization, UndiscoveredStaleFileIsProbedNotFolded) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        const auto stale = findMockPlugin("MOCK_STALE_BACKEND");
        if (!mock.has_value() || !stale.has_value()) {
            GTEST_SKIP() << "no mock plugin files available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        // Short marker so the valid copy can be renamed inside the mock binary.
        const std::string tag = std::to_string(::getpid() % 1000000) + "_" + std::to_string(run);
        const std::string upper = "S" + tag;
        const std::string lower = "s" + tag;
        ASSERT_LE(upper.size(), std::string(GetMockBackendName()).size());

        // One directory per spelling, as on case-insensitive file systems two
        // case-variant names in one directory are one file.
        const fs::path dir_upper = fs::path(testing::TempDir()) /
            ("nixl_stale_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_u");
        ASSERT_TRUE(fs::create_directories(dir_upper)) << dir_upper;
        TempPluginDir guard_upper{dir_upper};
        const fs::path dir_lower = fs::path(testing::TempDir()) /
            ("nixl_stale_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_l");
        ASSERT_TRUE(fs::create_directories(dir_lower)) << dir_lower;
        TempPluginDir guard_lower{dir_lower};

        auto &plugin_manager = nixlPluginManager::getInstance();
        // Unload only what this test loads: no same-fold spelling exists yet.
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(lower);
            for (const auto &name : avail) {
                if (upperFold(name) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << name << "' is already registered";
                }
            }
        }
        CacheRestore upper_restore{upper, true};
        CacheRestore lower_restore{lower, true};

        // A valid upper library: registered, discovered and loaded.
        ASSERT_TRUE(writeMockCopy(*mock, dir_upper / ("libplugin_" + upper + ".so"), upper));
        plugin_manager.addPluginDirectory(dir_upper.string());
        guard_upper.registered = true;
        {
            const auto handle = plugin_manager.loadBackendPlugin(upper);
            ASSERT_NE(handle, nullptr);
            EXPECT_STREQ(handle->getName(), upper.c_str());
        }

        // The stale-ABI mock lands behind the lowercase spelling after
        // registration, undiscovered; the rejection logs on purpose, and the
        // loader walks on to the other directories afterwards.
        plugin_manager.addPluginDirectory(dir_lower.string());
        guard_lower.registered = true;
        std::error_code ec;
        ASSERT_TRUE(fs::copy_file(*stale, dir_lower / ("libplugin_" + lower + ".so"), ec))
            << ec.message();
        {
            const LogIgnoreGuard lig_mismatch("Plugin API version mismatch");
            const LogIgnoreGuard lig_missing("Plugin file does not exist");
            nixl_mem_list_t mems;
            nixl_b_params_t params;
            EXPECT_EQ(plugin_manager.getBackendParams(lower, mems, params), NIXL_ERR_NOT_FOUND);
            EXPECT_GE(lig_mismatch.getIgnoredCount(), 1u);
        }
        // The cached upper library keeps serving params queries.
        nixl_mem_list_t mems;
        nixl_b_params_t params;
        EXPECT_EQ(plugin_manager.getBackendParams(upper, mems, params), NIXL_SUCCESS);
    }

    /* The duplicate rejection of an alias must not depend on which same-fold
     * spellings other agents loaded from disk without any registration. */
    TEST(BackendNameCanonicalization, AliasDuplicateStableForUnregisteredSpellings) {
        namespace fs = std::filesystem;
        const auto mock = findMockPlugin(GetMockBackendName());
        if (!mock.has_value()) {
            GTEST_SKIP() << "no mock plugin file available";
        }

        static std::atomic<int> run_counter{0};
        const int run = run_counter++;
        const std::string lower = "un_" + std::to_string(::getpid()) + "_" + std::to_string(run);
        std::string upper = lower;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        std::string alias = lower;
        alias[1] = static_cast<char>(std::toupper(static_cast<unsigned char>(alias[1])));

        // One directory per spelling: on a case-insensitive file system two
        // case-variant names in one directory are one file.
        const fs::path dir_lower = fs::path(testing::TempDir()) /
            ("nixl_unreg_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_l");
        ASSERT_TRUE(fs::create_directories(dir_lower)) << dir_lower;
        TempPluginDir guard_lower{dir_lower};
        const fs::path dir_upper = fs::path(testing::TempDir()) /
            ("nixl_unreg_" + std::to_string(::getpid()) + "_" + std::to_string(run) + "_u");
        ASSERT_TRUE(fs::create_directories(dir_upper)) << dir_upper;
        TempPluginDir guard_upper{dir_upper};

        auto &plugin_manager = nixlPluginManager::getInstance();
        // Unload only what this test loads: no same-fold spelling exists yet, so
        // the guards' alias-resolving unload can only hit keys this test adds.
        {
            const auto avail = plugin_manager.getAvailBackendPluginNames();
            const std::string fold = upperFold(lower);
            for (const auto &name : avail) {
                if (upperFold(name) == fold) {
                    GTEST_SKIP() << "a same-fold plugin '" << name << "' is already registered";
                }
            }
        }
        CacheRestore lower_restore{lower, true};
        CacheRestore upper_restore{upper, true};

        // Register the lowercase directory while empty and drop its file in:
        // the name is registered nowhere, it is only loadable from disk. A
        // exact-creates it, then the same-fold alias is a duplicate.
        plugin_manager.addPluginDirectory(dir_lower.string());
        guard_lower.registered = true;
        ASSERT_TRUE(writeMockCopy(*mock, dir_lower / ("libplugin_" + lower + ".so")));

        mocks::GMockBackendEngine engine_a;
        nixlAgent agent_a("unreg_agent_a", nixlAgentConfig(true));
        {
            nixl_b_params_t params;
            engine_a.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_a.createBackend(lower, params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), lower);
        }
        {
            LogIgnoreGuard ignore_dup("backend already created");
            nixl_b_params_t params;
            nixlBackendH *second = nullptr;
            EXPECT_EQ(agent_a.createBackend(alias, params, second), NIXL_ERR_INVALID_PARAM);
        }

        // Only now register the uppercase directory (new directories take
        // priority, so the lowercase load above never probed it and emitted no
        // missing-file warning) and let B exact-create the uppercase spelling
        // (exact requests keep creating distinct spellings); A's alias must stay
        // a duplicate.
        plugin_manager.addPluginDirectory(dir_upper.string());
        guard_upper.registered = true;
        ASSERT_TRUE(writeMockCopy(*mock, dir_upper / ("libplugin_" + upper + ".so")));
        {
            mocks::GMockBackendEngine engine_b;
            nixlAgent agent_b("unreg_agent_b", nixlAgentConfig(true));
            nixl_b_params_t params;
            engine_b.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_b.createBackend(upper, params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), upper);
        }
        {
            LogIgnoreGuard ignore_dup("backend already created");
            nixl_b_params_t params;
            nixlBackendH *second = nullptr;
            EXPECT_EQ(agent_a.createBackend(alias, params, second), NIXL_ERR_INVALID_PARAM);
        }

        // A padded but exact request for the uppercase spelling still creates it:
        // only alias spellings are held against same-fold engines.
        {
            nixl_b_params_t params;
            engine_a.SetToParams(params);
            nixlBackendH *first = nullptr;
            ASSERT_EQ(agent_a.createBackend(" " + upper + " ", params, first), NIXL_SUCCESS);
            ASSERT_NE(first, nullptr);
            EXPECT_EQ(first->getType(), upper);
        }
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

} // namespace plugin_manager
} // namespace gtest
