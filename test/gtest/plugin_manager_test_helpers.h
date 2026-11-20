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

#ifndef NIXL_TEST_GTEST_PLUGIN_MANAGER_TEST_HELPERS_H
#define NIXL_TEST_GTEST_PLUGIN_MANAGER_TEST_HELPERS_H

#include <filesystem>
#include <string>

#include "plugin_manager.h"

namespace gtest {
namespace plugin_manager {

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
} // namespace plugin_manager
} // namespace gtest

#endif // NIXL_TEST_GTEST_PLUGIN_MANAGER_TEST_HELPERS_H
