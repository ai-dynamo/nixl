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

#include "common.h"

#include <plugin_manager.h>
#include <gtest/gtest.h>
#include <absl/flags/usage.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_split.h>

ABSL_FLAG(std::string, tests_plugin_dirs, "",
          "Comma-separated list of plugin directories");
ABSL_FLAG(bool, fast, false, "Run tests in fast mode");

namespace gtest {

static void ParseArguments(int argc, char **argv)
{
    absl::SetProgramUsageMessage("NIXL testing options");
    absl::ParseCommandLine(argc, argv);

    auto plugin_dirs = absl::GetFlag(FLAGS_tests_plugin_dirs);
    for (auto dir : absl::StrSplit(plugin_dirs, ',')) {
        if (dir.empty())
            continue;
        Logger() << "Adding plugin directory: " << dir;
        nixlPluginManager::getInstance().addPluginDirectory(std::string(dir));
    }
}

bool isFast()
{
    static bool fast = absl::GetFlag(FLAGS_fast);
    return fast;
}

} // namespace gtest

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    gtest::ParseArguments(argc, argv);
    gtest::Logger() << "Running tests in "
                    << (gtest::isFast() ? "fast" : "slow") << " mode";
    return RUN_ALL_TESTS();
}
