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
#include "plugin_manager.h"
#include <gtest/gtest.h>

namespace gtest {
namespace utils {

std::string ExtractValueFromArgument(const std::string &arg) {
  const std::string delimiter = "=";
  size_t pos = arg.find(delimiter);
  if (pos != std::string::npos) {
    return arg.substr(pos + delimiter.length());
  }
  return "";
}
} // namespace utils

void ParseArguments(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).find("--tests_plugin_dir=") == 0) {
      const std::string plugin_dir = utils::ExtractValueFromArgument(argv[i]);

      if (!plugin_dir.empty()) {
        std::cout << "Adding plugin directory:" << plugin_dir << std::endl;
        nixlPluginManager::getInstance().addPluginDirectory(plugin_dir);
      }
    }
  }
}

int RunTests(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ParseArguments(argc, argv);

  return RUN_ALL_TESTS();
}
} // namespace gtest

int main(int argc, char **argv) { return gtest::RunTests(argc, argv); }
