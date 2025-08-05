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
#include "plugin_manager.h"
#include "common.h"
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace gtest {
std::vector<std::string> SplitWithDelimiter(const std::string &str,
                                            char delimiter) {
  std::istringstream tokenStream(str);
  std::vector<std::string> tokens;
  std::string token;

  while (std::getline(tokenStream, token, delimiter))
    tokens.push_back(token);

  return tokens;
}

void
ParseTcpPortRange(const std::string &arg) {
    if (arg.find("--min-tcp-port=") == 0) {
        const std::string min_port = SplitWithDelimiter(arg, '=').back();
        PortAllocator::instance().set_min_port(std::stoi(min_port));
    }

    if (arg.find("--max-tcp-port=") == 0) {
        const std::string max_port = SplitWithDelimiter(arg, '=').back();
        PortAllocator::instance().set_max_port(std::stoi(max_port));
    }
}

void ParseArguments(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).find("--tests_plugin_dirs=") == 0) {
      const std::string plugin_dirs = SplitWithDelimiter(argv[i], '=').back();

      if (!plugin_dirs.empty()) {
        for (const auto &dir : SplitWithDelimiter(plugin_dirs, ',')) {
          std::cout << "Adding plugin directory:" << dir << std::endl;
          nixlPluginManager::getInstance().addPluginDirectory(dir);
        }
      }
    }

    ParseTcpPortRange(argv[i]);
  }
}

int RunTests(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  ParseArguments(argc, argv);

  return RUN_ALL_TESTS();
}
} // namespace gtest

int main(int argc, char **argv) { return gtest::RunTests(argc, argv); }
