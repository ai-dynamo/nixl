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
#include "plugin_manager.h"
#include "common.h"

#include <absl/strings/str_split.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>
#include <string>

namespace gtest {

namespace {

[[nodiscard]] std::optional<std::string>
CheckArgument(const std::string &arg, const std::string &prefix) {
    if (arg.starts_with(prefix)) {
        return arg.substr(prefix.size());
    }
    return std::nullopt;
}

void
ParseTcpPortRange(const std::string &arg) {
    if (const auto val = CheckArgument(arg, "--min-tcp-port=")) {
        PortAllocator::instance().set_min_port(std::stoi(*val));
    } else if (const auto val = CheckArgument(arg, "--max-tcp-port=")) {
        PortAllocator::instance().set_max_port(std::stoi(*val));
    }
}

std::vector<std::string> test_plugin_dirs;

void
ParsePluginDirs(const std::string &arg) {
    if (const auto val = CheckArgument(arg, "--tests_plugin_dirs="); val && !val->empty()) {
        const std::vector<std::string> plugin_dirs = absl::StrSplit(*val, ',');
        for (const auto &dir : plugin_dirs) {
            std::cout << "Adding plugin directory:" << dir << std::endl;
            test_plugin_dirs.emplace_back(dir);
        }
    }
}

void
ParseArguments(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        ParsePluginDirs(arg);
        ParseTcpPortRange(arg);
    }
}

class PluginReinitializer
    : public testing::EmptyTestEventListener
{
    void OnTestStart(const testing::TestInfo&) override {
        std::cout << "Reinitializing plugin manager" << std::endl;

        auto& pm = nixlPluginManager::getInstance();

        pm.reinitializeForUnitTest();

        for (const std::string &dir : test_plugin_dirs) {
            pm.addPluginDirectory(dir);
        }
    }
};

const std::regex
    ib_regex("IB device\\(s\\) were detected, but accelerated IB support was not found");
const std::regex aws_regex("UCX version is less than 1.19, CUDA support is limited, including"
                           " the lack of support for multi-GPU within a single process.");
const std::regex non_gpu_regex("[0-9]+ NVIDIA GPU\\(s\\) were detected, but UCX CUDA support "
                               "was not found! GPU memory is not supported.");

int
RunAllTests() {
    LogProblemCounter lpc;
    std::list<LogIgnoreGuard> ligs;

    // TODO: Remove after the CI issues spuriously triggering this message are fixed.
    ligs.emplace_back(ib_regex);

    if (std::getenv("AWS_BATCH_JOB_ID") != nullptr) {
        ligs.emplace_back(aws_regex);
    }

    if (std::getenv("NIXL_CI_NON_GPU") != nullptr) {
        ligs.emplace_back(non_gpu_regex);
    }

    return RUN_ALL_TESTS();
}

int RunTests(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    ParseArguments(argc, argv);

    auto *instance = testing::UnitTest::GetInstance();
    auto &listeners = instance->listeners();
    listeners.Append(new PluginReinitializer);

    const int result = RunAllTests();

    if (const size_t problems = LogProblemCounter::getProblemCount(); problems > 0) {
        std::cerr << "ATTENTION: Unexpected NIXL warning(s) and/or error(s) detected!" << std::endl;
        std::cerr << "ATTENTION: Problem count is " << problems << std::endl;
        return 42;
    }
    return result;
}

} // namespace

} // namespace gtest

int main(int argc, char **argv) { return gtest::RunTests(argc, argv); }
