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

#include <fnmatch.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <list>
#include <mutex>
#include <set>
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

namespace {
    const std::set<std::string> non_efa_skips = {
        "HardwareWarningTest.EfaHardwareMismatchWarning",
        "HardwareWarningTest.EfaHardwareMismatchNoWarning",
        "LoadedPluginTestFixture.LibfabricPluginAdvertisesPostThreadOptions",
        "LibfabricLoadPluginInstantiation/LoadSinglePluginTestFixture.SimpleLifeCycleTest/0"};

    const std::set<std::string> non_gpu_skips = {
        "HardwareWarningTest.WarnWhenGpuPresentButCudaNotSupported",
        "HardwareWarningTest.WarnWhenIbPresentButRdmaNotSupported",
        "HardwareWarningTest.NoWarningWhenIbAndCudaSupported",
        "ucxDeviceApi*"};

    const std::set<std::string> nvtx_skips = {"*TestTransferTracing.Nvtx*",
                                              "*TestTransferTracingNsys*"};

    const std::set<std::string> san_skips = {"nixlDurationTest.*"};

    std::mutex mutex;
    std::vector<std::string> required_but_skipped;

} // namespace

class SkippedTestsChecker : public testing::EmptyTestEventListener {
public:
    void
    allowForSkip(const std::set<std::string> &set) {
        allowed_for_skip_.insert(set.begin(), set.end());
    }

private:
    std::set<std::string> allowed_for_skip_ = {
#if !defined(LOAD_ALL_PLUGINS)
        "UcxLoadPluginInstantiation/LoadSinglePluginTestFixture.SimpleLifeCycleTest/0",
        "GdsLoadPluginInstantiation/LoadSinglePluginTestFixture.SimpleLifeCycleTest/0",
        "LoadedPluginTestFixture.LibfabricPluginAdvertisesPostThreadOptions",
        "LibfabricLoadPluginInstantiation/LoadSinglePluginTestFixture.SimpleLifeCycleTest/0"
#endif
    };

    void
    OnTestEnd(const testing::TestInfo &info) override {
        if (const auto *result = info.result()) {
            if (result->Skipped()) {
                const std::string s = info.test_suite_name();
                const std::string n = info.name();
                const std::string name = s + "." + n;

                for (const std::string &allowed : allowed_for_skip_) {
                    if (::fnmatch(allowed.c_str(), name.c_str(), 0) == 0) {
                        return;
                    }
                }
                const std::scoped_lock ml(mutex);
                required_but_skipped.push_back(name);
            }
        }
    }
};


namespace {
    const std::regex
        ib_regex("IB device\\(s\\) were detected, but accelerated IB support was not found");
    const std::regex aws_regex("UCX version is less than 1.19, CUDA support is limited, including"
                               " the lack of support for multi-GPU within a single process.");
    const std::regex non_gpu_regex("[0-9]+ NVIDIA GPU\\(s\\) were detected, but UCX CUDA support "
                                   "was not found! GPU memory is not supported.");

} // namespace

int
RunAllTests() {
    LogProblemCounter lpc;
    std::list<LogIgnoreGuard> ligs;
    auto *stc = new SkippedTestsChecker;
    {
        auto *instance = testing::UnitTest::GetInstance();
        auto &listeners = instance->listeners();
        listeners.Append(stc); // Takes ownership and deletes "when the program finishes".
    }

    // TODO: Remove after the CI issues spuriously triggering this message are fixed.
    ligs.emplace_back(ib_regex);

    if (std::getenv("AWS_BATCH_JOB_ID") != nullptr) {
        ligs.emplace_back(aws_regex);
    }

    if (std::getenv("NIXL_CI_NON_GPU") != nullptr) {
        stc->allowForSkip(non_gpu_skips);
        std::cerr << "ALLOWING GPU tests to be skipped" << std::endl;
        ligs.emplace_back(non_gpu_regex);
    }

    const char *var = std::getenv("TEST_LIBFABRIC");
    if (var && (var == std::string("false"))) {
        stc->allowForSkip(non_efa_skips);
        std::cerr << "ALLOWING EFA tests to be skipped" << std::endl;
    }

    if (std::getenv("NIXL_CI_ALLOW_NVTX_SKIP") != nullptr) {
        stc->allowForSkip(nvtx_skips);
        std::cerr << "ALLOWING NVTX tracing tests to be skipped" << std::endl;
    }

    if (std::getenv("SAN_LABEL") != nullptr) {
        stc->allowForSkip(san_skips);
        std::cerr << "ALLOWING SANITIZER tests to be skipped" << std::endl;
    }

    return RUN_ALL_TESTS();
}

int RunTests(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    ParseArguments(argc, argv);

    int result = RunAllTests();

    if (const size_t skipped = required_but_skipped.size(); skipped > 0) {
        std::cerr << "ATTENTION: Required tests skipped without skip being allowed" << std::endl;
        for (const std::string &name : required_but_skipped) {
            std::cerr << "ATTENTION: Skipped " << name << std::endl;
        }
        result |= 1;
    }

    if (const size_t problems = LogProblemCounter::getProblemCount(); problems > 0) {
        std::cerr << "ATTENTION: Unexpected NIXL warning(s) and/or error(s) detected!" << std::endl;
        std::cerr << "ATTENTION: Problem count is " << problems << std::endl;
        result |= 1;
    }
    return result;
}
} // namespace gtest

int main(int argc, char **argv) { return gtest::RunTests(argc, argv); }
