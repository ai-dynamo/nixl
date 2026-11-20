/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <unistd.h>

extern "C" {
#include <ucs/config/parser.h>
}

#include "common.h"
#include "config.h"

namespace {

using condition_t = nixl::ucx::config::condition_t;

// The UCX config dump, after a newline. UCX comments out the keys that have their default values,
// and dumps the modified UCT keys without the UCX_ prefix.
[[nodiscard]] std::string
getDump(const nixl::ucx::config &config) {
    char *buf = nullptr;
    size_t size = 0;
    FILE *stream = open_memstream(&buf, &size);
    if (stream == nullptr) {
        ADD_FAILURE() << "open_memstream() failed";
        return {};
    }

    ucp_config_print(config.getUcpConfig(),
                     stream,
                     nullptr,
                     static_cast<ucs_config_print_flags_t>(UCS_CONFIG_PRINT_CONFIG |
                                                           UCS_CONFIG_PRINT_COMMENT_DEFAULT));
    fclose(stream);
    const auto dump = "\n" + std::string(buf, size);
    free(buf);
    return dump;
}

// The line of the key in the UCX config dump
[[nodiscard]] std::string
getDumpLine(const nixl::ucx::config &config, std::string_view key) {
    std::istringstream dump(getDump(config));
    const auto var = "UCX_" + std::string(key) + "=";
    for (std::string line; std::getline(dump, line);) {
        if (line.starts_with(var) || line.starts_with("# " + var)) {
            return line;
        }
    }

    ADD_FAILURE() << "No " << var << " in the UCX config";
    return {};
}

[[nodiscard]] std::string
getValue(const nixl::ucx::config &config, std::string_view key) {
    const auto line = getDumpLine(config, key);
    return line.substr(line.find('=') + 1);
}

// MAX_RMA_RAILS after setting it as the UCX context does, with the UCX config read now
[[nodiscard]] std::string
modifyMaxRmaRails() {
    const nixl::ucx::config config;
    config.modify("MAX_RMA_RAILS", "2", condition_t::EQUAL, "1");
    return getValue(config, "MAX_RMA_RAILS");
}

// Add a line of a UCX configuration file. UCX reads its configuration files once per process, so
// do it only in a child process.
void
addConfigFileLine(const std::string &line) {
    const auto dir = std::filesystem::temp_directory_path() /
        ("nixl_ucx_config_test." + std::to_string(getpid()));
    std::filesystem::create_directory(dir);
    std::ofstream(dir / "ucx.conf") << line << "\n";
    ucs_config_parse_config_file(dir.c_str(), "ucx.conf", 1);
    std::filesystem::remove_all(dir);
}

class UcxConfigTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        envHelper_.unsetVar("UCX_MAX_RMA_RAILS");
        envHelper_.unsetVar("UCX_MAX_RMA_LANES");
        envHelper_.unsetVar("UCX_MAX_EAGER_RAILS");
    }

    gtest::ScopedEnv envHelper_;
};

// GoogleTest runs death tests first, before other tests start threads
using UcxConfigDeathTest = UcxConfigTest;

} // namespace

TEST_F(UcxConfigTest, ModifySetsDefaultValue) {
    const nixl::ucx::config config;
    if (getDumpLine(config, "MAX_EAGER_RAILS") != "# UCX_MAX_EAGER_RAILS=1") {
        GTEST_SKIP() << "UCX_MAX_EAGER_RAILS is set in a UCX configuration file";
    }

    config.modify("MAX_EAGER_RAILS", "2");
    EXPECT_EQ(getValue(config, "MAX_EAGER_RAILS"), "2");
}

TEST_F(UcxConfigTest, ModifyKeepsEnvValue) {
    envHelper_.addVar("UCX_MAX_EAGER_RAILS", "3");
    const nixl::ucx::config config;
    config.modify("MAX_EAGER_RAILS", "2");
    EXPECT_EQ(getValue(config, "MAX_EAGER_RAILS"), "3");
}

TEST_F(UcxConfigDeathTest, ModifyKeepsConfigFileValue) {
    // A UCX version that dumps a value from a configuration file as a default one does not tell
    // it from a default value, so then the value is modified
    EXPECT_EXIT(
        {
            addConfigFileLine("UCX_MAX_EAGER_RAILS=3");
            const nixl::ucx::config config;
            const bool reported =
                (getDumpLine(config, "MAX_EAGER_RAILS") == "UCX_MAX_EAGER_RAILS=3");
            config.modify("MAX_EAGER_RAILS", "2");
            const auto value = getValue(config, "MAX_EAGER_RAILS");
            std::cerr << "reported " << reported << ", MAX_EAGER_RAILS=" << value << std::endl;
            std::_Exit((value == (reported ? "3" : "2")) ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST_F(UcxConfigTest, ModifySetsUctValue) {
    envHelper_.unsetVar("UCX_SELF_SEG_SIZE");
    const nixl::ucx::config config;
    config.modify("SELF_SEG_SIZE", "16k");
    EXPECT_NE(getDump(config).find("\nSELF_SEG_SIZE=16k\n"), std::string::npos);
}

TEST_F(UcxConfigTest, ModifyKeepsUctEnvValue) {
    envHelper_.addVar("UCX_SELF_SEG_SIZE", "16k");
    const nixl::ucx::config config;
    config.modify("SELF_SEG_SIZE", "32k");
    EXPECT_EQ(getDump(config).find("\nSELF_SEG_SIZE="), std::string::npos);
}

TEST_F(UcxConfigTest, ModifyNotInEnvKeepsEnvValue) {
    envHelper_.addVar("UCX_MAX_EAGER_RAILS", "3");
    const nixl::ucx::config config;
    config.modify("MAX_EAGER_RAILS", "2", condition_t::NOT_IN_ENV);
    EXPECT_EQ(getValue(config, "MAX_EAGER_RAILS"), "3");
}

TEST_F(UcxConfigDeathTest, ModifyNotInEnvOverridesConfigFileValue) {
    // As the engine config of the UCX backend does
    EXPECT_EXIT(
        {
            addConfigFileLine("UCX_MAX_EAGER_RAILS=3");
            const nixl::ucx::config config;
            config.modify("MAX_EAGER_RAILS", "2", condition_t::NOT_IN_ENV);
            std::cerr << "MAX_EAGER_RAILS=" << getValue(config, "MAX_EAGER_RAILS") << std::endl;
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0),
        "MAX_EAGER_RAILS=2");
}

TEST_F(UcxConfigTest, ModifyAlwaysOverridesEnvValue) {
    envHelper_.addVar("UCX_MAX_EAGER_RAILS", "3");
    const nixl::ucx::config config;
    config.modify("MAX_EAGER_RAILS", "2", condition_t::ALWAYS);
    EXPECT_EQ(getValue(config, "MAX_EAGER_RAILS"), "2");
}

TEST_F(UcxConfigTest, ModifyEqualSetsDefaultValue) {
    if (getDumpLine(nixl::ucx::config(), "MAX_RMA_RAILS") != "# UCX_MAX_RMA_RAILS=1") {
        GTEST_SKIP() << "UCX_MAX_RMA_RAILS is set in a UCX configuration file";
    }

    EXPECT_EQ(modifyMaxRmaRails(), "2");
}

TEST_F(UcxConfigTest, ModifyEqualKeepsEnvValue) {
    for (const std::string value : {"1", "4"}) {
        envHelper_.addVar("UCX_MAX_RMA_RAILS", value);
        EXPECT_EQ(modifyMaxRmaRails(), value);
        envHelper_.popVar();
    }
}

TEST_F(UcxConfigTest, ModifyEqualKeepsAliasValue) {
    envHelper_.addVar("UCX_MAX_RMA_LANES", "4");
    EXPECT_EQ(modifyMaxRmaRails(), "4");
}

TEST_F(UcxConfigDeathTest, ModifyEqualKeepsConfigFileValue) {
    // As a platform section of ucx.conf sets 4 rails
    EXPECT_EXIT(
        {
            addConfigFileLine("UCX_MAX_RMA_RAILS=4");
            std::cerr << "MAX_RMA_RAILS=" << modifyMaxRmaRails() << std::endl;
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0),
        "MAX_RMA_RAILS=4");
}
