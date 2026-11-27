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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
using ucp_config_ptr = std::unique_ptr<ucp_config_t, decltype(&ucp_config_release)>;

// UCX reads NIXL_UCX_<key> first, then UCX_<key>
constexpr std::array<std::string_view, 2> ucx_env_var_prefixes = {"NIXL_UCX_", "UCX_"};

[[nodiscard]] ucp_config_ptr
readConfig(const char *env_prefix) {
    ucp_config_t *config = nullptr;
    EXPECT_EQ(ucp_config_read(env_prefix, nullptr, &config), UCS_OK);
    return {config, &ucp_config_release};
}

// ucp_config_t is opaque, so read the effective value back from its printout.
// ucp_config_print() names the fields UCX_<key> whatever the read prefix was.
[[nodiscard]] std::string
getConfigValue(const ucp_config_t *config, const std::string &key) {
    char *buf = nullptr;
    size_t size = 0;
    FILE *stream = open_memstream(&buf, &size);
    if (stream == nullptr) {
        ADD_FAILURE() << "open_memstream() failed";
        return {};
    }

    ucp_config_print(config, stream, nullptr, UCS_CONFIG_PRINT_CONFIG);
    fclose(stream);
    const std::unique_ptr<char, decltype(&free)> buf_holder(buf, &free);

    std::istringstream lines(std::string(buf, size));
    const std::string field = "UCX_" + key + "=";
    for (std::string line; std::getline(lines, line);) {
        if (line.rfind(field, 0) == 0) {
            return line.substr(field.size());
        }
    }

    ADD_FAILURE() << "UCX config field " << key << " not found";
    return {};
}

// UCX normalizes values when printing them, so compare against how UCX prints
// the same value instead of hardcoding its format.
[[nodiscard]] std::string
getPrintedValue(const std::string &key, const std::string &value) {
    const auto config = readConfig(nullptr);
    EXPECT_EQ(ucp_config_modify(config.get(), key.c_str(), value.c_str()), UCS_OK);
    return getConfigValue(config.get(), key);
}

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
    config.modifyMaxRmaRails(2);
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

} // namespace

class UcxConfigTest : public ::testing::Test {
protected:
    gtest::ScopedEnv env_;

    void
    SetUp() override {
        for (const auto *key : {"TLS",
                                "MAX_RMA_RAILS",
                                "MAX_RMA_LANES",
                                "MAX_EAGER_RAILS",
                                "RNDV_THRESH",
                                "SELF_SEG_SIZE"}) {
            env_.unsetVar(std::string("UCX_") + key);
            env_.unsetVar(std::string("NIXL_UCX_") + key);
        }
    }
};

// GoogleTest runs death tests first, before other tests start threads
using UcxConfigDeathTest = UcxConfigTest;

TEST_F(UcxConfigTest, PrefixedVarOverridesGlobalVar) {
    env_.addVar("UCX_MAX_RMA_RAILS", "3");
    env_.addVar("NIXL_UCX_MAX_RMA_RAILS", "4");

    const nixl::ucx::config config;
    EXPECT_EQ(getConfigValue(config.getUcpConfig(), "MAX_RMA_RAILS"),
              getPrintedValue("MAX_RMA_RAILS", "4"));
}

TEST_F(UcxConfigTest, GlobalVarAppliesWithoutPrefixedVar) {
    env_.addVar("UCX_MAX_RMA_RAILS", "3");

    const nixl::ucx::config config;
    EXPECT_EQ(getConfigValue(config.getUcpConfig(), "MAX_RMA_RAILS"),
              getPrintedValue("MAX_RMA_RAILS", "3"));
}

TEST_F(UcxConfigTest, DefaultAppliesWithoutEnvVar) {
    const nixl::ucx::config config;
    if (!getDumpLine(config, "RNDV_THRESH").starts_with("# ")) {
        GTEST_SKIP() << "UCX_RNDV_THRESH is set in a UCX configuration file";
    }

    config.modify("RNDV_THRESH", "inf");
    EXPECT_EQ(getConfigValue(config.getUcpConfig(), "RNDV_THRESH"),
              getPrintedValue("RNDV_THRESH", "inf"));
}

TEST_F(UcxConfigTest, DefaultDoesNotOverrideEnvVar) {
    for (const auto *var : {"NIXL_UCX_RNDV_THRESH", "UCX_RNDV_THRESH"}) {
        SCOPED_TRACE(var);
        gtest::ScopedEnv env;
        env.addVar(var, "8k");

        const nixl::ucx::config config;
        config.modify("RNDV_THRESH", "inf");
        EXPECT_EQ(getConfigValue(config.getUcpConfig(), "RNDV_THRESH"),
                  getPrintedValue("RNDV_THRESH", "8k"));
    }
}

TEST_F(UcxConfigTest, PrefixedVarDoesNotAffectOtherContexts) {
    const auto default_tls = getConfigValue(readConfig("MPI").get(), "TLS");
    env_.addVar("NIXL_UCX_TLS", "tcp");

    const nixl::ucx::config config;
    EXPECT_EQ(getConfigValue(config.getUcpConfig(), "TLS"), getPrintedValue("TLS", "tcp"));
    EXPECT_EQ(getConfigValue(readConfig("MPI").get(), "TLS"), default_tls);
    EXPECT_EQ(getConfigValue(readConfig(nullptr).get(), "TLS"), default_tls);
}

TEST_F(UcxConfigDeathTest, ModifyKeepsConfigFileValue) {
    // A UCX version that dumps a value from a configuration file as a default one does not tell
    // it from a default value, so then the value is modified
    EXPECT_EXIT(
        {
            addConfigFileLine("UCX_MAX_EAGER_RAILS=3");
            const nixl::ucx::config config;
            if (getValue(config, "MAX_EAGER_RAILS") != "3") {
                std::cerr << "The UCX configuration file is not applied" << std::endl;
                std::_Exit(1);
            }

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
    const nixl::ucx::config config;
    config.modify("SELF_SEG_SIZE", "16k");
    EXPECT_NE(getDump(config).find("\nSELF_SEG_SIZE=16k\n"), std::string::npos);
}

TEST_F(UcxConfigTest, ModifyKeepsUctEnvValue) {
    for (const auto prefix : ucx_env_var_prefixes) {
        const auto var = std::string(prefix) + "SELF_SEG_SIZE";
        SCOPED_TRACE(var);
        gtest::ScopedEnv env;
        env.addVar(var, "16k");

        const nixl::ucx::config config;
        config.modify("SELF_SEG_SIZE", "32k");
        EXPECT_EQ(getDump(config).find("\nSELF_SEG_SIZE="), std::string::npos);
    }
}

TEST_F(UcxConfigTest, ModifyAlwaysOverridesEnvValue) {
    for (const auto prefix : ucx_env_var_prefixes) {
        const auto var = std::string(prefix) + "MAX_EAGER_RAILS";
        SCOPED_TRACE(var);
        gtest::ScopedEnv env;
        env.addVar(var, "3");

        const nixl::ucx::config config;
        config.modify("MAX_EAGER_RAILS", "2", condition_t::ALWAYS);
        EXPECT_EQ(getValue(config, "MAX_EAGER_RAILS"), "2");
    }
}

TEST_F(UcxConfigTest, ModifyMaxRmaRailsSetsDefaultValue) {
    // Since openucx/ucx#12325 the UCX default is 1 rail, except for the GPU buffers whose NICs
    // UCX assigns: these buffers get every lane of their NICs
    const auto line = getDumpLine(nixl::ucx::config(), "MAX_RMA_RAILS");
    std::string rails;
    if (line == "# UCX_MAX_RMA_RAILS=1") {
        rails = "2";
    } else if (line == "# UCX_MAX_RMA_RAILS=default:1,assigned:auto") {
        rails = "2,assigned:auto";
    } else {
        GTEST_SKIP() << "UCX_MAX_RMA_RAILS is set in a UCX configuration file: " << line;
    }

    EXPECT_EQ(modifyMaxRmaRails(), getPrintedValue("MAX_RMA_RAILS", rails));
}

TEST_F(UcxConfigTest, ModifyMaxRmaRailsKeepsEnvValue) {
    for (const auto prefix : ucx_env_var_prefixes) {
        for (const std::string value : {"1", "4"}) {
            const auto var = std::string(prefix) + "MAX_RMA_RAILS";
            SCOPED_TRACE(var + "=" + value);
            gtest::ScopedEnv env;
            env.addVar(var, value);
            EXPECT_EQ(modifyMaxRmaRails(), getPrintedValue("MAX_RMA_RAILS", value));
        }
    }
}

TEST_F(UcxConfigTest, ModifyMaxRmaRailsKeepsAliasValue) {
    env_.addVar("UCX_MAX_RMA_LANES", "4");
    EXPECT_EQ(modifyMaxRmaRails(), getPrintedValue("MAX_RMA_RAILS", "4"));
}

TEST_F(UcxConfigDeathTest, ModifyMaxRmaRailsKeepsConfigFileValue) {
    // As a platform section of ucx.conf sets 4 rails
    EXPECT_EXIT(
        {
            addConfigFileLine("UCX_MAX_RMA_RAILS=4");
            const auto rails = getPrintedValue("MAX_RMA_RAILS", "4");
            if (getValue(nixl::ucx::config(), "MAX_RMA_RAILS") != rails) {
                std::cerr << "The UCX configuration file is not applied" << std::endl;
                std::_Exit(1);
            }

            const auto value = modifyMaxRmaRails();
            std::cerr << "MAX_RMA_RAILS=" << value << ", expected " << rails << std::endl;
            std::_Exit((value == rails) ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}
