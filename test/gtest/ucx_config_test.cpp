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
#include <memory>
#include <sstream>
#include <string>

#include "common.h"
#include "config.h"

namespace {

using ucp_config_ptr = std::unique_ptr<ucp_config_t, decltype(&ucp_config_release)>;

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

} // namespace

class UcxConfigTest : public ::testing::Test {
protected:
    gtest::ScopedEnv env_;

    void
    SetUp() override {
        for (const auto *key : {"TLS", "MAX_RMA_RAILS", "RNDV_THRESH"}) {
            env_.unsetVar(std::string("UCX_") + key);
            env_.unsetVar(std::string("NIXL_UCX_") + key);
        }
    }
};

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
