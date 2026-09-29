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

#include "common/backend.h"
#include "common/configuration.h"
#include "gtest/gtest.h"
#include "common.h"
#include "nixl.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdlib.h>
#include <string>
#include <vector>

namespace {

const std::string variable = "NIXL_CONFIG_TEST";
const std::string undefined = "ASDLFHASLK1298159816";

} // namespace

namespace nixl::config {

TEST(Config, EnvWrapper) {
    const std::string value = "foo";
    ASSERT_EQ(::setenv(variable.c_str(), value.c_str(), 1), 0);
    ASSERT_EQ(internal::getenvOptional(variable), value);
    const std::string fallback = "bar";
    ASSERT_EQ(internal::getenvDefaulted(variable, fallback), value);
    ASSERT_FALSE(internal::getenvOptional(undefined).has_value());
    ASSERT_EQ(internal::getenvDefaulted(undefined, fallback), fallback);
}

TEST(Config, Undefined) {
    ASSERT_EQ(getValueOptional<bool>(undefined), std::nullopt);
    ASSERT_EQ(getValueOptional<short>(undefined), std::nullopt);
    ASSERT_EQ(getValueOptional<std::string>(undefined), std::nullopt);

    ASSERT_EQ(getValueDefaulted<bool>(undefined, true), true);
    ASSERT_EQ(getValueDefaulted<bool>(undefined, false), false);
    ASSERT_EQ(getValueDefaulted<short>(undefined, 42), 42);
    const std::string value = "foo";
    ASSERT_EQ(getValueDefaulted<std::string>(undefined, value), value);

    bool b;
    ASSERT_EQ(getValueWithStatus(b, undefined), NIXL_ERR_NOT_FOUND);
    short c;
    ASSERT_EQ(getValueWithStatus(c, undefined), NIXL_ERR_NOT_FOUND);
    std::string s;
    ASSERT_EQ(getValueWithStatus(s, undefined), NIXL_ERR_NOT_FOUND);
}

namespace {

    template<typename T>
    void
    testSimpleSuccess(const std::string &input, const T value) {
        ASSERT_EQ(::setenv(variable.c_str(), input.c_str(), 1), 0);
        EXPECT_EQ(getValue<T>(variable), value);
        EXPECT_EQ(getValueOptional<T>(variable), value);
        EXPECT_EQ(getValueDefaulted<T>(variable, !value), value);
        EXPECT_EQ(getValue<std::string>(variable), input);
        EXPECT_EQ(getValueOptional<std::string>(variable), input);
        EXPECT_EQ(getValueDefaulted<std::string>(variable, variable), input);
        T out;
        EXPECT_EQ(getValueWithStatus(out, variable), NIXL_SUCCESS);
        EXPECT_EQ(out, value);
        std::string str;
        EXPECT_EQ(getValueWithStatus(str, variable), NIXL_SUCCESS);
        EXPECT_EQ(str, input);
    }

    template<typename T>
    void
    testSimpleFailure(const std::string &input) {
        ASSERT_EQ(::setenv(variable.c_str(), input.c_str(), 1), 0);
        EXPECT_ANY_THROW((void)getValue<T>(variable));
        EXPECT_ANY_THROW((void)getValueOptional<T>(variable));
        EXPECT_ANY_THROW((void)getValueDefaulted<T>(variable, T()));
        EXPECT_EQ(getValue<std::string>(variable), input);
        EXPECT_EQ(getValueOptional<std::string>(variable), input);
        EXPECT_EQ(getValueDefaulted<std::string>(variable, variable), input);
        T out;
        EXPECT_EQ(getValueWithStatus(out, variable), NIXL_ERR_MISMATCH);
    }

} // namespace

TEST(Config, ConvertBool) {
    testSimpleSuccess("1", true);
    testSimpleSuccess("0", false);
    testSimpleSuccess("yes", true);
    testSimpleSuccess("no", false);
    testSimpleSuccess("Yes", true);
    testSimpleSuccess("No", false);
    testSimpleSuccess("YeS", true);
    testSimpleSuccess("nO", false);
    testSimpleSuccess("enable", true);
    testSimpleSuccess("disable", false);
    testSimpleSuccess("on", true);
    testSimpleSuccess("off", false);
    testSimpleSuccess("oN", true);
    testSimpleSuccess("oFF", false);
    testSimpleSuccess("TRUE", true);
    testSimpleSuccess("FALSE", false);
    testSimpleSuccess("true", true);
    testSimpleSuccess("false", false);

    testSimpleFailure<bool>("");
    testSimpleFailure<bool>("2");
    testSimpleFailure<bool>("enabled");
}

namespace {
    template<typename T>
    void
    testSigned() {
        testSimpleSuccess("0", T(0));
        testSimpleSuccess("1", T(1));
        testSimpleSuccess("-1", T(-1));
        testSimpleSuccess("42", T(42));
        testSimpleSuccess("-42", T(-42));
        const T min_value = std::numeric_limits<T>::min();
        const std::string min_string = std::to_string(min_value);
        testSimpleSuccess(min_string, min_value);
        const T max_value = std::numeric_limits<T>::max();
        const std::string max_string = std::to_string(max_value);
        testSimpleSuccess(max_string, max_value);

        testSimpleFailure<T>("");
        testSimpleFailure<T>("-");
        testSimpleFailure<T>("+");
        testSimpleFailure<T>("+0");
        testSimpleFailure<T>("+1");
        testSimpleFailure<T>("r");
        testSimpleFailure<T>("0y");
        testSimpleFailure<T>("0x");
        testSimpleFailure<T>(max_string + '0');
        testSimpleFailure<T>(min_string + '0');

        testSimpleFailure<T>("0x0");
        testSimpleFailure<T>("0x000");
        testSimpleFailure<T>("0x01");
        testSimpleFailure<T>("0x1f");
        testSimpleFailure<T>("-0x01");
    }

    template<typename T>
    void
    testUnsigned() {
        testSimpleSuccess("0", T(0));
        testSimpleSuccess("1", T(1));
        testSimpleSuccess("42", T(42));
        const T max_value = std::numeric_limits<T>::max();
        const std::string max_string = std::to_string(max_value);
        testSimpleSuccess(max_string, max_value);

        testSimpleFailure<T>("");
        testSimpleFailure<T>(" 0");
        testSimpleFailure<T>("0 ");
        testSimpleFailure<T>("-");
        testSimpleFailure<T>("+");
        testSimpleFailure<T>("-0");
        testSimpleFailure<T>("-1");
        testSimpleFailure<T>("-42");
        testSimpleFailure<T>("+1");
        testSimpleFailure<T>("+0");
        testSimpleFailure<T>("+42");
        testSimpleFailure<T>("-m");
        testSimpleFailure<T>("r");
        testSimpleFailure<T>("0y");
        testSimpleFailure<T>("0x");
        testSimpleFailure<T>(max_string + '0');

        testSimpleSuccess<T>("0x0", T(0));
        testSimpleSuccess<T>("0x000", T(0));
        testSimpleSuccess<T>("0x01", T(1));

        testSimpleSuccess<T>("0X0", T(0));
        testSimpleSuccess<T>("0X000", T(0));
        testSimpleSuccess<T>("0X01", T(1));

        testSimpleSuccess<T>("0x1f", T(31));
        testSimpleSuccess<T>("0X1f", T(31));
        testSimpleSuccess<T>("0X1F", T(31));
        testSimpleSuccess<T>("0x1F", T(31));

        testSimpleFailure<T>("+0x00");
        testSimpleFailure<T>("1x00");
    }

} // namespace

TEST(Config, ConvertSigned) {
    testSigned<std::int8_t>();
    testSigned<std::int16_t>();
    testSigned<std::int32_t>();
    testSigned<std::int64_t>();
}

TEST(Config, ConvertUnsigned) {
    testUnsigned<std::uint8_t>();
    testUnsigned<std::uint16_t>();
    testUnsigned<std::uint32_t>();
    testUnsigned<std::uint64_t>();
}

TEST(Config, BackendBasics) {
    const std::string negative = "negative";
    const std::string positive = "positive";
    const std::string string = "string";
    const std::string boolean = "boolean";
    const std::string value = "transmogrify";
    const std::string unknown = "unknownkey";

    const nixl_b_params_t p = {
        {negative, "-42"}, {positive, "129"}, {string, value}, {boolean, "no"}};

    // Test Optional
    {
        const auto r = nixl::getBackendParamOptional<int>(p, negative);
        EXPECT_TRUE(r.has_value());
        EXPECT_EQ(*r, -42);
    }
    {
        const auto r = nixl::getBackendParamOptional<unsigned>(&p, positive);
        EXPECT_TRUE(r.has_value());
        EXPECT_EQ(*r, 129);
    }
    {
        const auto r = nixl::getBackendParamOptional<bool>(&p, unknown);
        EXPECT_FALSE(r.has_value());
    }
    { EXPECT_THROW((void)nixl::getBackendParamOptional<int>(&p, string), std::runtime_error); }
    {
        const auto r = nixl::getBackendParamOptional<bool>(nullptr, boolean);
        EXPECT_FALSE(r.has_value());
    }
    // Test Defaulted
    {
        const bool r = nixl::getBackendParamDefaulted(p, boolean, true);
        EXPECT_FALSE(r);
    }
    {
        const bool r = nixl::getBackendParamDefaulted(&p, unknown, true);
        EXPECT_TRUE(r);
    }
    {
        const std::string r = nixl::getBackendParamDefaulted<std::string>(p, string, "wrong");
        EXPECT_EQ(r, value);
    }
    { EXPECT_THROW((void)nixl::getBackendParamDefaulted(&p, boolean, 0), std::runtime_error); }
    {
        const int v = 12345;
        const unsigned r = nixl::getBackendParamDefaulted(nullptr, value, v);
        EXPECT_EQ(r, v);
    }
}

namespace {

    const std::string pid = std::to_string(::getpid());
    const std::string bool_name = "bool" + pid;
    const std::string number_name = "number" + pid;
    const std::string string_name = "string" + pid;
    const std::string env1name = "env" + pid + "var1";
    const std::string env1value = "value" + pid;
    const std::string env2name = "env" + pid + "var2";
    const std::string env2value = "not_an_int";

} // namespace

namespace internal {

    // This function is safe to be called because no other thread is accessing the config.

    void
    refreshConfigFileForUnitTest();

} // namespace internal

TEST(Config, ReadConfigFile) {
    const auto pid = ::getpid();
    const auto file = "nixl_test_" + std::to_string(pid) + ".cfg";
    const auto path = std::filesystem::temp_directory_path() / file;
    {
        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        ofs << bool_name << " = true\n";
        ofs << number_name << " = 42\n";
        ofs << string_name << " = \"hello\"\n";
        ofs << env1name << " = \"dummy\"\n";
        ofs << env2name << " = 0\n";
    }
    gtest::ScopedEnv vars;
    vars.addVar("NIXL_CONFIG_FILE", path.native());
    vars.addVar(env1name, env1value);
    vars.addVar(env2name, env2value);
    internal::refreshConfigFileForUnitTest();
    {
        const auto value = nixl::config::getValue<bool>(bool_name);
        EXPECT_TRUE(value);
    }
    {
        const auto value = nixl::config::getValue<unsigned>(number_name);
        EXPECT_EQ(value, 42);
    }
    {
        const auto value = nixl::config::getValue<std::string>(string_name);
        EXPECT_EQ(value, "hello");
    }
    {
        // Check that environment wins against config file.
        const auto value = nixl::config::getValue<std::string>(env1name);
        EXPECT_EQ(value, env1value);
    }
    {
        // Check that conversion failure on env var does not trigger lookup in config file.
        EXPECT_THROW((void)nixl::config::getValue<int>(env2name), std::runtime_error);
    }
}

namespace {

    const std::string ucx_vram_memtype_hint_key = "ucx_vram_memtype_hint";

    // Creates a UCX backend with the given VRAM memtype hint in a fresh agent, since an agent holds
    // at most one backend per type. Returns std::nullopt when the UCX plugin is not built.
    [[nodiscard]] std::optional<nixl_status_t>
    createUcxBackendWithHint(const std::string &hint) {
        gtest::ScopedEnv env;
        env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");

        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        nixlAgent agent("ucx_vram_memtype_hint_" + hint, cfg);

        std::vector<nixl_backend_t> plugins;
        EXPECT_EQ(agent.getAvailPlugins(plugins), NIXL_SUCCESS);
        if (std::find(plugins.begin(), plugins.end(), "UCX") == plugins.end()) {
            return std::nullopt;
        }

        nixl_mem_list_t mems;
        nixl_b_params_t params;
        EXPECT_EQ(agent.getPluginParams("UCX", mems, params), NIXL_SUCCESS);
        EXPECT_EQ(params[ucx_vram_memtype_hint_key], "auto");
        params[ucx_vram_memtype_hint_key] = hint;

        nixlBackendH *backend = nullptr;
        const auto status = agent.createBackend("UCX", params, backend);
        EXPECT_EQ(status == NIXL_SUCCESS, backend != nullptr) << "hint '" << hint << "'";
        return status;
    }

} // namespace

TEST(Config, UcxVramMemtypeHint) {
    // auto and none never require a memory type from the UCX context.
    for (const std::string hint : {"auto", "none"}) {
        const auto status = createUcxBackendWithHint(hint);
        if (!status) {
            GTEST_SKIP() << "UCX plugin not available";
        }
        EXPECT_EQ(*status, NIXL_SUCCESS) << "hint '" << hint << "'";
    }

    // Matching is case-sensitive.
    const gtest::LogIgnoreGuard lig_invalid(
        "Failed to create engine: Invalid VRAM memtype hint mode: .*");
    const gtest::LogIgnoreGuard lig_backend(
        "backend (creation failed|initialization error) for 'UCX'");
    const auto status = createUcxBackendWithHint("CUDA");
    ASSERT_TRUE(status.has_value());
    EXPECT_NE(*status, NIXL_SUCCESS);
    EXPECT_EQ(lig_invalid.getIgnoredCount(), 1u);
}

TEST(Config, UcxVramMemtypeHintUnsupportedByContext) {
    const gtest::LogIgnoreGuard lig_unsupported(
        "Failed to create engine: Configured VRAM memtype hint '.*' is not supported by "
        "current UCX context");
    const gtest::LogIgnoreGuard lig_backend(
        "backend (creation failed|initialization error) for 'UCX'");

    // An explicit hint either succeeds or is rejected as unsupported by the UCX context.
    size_t rejected = 0;
    for (const std::string hint : {"cuda", "cuda-managed", "rocm", "ze-device"}) {
        const auto status = createUcxBackendWithHint(hint);
        if (!status) {
            GTEST_SKIP() << "UCX plugin not available";
        }
        if (*status != NIXL_SUCCESS) {
            ++rejected;
        }
    }

    EXPECT_EQ(lig_unsupported.getIgnoredCount(), rejected);
    if (rejected == 0) {
        GTEST_SKIP() << "UCX context supports every explicit hint, no rejection exercised";
    }
}

} // namespace nixl::config
