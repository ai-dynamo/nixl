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

#include "common/configuration.h"
#include "gtest/gtest.h"

#include <limits>
#include <stdlib.h>
#include <string>

namespace {

const std::string variable = "NIXL_CONFIG_TEST";
const std::string undefined = "ASDLFHASLK1298159816";

} // namespace

namespace nixl::config {

TEST(Config, EnvWrapper) {
    const std::string value = "foo";
    ASSERT_EQ(::setenv(variable.c_str(), value.c_str(), 1), 0);
    ASSERT_EQ(getenvOptional(variable), value);
    const std::string fallback = "bar";
    ASSERT_EQ(getenvDefaulted(variable, fallback), value);
    ASSERT_FALSE(getenvOptional(undefined).has_value());
    ASSERT_EQ(getenvDefaulted(undefined, fallback), fallback);
}

TEST(Config, Undefined) {
    ASSERT_EQ(getValueOptional<bool>(undefined), std::nullopt);
    ASSERT_EQ(getValueOptional<char>(undefined), std::nullopt);
    ASSERT_EQ(getValueOptional<std::string>(undefined), std::nullopt);

    ASSERT_EQ(getValueDefaulted<bool>(undefined, true), true);
    ASSERT_EQ(getValueDefaulted<bool>(undefined, false), false);
    ASSERT_EQ(getValueDefaulted<char>(undefined, 42), 42);
    const std::string value = "foo";
    ASSERT_EQ(getValueDefaulted<std::string>(undefined, value), value);

    bool b;
    ASSERT_EQ(getValueWithStatus(b, undefined), NIXL_ERR_NOT_FOUND);
    char c;
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

} // namespace nixl::config
