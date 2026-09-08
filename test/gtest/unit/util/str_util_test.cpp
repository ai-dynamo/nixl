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

#include <set>
#include <string>
#include <vector>

#include "common/str_util.h"

namespace {

using nixl::str::splitStripped;
using nixl::str::splitStrippedSet;

TEST(StrSplitStripped, SplitsAndTrimsTokens) {
    EXPECT_EQ(splitStripped("nvtx,chakra"), (std::vector<std::string>{"nvtx", "chakra"}));
    EXPECT_EQ(splitStripped(" nvtx , chakra "), (std::vector<std::string>{"nvtx", "chakra"}));
    EXPECT_EQ(splitStripped("\tnvtx\t"), (std::vector<std::string>{"nvtx"}));
}

TEST(StrSplitStripped, PreservesOrderAndDuplicates) {
    EXPECT_EQ(splitStripped("chakra,nvtx"), (std::vector<std::string>{"chakra", "nvtx"}));
    EXPECT_EQ(splitStripped("nvtx,nvtx"), (std::vector<std::string>{"nvtx", "nvtx"}));
}

TEST(StrSplitStripped, DropsEmptyAndWhitespaceOnlyTokens) {
    EXPECT_TRUE(splitStripped("").empty());
    EXPECT_TRUE(splitStripped("   ").empty());
    EXPECT_TRUE(splitStripped(",").empty());
    EXPECT_TRUE(splitStripped(" , ").empty());
    EXPECT_EQ(splitStripped(",nvtx,,chakra,"), (std::vector<std::string>{"nvtx", "chakra"}));
}

TEST(StrSplitStripped, HonorsCustomDelimiter) {
    EXPECT_EQ(splitStripped("a:b: c ", ':'), (std::vector<std::string>{"a", "b", "c"}));
    EXPECT_EQ(splitStripped("a,b", ':'), (std::vector<std::string>{"a,b"}));
}

TEST(StrSplitStrippedSet, DeduplicatesAndSorts) {
    EXPECT_EQ(splitStrippedSet("nvtx,chakra"), (std::set<std::string>{"chakra", "nvtx"}));
    EXPECT_EQ(splitStrippedSet(" nvtx , nvtx , chakra "),
              (std::set<std::string>{"chakra", "nvtx"}));
    EXPECT_EQ(splitStrippedSet("chakra,chakra,chakra"), (std::set<std::string>{"chakra"}));
}

TEST(StrSplitStrippedSet, DropsEmptyAndWhitespaceOnlyTokens) {
    EXPECT_TRUE(splitStrippedSet("").empty());
    EXPECT_TRUE(splitStrippedSet(" , ").empty());
    EXPECT_EQ(splitStrippedSet(",nvtx,,nvtx,"), (std::set<std::string>{"nvtx"}));
}

TEST(StrSplitStrippedSet, HonorsCustomDelimiter) {
    EXPECT_EQ(splitStrippedSet("a:b: a ", ':'), (std::set<std::string>{"a", "b"}));
}

} // namespace
