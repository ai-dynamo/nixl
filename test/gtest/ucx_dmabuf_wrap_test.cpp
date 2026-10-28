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

#include "ucx_utils.h"

namespace {

constexpr uintptr_t GiB = uintptr_t{1} << 30;

TEST(UcxDmabufWrapTest, PeriodMatchesExactAlignment) {
    // Region bases aligned to 2^31 or less never match, whatever the length.
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(2 * GiB, 12 * GiB), size_t{0});
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(6 * GiB, 15 * GiB), size_t{0});
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(0, 15 * GiB), size_t{0});

    // Base aligned to exactly 2^32: only regions longer than 4 GiB match.
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(4 * GiB, 4 * GiB), size_t{0});
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(4 * GiB, 4 * GiB + 1), 4 * GiB);
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(12 * GiB, 15 * GiB), 4 * GiB);

    // Base aligned to exactly 2^33 or 2^34: the period follows the alignment.
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(8 * GiB, 8 * GiB), size_t{0});
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(8 * GiB, 12 * GiB), 8 * GiB);
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(16 * GiB, 15 * GiB), size_t{0});
    EXPECT_EQ(nixl::ucx::dmabufWrapPeriod(16 * GiB, 17 * GiB), 16 * GiB);
}

TEST(UcxDmabufWrapTest, ConfigOffValues) {
    for (const char *value : {"n", "no", "NO", "off", "Off", "0", "false"}) {
        EXPECT_TRUE(nixl::ucx::isUcxConfigOff(value)) << value;
    }
    for (const char *value : {"", "y", "yes", "try", "auto", "on", "1"}) {
        EXPECT_FALSE(nixl::ucx::isUcxConfigOff(value)) << value;
    }
}

} // namespace
