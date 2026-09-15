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

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "gds_mt_backend.h"

namespace {

constexpr size_t registered_size = 4096;
constexpr size_t interior_offset = 1024;
constexpr size_t interior_size = 512;

TEST(GdsMtOffsetTest, InteriorDescriptorUsesRegisteredBase) {
    alignas(4096) char registered[registered_size]{};
    const uintptr_t descriptor_addr = reinterpret_cast<uintptr_t>(registered) + interior_offset;

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(
        registered, sizeof(registered), descriptor_addr, interior_size, resolved);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_TRUE(resolved.devPtrBase == static_cast<void *>(registered));
    EXPECT_EQ(resolved.devPtrOffset, interior_offset);
}

TEST(GdsMtOffsetTest, ExactBaseUsesZeroOffset) {
    alignas(4096) char registered[registered_size]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status =
        gdsMtResolveRegisteredBuffer(registered,
                                     sizeof(registered),
                                     reinterpret_cast<uintptr_t>(registered),
                                     sizeof(registered),
                                     resolved);

    EXPECT_EQ(status, NIXL_SUCCESS);
    EXPECT_TRUE(resolved.devPtrBase == static_cast<void *>(registered));
    EXPECT_EQ(resolved.devPtrOffset, static_cast<size_t>(0));
}

TEST(GdsMtOffsetTest, DescriptorBeforeRegisteredBaseIsRejected) {
    alignas(4096) char registered[registered_size]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status = gdsMtResolveRegisteredBuffer(
        registered, sizeof(registered), reinterpret_cast<uintptr_t>(registered) - 1, 1, resolved);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

TEST(GdsMtOffsetTest, DescriptorPastRegisteredEndIsRejected) {
    alignas(4096) char registered[registered_size]{};

    gdsMtResolvedBuffer resolved{};
    const nixl_status_t status =
        gdsMtResolveRegisteredBuffer(registered,
                                     sizeof(registered),
                                     reinterpret_cast<uintptr_t>(registered) + 3584,
                                     1024,
                                     resolved);

    EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
}

} // namespace
