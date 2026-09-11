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

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "device/device_allocator.h"

namespace gtest {
namespace device_allocator {

    constexpr size_t kSize = 4096;

    class deviceAllocatorTest : public testing::Test {
    protected:
        nixlDeviceAllocator *allocator_ = nullptr;

        void
        SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
                GTEST_SKIP() << "No CUDA-capable GPU is available.";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            allocator_ = &nixlGetDeviceAllocator();
        }
    };

    TEST_F(deviceAllocatorTest, AllocCopyFree) {
        nixlDeviceAllocator &allocator = *allocator_;

        nixlDeviceMem mem;
        ASSERT_EQ(allocator.allocDeviceMem(kSize, mem), NIXL_SUCCESS);

        std::vector<unsigned char> src(kSize, 0xA5);
        std::vector<unsigned char> dst(kSize, 0);
        ASSERT_EQ(allocator.copyHostToDevice(mem.get(), src.data(), kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.get(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, src);

        ASSERT_EQ(allocator.memsetDeviceMem(mem.get(), 0, kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.get(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0));

        allocator.freeDeviceMem(mem.release());

        nixlMappedHostMem mapped;
        ASSERT_EQ(allocator.allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        std::fill_n(mapped.asHost<unsigned char>(), kSize, 0x5A);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mapped.devPtr(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0x5A));
    }

} // namespace device_allocator
} // namespace gtest
