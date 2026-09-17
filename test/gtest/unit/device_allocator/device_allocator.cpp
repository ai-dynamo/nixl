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

#include <algorithm>
#include <vector>

#include "device/device_allocator.h"
#include "gpu_utils.h"

namespace gtest {
namespace device_allocator {

    using nixl::deviceAllocator;
    using nixl::deviceMem;
    using nixl::mappedHostMem;
    using nixl::getDeviceAllocator;

    constexpr size_t kSize = 4096;

    TEST(deviceMemHost, EmptyHandleOperationsAreSafe) {
        deviceMem mem;
        EXPECT_FALSE(static_cast<bool>(mem));
        EXPECT_EQ(mem.devicePointer(), nullptr);
        EXPECT_EQ(mem.size(), 0u);
        mem.reset();
        EXPECT_EQ(mem.release(), nullptr);

        deviceMem moved(std::move(mem));
        EXPECT_FALSE(static_cast<bool>(moved));
    }

    TEST(mappedHostMemHost, EmptyHandleOperationsAreSafe) {
        mappedHostMem mapped;
        EXPECT_FALSE(static_cast<bool>(mapped));
        EXPECT_EQ(mapped.hostPointer(), nullptr);
        EXPECT_EQ(mapped.devicePointer(), nullptr);
        EXPECT_EQ(mapped.size(), 0u);
        mapped.reset();

        mappedHostMem moved(std::move(mapped));
        EXPECT_FALSE(static_cast<bool>(moved));
    }

    TEST(deviceAllocatorHost, AccessorIsStableAndFreeNullIsSafe) {
        deviceAllocator &allocator = getDeviceAllocator();
        EXPECT_EQ(&allocator, &getDeviceAllocator());
        allocator.freeDeviceMem(nullptr);
    }

    TEST(deviceAllocatorHost, ZeroSizeAllocationsLeaveOutputsEmpty) {
        deviceAllocator &allocator = getDeviceAllocator();
        deviceMem device_mem;
        EXPECT_NE(allocator.allocDeviceMem(0, device_mem), NIXL_SUCCESS);
        EXPECT_FALSE(device_mem);

        mappedHostMem mapped_mem;
        EXPECT_NE(allocator.allocMappedHostMem(0, mapped_mem), NIXL_SUCCESS);
        EXPECT_FALSE(mapped_mem);
    }

    class deviceAllocatorTest : public testing::Test {
    protected:
        deviceAllocator *allocator_ = nullptr;

        void
        SetUp() override {
            int count = 0;
            gpuGetDeviceCount(&count, "Probing GPU devices");
            if (count < 1) {
                GTEST_SKIP() << "No GPU is available.";
            }
            gpuSetDevice(0, "Selecting GPU 0");
            allocator_ = &getDeviceAllocator();
            int device = 0;
            const nixl_status_t status = allocator_->getActiveDevice(device);
            if (status == NIXL_ERR_NOT_SUPPORTED) {
                GTEST_SKIP() << "No device allocator implementation is available.";
            }
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
    };

    TEST_F(deviceAllocatorTest, AllocCopyFree) {
        deviceAllocator &allocator = *allocator_;

        deviceMem mem;
        ASSERT_EQ(allocator.allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *const device_ptr = mem.devicePointer();
        EXPECT_NE(allocator.allocDeviceMem(0, mem), NIXL_SUCCESS);
        EXPECT_EQ(mem.devicePointer(), device_ptr);
        EXPECT_EQ(mem.size(), kSize);

        std::vector<unsigned char> src(kSize, 0xA5);
        std::vector<unsigned char> dst(kSize, 0);
        ASSERT_EQ(allocator.copyHostToDevice(mem.devicePointer(), src.data(), kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.devicePointer(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, src);

        ASSERT_EQ(allocator.memsetDeviceMem(mem.devicePointer(), 0, kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mem.devicePointer(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0));

        allocator.freeDeviceMem(mem.release());

        mappedHostMem mapped;
        ASSERT_EQ(allocator.allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        void *const host_ptr = mapped.hostPointer();
        void *const mapped_device_ptr = mapped.devicePointer();
        EXPECT_NE(allocator.allocMappedHostMem(0, mapped), NIXL_SUCCESS);
        EXPECT_EQ(mapped.hostPointer(), host_ptr);
        EXPECT_EQ(mapped.devicePointer(), mapped_device_ptr);
        EXPECT_EQ(mapped.size(), kSize);
        std::fill_n(mapped.hostPointer<unsigned char>(), kSize, 0x5A);
        ASSERT_EQ(allocator.copyDeviceToHost(dst.data(), mapped.devicePointer(), kSize),
                  NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0x5A));
    }

} // namespace device_allocator
} // namespace gtest
