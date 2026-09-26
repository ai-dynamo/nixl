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

#include "device/device_ops.h"
#include "gpu_utils.h"

namespace gtest {
namespace device_ops {

    using nixl::deviceOps;
    using nixl::deviceMem;
    using nixl::mappedHostMem;
    using nixl::getDeviceOps;
    using copyDirection = deviceOps::copyDirection;

    constexpr size_t kSize = 4096;

    TEST(deviceMemHost, EmptyHandleOperationsAreSafe) {
        deviceMem mem;
        EXPECT_FALSE(mem);
        EXPECT_EQ(mem.get(), nullptr);
        mem.reset();
        EXPECT_EQ(mem.release(), nullptr);

        deviceMem moved(std::move(mem));
        EXPECT_FALSE(moved);
    }

    TEST(mappedHostMemHost, EmptyHandleOperationsAreSafe) {
        mappedHostMem mapped;
        EXPECT_FALSE(mapped);
        EXPECT_EQ(mapped.hostPointer(), nullptr);
        EXPECT_EQ(mapped.devicePointer(), nullptr);
        mapped.reset();

        mappedHostMem moved(std::move(mapped));
        EXPECT_FALSE(moved);
    }

    TEST(deviceOpsHost, AccessorIsStableAndNullHandleIsSafe) {
        deviceOps *ops = getDeviceOps();
        EXPECT_EQ(ops, getDeviceOps());
        deviceMem empty(nullptr, {ops});
        EXPECT_FALSE(empty);
        EXPECT_EQ(empty.get(), nullptr);
    }

    TEST(deviceOpsHost, ZeroSizeOperationsLeaveOutputsUnchanged) {
        auto *impl = getDeviceOps();
        if (impl == nullptr) {
            GTEST_SKIP() << "No device operations implementation is available.";
        }
        deviceOps &ops = *impl;
        deviceMem device_mem;
        EXPECT_EQ(ops.allocDeviceMem(0, device_mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(device_mem);

        mappedHostMem mapped_mem;
        EXPECT_EQ(ops.allocMappedHostMem(0, mapped_mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(mapped_mem);

        unsigned char byte = 0xA5;
        void *pointers[] = {nullptr, &byte};
        for (void *ptr : pointers) {
            EXPECT_EQ(ops.copy(ptr, ptr, 0, copyDirection::HostToDevice), NIXL_ERR_INVALID_PARAM);
            EXPECT_EQ(ops.copy(ptr, ptr, 0, copyDirection::DeviceToHost), NIXL_ERR_INVALID_PARAM);
            EXPECT_EQ(ops.memsetDeviceMem(ptr, 0, 0), NIXL_ERR_INVALID_PARAM);
        }
        EXPECT_EQ(byte, 0xA5);
    }

    class deviceOpsTest : public testing::Test {
    protected:
        deviceOps *ops_ = nullptr;

        void
        SetUp() override {
            int count = 0;
            gpuGetDeviceCount(&count, "Probing GPU devices");
            if (count < 1) {
                GTEST_SKIP() << "No GPU is available.";
            }
            gpuSetDevice(0, "Selecting GPU 0");
            ops_ = getDeviceOps();
#ifndef HAVE_CUDA
            if (ops_ == nullptr) {
                GTEST_SKIP() << "No device operations implementation is available.";
            }
#endif
            ASSERT_NE(ops_, nullptr);
            int device = 0;
            ASSERT_EQ(ops_->getActiveDevice(device), NIXL_SUCCESS);
        }
    };

    TEST_F(deviceOpsTest, AllocCopyFree) {
        deviceOps &ops = *ops_;

        deviceMem mem;
        ASSERT_EQ(ops.allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *const device_ptr = mem.get();
        EXPECT_EQ(ops.allocDeviceMem(0, mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem.get(), device_ptr);

        deviceMem moved(std::move(mem));
        EXPECT_FALSE(mem);
        ASSERT_EQ(ops.allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        mem = std::move(moved);
        EXPECT_FALSE(moved);
        EXPECT_EQ(mem.get(), device_ptr);

        std::vector<unsigned char> src(kSize, 0xA5);
        std::vector<unsigned char> dst(kSize, 0);
        ASSERT_EQ(ops.copy(mem.get(), src.data(), kSize, copyDirection::HostToDevice),
                  NIXL_SUCCESS);
        EXPECT_EQ(ops.copy(mem.get(), dst.data(), 0, copyDirection::HostToDevice),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(ops.copy(dst.data(), mem.get(), 0, copyDirection::DeviceToHost),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0));
        EXPECT_EQ(ops.memsetDeviceMem(mem.get(), 0, 0), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(ops.copy(nullptr, src.data(), kSize, copyDirection::HostToDevice),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(ops.copy(dst.data(), nullptr, kSize, copyDirection::DeviceToHost),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(ops.copy(mem.get(), src.data(), kSize, static_cast<copyDirection>(-1)),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(ops.memsetDeviceMem(nullptr, 0, kSize), NIXL_ERR_INVALID_PARAM);
        ASSERT_EQ(ops.copy(dst.data(), mem.get(), kSize, copyDirection::DeviceToHost),
                  NIXL_SUCCESS);
        EXPECT_EQ(dst, src);

        ASSERT_EQ(ops.memsetDeviceMem(mem.get(), 0, kSize), NIXL_SUCCESS);
        ASSERT_EQ(ops.copy(dst.data(), mem.get(), kSize, copyDirection::DeviceToHost),
                  NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0));

        {
            deviceMem adopted(mem.release(), {&ops});
            EXPECT_EQ(adopted.get(), device_ptr);
        }

        mappedHostMem mapped;
        ASSERT_EQ(ops.allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        void *const host_ptr = mapped.hostPointer();
        void *const mapped_device_ptr = mapped.devicePointer();
        EXPECT_EQ(ops.allocMappedHostMem(0, mapped), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mapped.hostPointer(), host_ptr);
        EXPECT_EQ(mapped.devicePointer(), mapped_device_ptr);
        mappedHostMem moved_mapped(std::move(mapped));
        EXPECT_FALSE(mapped);
        EXPECT_EQ(mapped.devicePointer(), nullptr);
        ASSERT_EQ(ops.allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        mapped = std::move(moved_mapped);
        EXPECT_FALSE(moved_mapped);
        EXPECT_EQ(moved_mapped.devicePointer(), nullptr);
        EXPECT_EQ(mapped.hostPointer(), host_ptr);
        EXPECT_EQ(mapped.devicePointer(), mapped_device_ptr);
        std::fill_n(mapped.hostPointer<unsigned char>(), kSize, 0x5A);
        ASSERT_EQ(ops.copy(dst.data(), mapped.devicePointer(), kSize, copyDirection::DeviceToHost),
                  NIXL_SUCCESS);
        EXPECT_EQ(dst, std::vector<unsigned char>(kSize, 0x5A));
    }

} // namespace device_ops
} // namespace gtest
