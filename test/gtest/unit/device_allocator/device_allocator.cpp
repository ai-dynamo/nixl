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

// Ownership semantics of device-memory RAII handles against a fake allocator.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "device/device_allocator.h"

namespace gtest {
namespace device_allocator {

    class fakeAllocator : public nixlDeviceAllocator {
    public:
        std::vector<void *> freed;
        std::vector<void *> freedHost;
        int activeDevice = 0;
        bool failAlloc = false;

        nixl_status_t
        copyHostToDevice(void *, const void *, size_t) noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        copyDeviceToHost(void *, const void *, size_t) noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        memsetDeviceMem(void *, int, size_t) noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        synchronize() noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        getActiveDevice(int &device_id) noexcept override {
            device_id = activeDevice;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        setActiveDevice(int device_id) noexcept override {
            activeDevice = device_id;
            return NIXL_SUCCESS;
        }

    protected:
        nixl_status_t
        doAllocDeviceMem(void **ptr, size_t size) noexcept override {
            if (ptr == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (failAlloc) {
                return NIXL_ERR_BACKEND;
            }
            *ptr = nextToken();
            return NIXL_SUCCESS;
        }

        void
        doFreeDeviceMem(void *ptr) noexcept override {
            freed.push_back(ptr);
        }

        nixl_status_t
        doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept override {
            if (host_ptr == nullptr || dev_ptr == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (failAlloc) {
                return NIXL_ERR_BACKEND;
            }
            *host_ptr = nextToken();
            *dev_ptr = nextToken();
            return NIXL_SUCCESS;
        }

        void
        doFreeMappedHostMem(void *host_ptr) noexcept override {
            freedHost.push_back(host_ptr);
        }

    private:
        void *
        nextToken() noexcept {
            return reinterpret_cast<void *>(++counter_ * sizeof(void *));
        }

        std::uintptr_t counter_ = 0;
    };

    TEST(deviceAllocatorTest, HandleFreesOnScopeExit) {
        fakeAllocator allocator;
        void *raw = nullptr;
        {
            nixlDeviceMem mem;
            ASSERT_EQ(allocator.allocDeviceMem(128, mem), NIXL_SUCCESS);
            EXPECT_EQ(mem.size(), 128u);
            raw = mem.get();
        }
        ASSERT_EQ(allocator.freed.size(), 1u);
        EXPECT_EQ(allocator.freed.front(), raw);
    }

    TEST(deviceAllocatorTest, MoveTransfersOwnershipExactlyOnce) {
        fakeAllocator allocator;
        nixlDeviceMem first;
        ASSERT_EQ(allocator.allocDeviceMem(64, first), NIXL_SUCCESS);
        void *raw = first.get();
        nixlDeviceMem second = std::move(first);

        EXPECT_FALSE(static_cast<bool>(first));
        EXPECT_EQ(second.get(), raw);
        second.reset();
        first.reset();

        ASSERT_EQ(allocator.freed.size(), 1u);
        EXPECT_EQ(allocator.freed.front(), raw);
    }

    TEST(deviceAllocatorTest, MoveAssignFreesTheOverwrittenAllocation) {
        fakeAllocator allocator;
        nixlDeviceMem victim, donor;
        ASSERT_EQ(allocator.allocDeviceMem(32, victim), NIXL_SUCCESS);
        ASSERT_EQ(allocator.allocDeviceMem(32, donor), NIXL_SUCCESS);
        void *victim_raw = victim.get();

        victim = std::move(donor);

        ASSERT_EQ(allocator.freed.size(), 1u);
        EXPECT_EQ(allocator.freed.front(), victim_raw);
    }

    TEST(deviceAllocatorTest, ReleaseHandsOwnershipToTheCaller) {
        fakeAllocator allocator;
        nixlDeviceMem mem;
        ASSERT_EQ(allocator.allocDeviceMem(16, mem), NIXL_SUCCESS);
        void *raw = mem.release();

        EXPECT_FALSE(static_cast<bool>(mem));
        allocator.freeDeviceMem(raw);
        ASSERT_EQ(allocator.freed.size(), 1u);
        EXPECT_EQ(allocator.freed.front(), raw);
    }

    TEST(deviceAllocatorTest, FailedAllocationLeavesTheHandleUntouched) {
        fakeAllocator allocator;
        nixlDeviceMem mem;
        ASSERT_EQ(allocator.allocDeviceMem(16, mem), NIXL_SUCCESS);
        void *raw = mem.get();
        const size_t size = mem.size();
        allocator.failAlloc = true;

        // Every member has to survive, not just the pointer: a handle left
        // with the old pointer and a clobbered size still describes the
        // wrong buffer.
        EXPECT_EQ(allocator.allocDeviceMem(16, mem), NIXL_ERR_BACKEND);
        EXPECT_EQ(mem.get(), raw);
        EXPECT_EQ(mem.size(), size);
        EXPECT_TRUE(allocator.freed.empty());
    }

    TEST(deviceAllocatorTest, DeviceAllocationRejectsZeroSize) {
        fakeAllocator allocator;
        nixlDeviceMem mem;
        EXPECT_EQ(allocator.allocDeviceMem(0, mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(static_cast<bool>(mem));
    }

    TEST(deviceAllocatorTest, MappedHostHandleExposesBothPointersAndFreesHost) {
        fakeAllocator allocator;
        void *host = nullptr;
        {
            nixlMappedHostMem mapped;
            ASSERT_EQ(allocator.allocMappedHostMem(256, mapped), NIXL_SUCCESS);
            EXPECT_NE(mapped.hostPtr(), nullptr);
            EXPECT_NE(mapped.devPtr(), nullptr);
            EXPECT_NE(mapped.hostPtr(), mapped.devPtr());
            EXPECT_EQ(mapped.size(), 256u);
            host = mapped.hostPtr();
        }
        ASSERT_EQ(allocator.freedHost.size(), 1u);
        EXPECT_EQ(allocator.freedHost.front(), host);
    }

    TEST(deviceAllocatorTest, MappedHostMoveTransfersOwnershipExactlyOnce) {
        fakeAllocator allocator;
        nixlMappedHostMem first;
        ASSERT_EQ(allocator.allocMappedHostMem(64, first), NIXL_SUCCESS);
        void *host = first.hostPtr();
        nixlMappedHostMem second = std::move(first);

        EXPECT_FALSE(static_cast<bool>(first));
        second.reset();
        first.reset();

        ASSERT_EQ(allocator.freedHost.size(), 1u);
        EXPECT_EQ(allocator.freedHost.front(), host);
    }

    /**
     * operator= has to reset the target before taking the donor's four
     * members. Dropping either the reset or the dev_ptr_ transfer would leak
     * the overwritten host allocation without failing any other test here.
     */
    TEST(deviceAllocatorTest, MappedHostMoveAssignFreesTheOverwrittenAllocation) {
        fakeAllocator allocator;
        nixlMappedHostMem victim, donor;
        ASSERT_EQ(allocator.allocMappedHostMem(64, victim), NIXL_SUCCESS);
        ASSERT_EQ(allocator.allocMappedHostMem(64, donor), NIXL_SUCCESS);
        void *victim_host = victim.hostPtr();
        void *donor_host = donor.hostPtr();
        void *donor_dev = donor.devPtr();

        victim = std::move(donor);

        // The overwritten allocation is freed exactly once, as part of the
        // assignment rather than at scope exit.
        ASSERT_EQ(allocator.freedHost.size(), 1u);
        EXPECT_EQ(allocator.freedHost.front(), victim_host);

        // Both pointers moved across, so the device alias cannot be stranded.
        EXPECT_EQ(victim.hostPtr(), donor_host);
        EXPECT_EQ(victim.devPtr(), donor_dev);
        EXPECT_FALSE(static_cast<bool>(donor));
    }

    TEST(deviceAllocatorTest, MappedHostFailedAllocationLeavesTheHandleUntouched) {
        fakeAllocator allocator;
        nixlMappedHostMem mapped;
        ASSERT_EQ(allocator.allocMappedHostMem(16, mapped), NIXL_SUCCESS);
        void *host = mapped.hostPtr();
        void *dev = mapped.devPtr();
        const size_t size = mapped.size();
        allocator.failAlloc = true;

        // The device alias and the size matter as much as the host pointer:
        // keeping one and losing the others leaves the two halves of the
        // mapping describing different memory.
        EXPECT_EQ(allocator.allocMappedHostMem(16, mapped), NIXL_ERR_BACKEND);
        EXPECT_EQ(mapped.hostPtr(), host);
        EXPECT_EQ(mapped.devPtr(), dev);
        EXPECT_EQ(mapped.size(), size);
        EXPECT_TRUE(allocator.freedHost.empty());
    }

    TEST(deviceAllocatorTest, MappedHostAllocationRejectsZeroSize) {
        fakeAllocator allocator;
        nixlMappedHostMem mapped;
        EXPECT_EQ(allocator.allocMappedHostMem(0, mapped), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(static_cast<bool>(mapped));
    }

} // namespace device_allocator
} // namespace gtest
