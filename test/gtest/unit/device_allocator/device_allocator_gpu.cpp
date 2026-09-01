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

// Integration coverage for the real CUDA allocator. The sibling suite uses a
// fake allocator and asserts the handle contract; this one asserts that the
// CUDA implementation behind it actually allocates, frees on the owning
// device, and recovers that device for a pointer whose handle is gone.
//
// Every case skips when no GPU is present, so the suite is safe on CPU-only
// machines and in CI containers without a device.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#include "common.h"
#include "device/device_allocator.h"

namespace gtest {
namespace device_allocator_gpu {

    constexpr size_t kSize = 4096;

    [[nodiscard]] int
    cudaDeviceCount() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
    }

    /**
     * CUDA reports a pointer it no longer knows about as unregistered rather
     * than failing, so this distinguishes a live device allocation from one
     * that has been freed without dereferencing anything.
     */
    [[nodiscard]] bool
    isLiveDevicePointer(const void *ptr) {
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return false;
        }
        return attributes.type == cudaMemoryTypeDevice;
    }

    [[nodiscard]] int
    deviceOfPointer(const void *ptr) {
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return -1;
        }
        return attributes.device;
    }

    class deviceAllocatorGpuTest : public testing::Test {
    protected:
        nixlDeviceAllocator *allocator_ = nullptr;
        int entryDevice_ = 0;

        void
        SetUp() override {
            if (cudaDeviceCount() < 1) {
                GTEST_SKIP() << "No CUDA-capable GPU is available.";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaGetDevice(&entryDevice_), cudaSuccess);
            allocator_ = &nixlGetDeviceAllocator();
        }

        void
        TearDown() override {
            if (allocator_ == nullptr) {
                return;
            }

            int current = -1;
            const cudaError_t query = cudaGetDevice(&current);

            // Restore rather than only check. A test that trips an ASSERT_*
            // returns before whatever cleanup it does itself, so a leaked
            // device selection or a pending asynchronous error would be
            // inherited by every later test in this binary - turning one
            // failure into a cascade that hides its own cause.
            static_cast<void>(cudaSetDevice(entryDevice_));
            static_cast<void>(cudaGetLastError());

            // Having restored it, still report a leak: reaching here on
            // another device means the code under test did not put it back.
            EXPECT_EQ(query, cudaSuccess);
            EXPECT_EQ(current, entryDevice_) << "a device switch leaked out of the test";
        }
    };

    TEST_F(deviceAllocatorGpuTest, ScopeExitActuallyFreesTheAllocation) {
        void *raw = nullptr;
        {
            nixlDeviceMem mem;
            ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);
            raw = mem.get();
            ASSERT_TRUE(isLiveDevicePointer(raw));
        }
        // The handle is gone; CUDA no longer recognises the address.
        EXPECT_FALSE(isLiveDevicePointer(raw));
    }

    /**
     * Ownership can leave RAII through release(), after which no handle
     * carries the owning device any more - so the raw free has to recover it
     * from the pointer itself.
     */
    TEST_F(deviceAllocatorGpuTest, ReleaseThenRawFreeReclaimsTheAllocation) {
        nixlDeviceMem mem;
        ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);

        void *raw = mem.release();
        ASSERT_FALSE(static_cast<bool>(mem));
        ASSERT_TRUE(isLiveDevicePointer(raw));

        // No device id: the implementation must work it out.
        allocator_->freeDeviceMem(raw);
        EXPECT_FALSE(isLiveDevicePointer(raw));
    }

    TEST_F(deviceAllocatorGpuTest, RoundTripsDataThroughTheAllocatorCopies) {
        nixlDeviceMem mem;
        ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);

        std::vector<unsigned char> written(kSize, 0xA5);
        std::vector<unsigned char> read_back(kSize, 0x00);
        ASSERT_EQ(allocator_->copyHostToDevice(mem.get(), written.data(), kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator_->copyDeviceToHost(read_back.data(), mem.get(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(read_back, written);

        ASSERT_EQ(allocator_->memsetDeviceMem(mem.get(), 0, kSize), NIXL_SUCCESS);
        ASSERT_EQ(allocator_->copyDeviceToHost(read_back.data(), mem.get(), kSize), NIXL_SUCCESS);
        EXPECT_EQ(read_back, std::vector<unsigned char>(kSize, 0x00));
    }

    TEST_F(deviceAllocatorGpuTest, MappedHostMemoryIsReachableFromBothAliases) {
        nixlMappedHostMem mapped;
        ASSERT_EQ(allocator_->allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
        ASSERT_TRUE(static_cast<bool>(mapped));
        ASSERT_NE(mapped.hostPtr(), nullptr);
        ASSERT_NE(mapped.devPtr(), nullptr);
        EXPECT_EQ(mapped.size(), kSize);

        // Write through the host alias, read through the device alias.
        auto *host = mapped.asHost<unsigned char>();
        for (size_t i = 0; i < kSize; ++i) {
            host[i] = static_cast<unsigned char>(i & 0xFF);
        }

        std::vector<unsigned char> read_back(kSize, 0x00);
        ASSERT_EQ(allocator_->copyDeviceToHost(read_back.data(), mapped.devPtr(), kSize),
                  NIXL_SUCCESS);
        for (size_t i = 0; i < kSize; ++i) {
            ASSERT_EQ(read_back[i], static_cast<unsigned char>(i & 0xFF)) << "at byte " << i;
        }
    }

    TEST_F(deviceAllocatorGpuTest, MappedHostAllocationIsFreedOnScopeExit) {
        void *host = nullptr;
        {
            nixlMappedHostMem mapped;
            ASSERT_EQ(allocator_->allocMappedHostMem(kSize, mapped), NIXL_SUCCESS);
            host = mapped.hostPtr();

            cudaPointerAttributes attributes{};
            ASSERT_EQ(cudaPointerGetAttributes(&attributes, host), cudaSuccess);
            EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
        }
        // cudaFreeHost unregisters the mapping, so CUDA stops recognising it.
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, host), cudaSuccess);
        EXPECT_NE(attributes.type, cudaMemoryTypeHost);
    }

    /**
     * A raw free must not hand a non-device pointer to cudaFree. CUDA reports
     * an unknown or host address as unregistered rather than failing, so the
     * implementation has to check the type rather than trust the caller.
     */
    TEST_F(deviceAllocatorGpuTest, RawFreeRefusesAPointerThatIsNotDeviceMemory) {
        // The refusal is logged at error level. The unit binary does not
        // install LogProblemCounter, so this only silences the message if the
        // suite is ever linked into a binary that does.
        LogIgnoreGuard ignore("Refusing to cudaFree");

        std::vector<unsigned char> host(kSize, 0x11);
        allocator_->freeDeviceMem(host.data());

        // Still ours, still readable: nothing was freed underneath us.
        EXPECT_EQ(host[0], 0x11);
        EXPECT_EQ(host[kSize - 1], 0x11);
    }

    /**
     * cudaPointerGetAttributes can return a leftover asynchronous error
     * instead of classifying the pointer. Free must clear that error, retry
     * classification, and still reclaim a live allocation.
     */
    TEST_F(deviceAllocatorGpuTest, FreeReclaimsAfterAStickyCudaError) {
        nixlDeviceMem mem;
        ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *raw = mem.get();
        ASSERT_TRUE(isLiveDevicePointer(raw));

        ASSERT_NE(cudaMemcpy(nullptr, nullptr, 1, cudaMemcpyHostToDevice), cudaSuccess);

        mem.reset();
        EXPECT_FALSE(isLiveDevicePointer(raw));
    }

    /**
     * Freeing an allocation while a different device is current is the case
     * the recorded device id exists for. Needs a second GPU to be meaningful.
     */
    TEST_F(deviceAllocatorGpuTest, FreesOnTheAllocatingDeviceWhenAnotherIsCurrent) {
        if (cudaDeviceCount() < 2) {
            GTEST_SKIP() << "Needs two GPUs; this machine has " << cudaDeviceCount() << ".";
        }

        nixlDeviceMem mem;
        ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *raw = mem.get();
        // The premise of the test: the allocation lives on the entry device,
        // so freeing it from another one exercises the recovery path.
        ASSERT_EQ(deviceOfPointer(raw), entryDevice_);

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        mem.reset();

        EXPECT_FALSE(isLiveDevicePointer(raw));

        // The free must leave the caller's device selection where it found it.
        int current = -1;
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 1);

        ASSERT_EQ(cudaSetDevice(entryDevice_), cudaSuccess);
    }

    /** Same case for the raw path, where the device comes from the pointer. */
    TEST_F(deviceAllocatorGpuTest, RawFreeRecoversTheDeviceWhenAnotherIsCurrent) {
        if (cudaDeviceCount() < 2) {
            GTEST_SKIP() << "Needs two GPUs; this machine has " << cudaDeviceCount() << ".";
        }

        nixlDeviceMem mem;
        ASSERT_EQ(allocator_->allocDeviceMem(kSize, mem), NIXL_SUCCESS);
        void *raw = mem.release();
        ASSERT_EQ(deviceOfPointer(raw), entryDevice_);

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        allocator_->freeDeviceMem(raw);

        EXPECT_FALSE(isLiveDevicePointer(raw));
        int current = -1;
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 1);

        ASSERT_EQ(cudaSetDevice(entryDevice_), cudaSuccess);
    }

} // namespace device_allocator_gpu
} // namespace gtest
