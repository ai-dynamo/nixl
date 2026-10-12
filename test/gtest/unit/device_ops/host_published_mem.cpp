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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "device/device_ops.h"
#include "device/host_published_mem.h"
#include "gpu_utils.h"

namespace gtest {
namespace host_published_mem {

    using nixl::deviceOps;
    using nixl::hostPublishedDeviceMem;
    using backing = hostPublishedDeviceMem::backing;

    /** Stands in for a backend's device-resident backing. */
    class fakeDevicePublishedMem final : public hostPublishedDeviceMem {
    public:
        explicit fakeDevicePublishedMem(size_t size)
            : hostPublishedDeviceMem(&word_, size, backing::device) {}

    private:
        nixl_status_t
        doPublish(size_t, uint64_t value) noexcept override {
            word_ = value;
            return NIXL_SUCCESS;
        }

        uint64_t word_ = 0;
    };

    /** Host-only deviceOps whose mapped memory is plain heap memory. */
    class fakeOps : public deviceOps {
    public:
        /** Result of the backend allocation; NIXL_ERR_NOT_SUPPORTED mimics the default. */
        nixl_status_t backend_status = NIXL_ERR_NOT_SUPPORTED;
        bool fail_mapped = false;
        int mapped_frees = 0;

        nixl_status_t
        doAllocDeviceMem(void *&, size_t) noexcept override {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        void
        doFreeDeviceMem(void *) noexcept override {}

        nixl_status_t
        doAllocMappedHostMem(void *&host_ptr, void *&dev_ptr, size_t size) noexcept override {
            if (fail_mapped) {
                return NIXL_ERR_BACKEND;
            }
            // Poisoned so the test sees that allocHostPublishedMem zeroes it.
            host_ptr = std::malloc(size);
            std::memset(host_ptr, 0xA5, size);
            dev_ptr = host_ptr;
            return NIXL_SUCCESS;
        }

        void
        doFreeMappedHostMem(void *host_ptr) noexcept override {
            std::free(host_ptr);
            ++mapped_frees;
        }

        nixl_status_t
        doAllocHostPublishedMem(size_t size,
                                std::unique_ptr<hostPublishedDeviceMem> &out) noexcept override {
            if (backend_status == NIXL_SUCCESS) {
                out = std::make_unique<fakeDevicePublishedMem>(size);
            }
            return backend_status;
        }

        nixl_status_t
        copy(void *, const void *, size_t, copyDirection) noexcept override {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        nixl_status_t
        memsetDeviceMem(void *, int, size_t) noexcept override {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        nixl_status_t
        synchronize() noexcept override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        getActiveDevice(int &device_id) noexcept override {
            device_id = 0;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        setActiveDevice(int) noexcept override {
            return NIXL_SUCCESS;
        }
    };

    TEST(hostPublishedMem, ZeroSizeLeavesOutputUnchanged) {
        fakeOps ops;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        EXPECT_EQ(ops.allocHostPublishedMem(0, mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_FALSE(mem);

        ASSERT_EQ(ops.allocHostPublishedMem(8, mem), NIXL_SUCCESS);
        hostPublishedDeviceMem *const kept = mem.get();
        EXPECT_EQ(ops.allocHostPublishedMem(0, mem), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem.get(), kept);
    }

    TEST(hostPublishedMem, MappedHostIsZeroedAndPublishesWholeWords) {
        fakeOps ops;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        ASSERT_EQ(ops.allocHostPublishedMem(16, mem), NIXL_SUCCESS);
        EXPECT_EQ(mem->backingKind(), backing::mapped_host);
        EXPECT_EQ(mem->size(), 16u);

        auto *words = static_cast<uint64_t *>(mem->devicePointer());
        EXPECT_EQ(words[0], 0u);
        EXPECT_EQ(words[1], 0u);

        ASSERT_EQ(mem->publish(0, 0xffffffff00000000ull), NIXL_SUCCESS);
        EXPECT_EQ(words[0], 0xffffffff00000000ull);
        ASSERT_EQ(mem->publish(8, 1), NIXL_SUCCESS);
        EXPECT_EQ(words[1], 1u);

        EXPECT_EQ(mem->publish(1, 2), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem->publish(16, 2), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(mem->publish(SIZE_MAX - 7, 2), NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(words[1], 1u);
    }

    TEST(hostPublishedMem, PrefersTheBackendBacking) {
        fakeOps ops;
        ops.backend_status = NIXL_SUCCESS;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        ASSERT_EQ(ops.allocHostPublishedMem(8, mem), NIXL_SUCCESS);
        EXPECT_EQ(mem->backingKind(), backing::device);
        ASSERT_EQ(mem->publish(0, 7), NIXL_SUCCESS);
        EXPECT_EQ(*static_cast<uint64_t *>(mem->devicePointer()), 7u);
        mem.reset();
        EXPECT_EQ(ops.mapped_frees, 0);
    }

    TEST(hostPublishedMem, BackendFailureFallsBackToMappedHost) {
        fakeOps ops;
        ops.backend_status = NIXL_ERR_BACKEND;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        ASSERT_EQ(ops.allocHostPublishedMem(8, mem), NIXL_SUCCESS);
        EXPECT_EQ(mem->backingKind(), backing::mapped_host);
    }

    TEST(hostPublishedMem, MappedFailureLeavesOutputUnchanged) {
        fakeOps ops;
        ops.fail_mapped = true;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        EXPECT_EQ(ops.allocHostPublishedMem(8, mem), NIXL_ERR_BACKEND);
        EXPECT_FALSE(mem);
    }

    TEST(hostPublishedMem, DestroyingMappedHostFreesIt) {
        fakeOps ops;
        {
            std::unique_ptr<hostPublishedDeviceMem> mem;
            ASSERT_EQ(ops.allocHostPublishedMem(8, mem), NIXL_SUCCESS);
        }
        EXPECT_EQ(ops.mapped_frees, 1);
    }

    TEST(hostPublishedMemGpu, IsZeroedAndPublishesToTheDevice) {
        int count = 0;
        gpuGetDeviceCount(&count, "Probing GPU devices");
        if (count < 1) {
            GTEST_SKIP() << "No GPU is available.";
        }
        deviceOps *ops = nixl::getDeviceOps();
        if (ops == nullptr) {
            GTEST_SKIP() << "No device operations implementation is available.";
        }

        constexpr size_t kWords = 3;
        std::unique_ptr<hostPublishedDeviceMem> mem;
        ASSERT_EQ(ops->allocHostPublishedMem(kWords * sizeof(uint64_t), mem), NIXL_SUCCESS);
        RecordProperty("backing", mem->backingKind() == backing::device ? "device" : "mapped_host");

        uint64_t got[kWords] = {1, 1, 1};
        ASSERT_EQ(
            ops->copy(
                got, mem->devicePointer(), sizeof(got), deviceOps::copyDirection::DeviceToHost),
            NIXL_SUCCESS);
        EXPECT_EQ(got[0], 0u);
        EXPECT_EQ(got[1], 0u);
        EXPECT_EQ(got[2], 0u);

        ASSERT_EQ(mem->publish(sizeof(uint64_t), 0x123456789abcdef0ull), NIXL_SUCCESS);
        ASSERT_EQ(
            ops->copy(
                got, mem->devicePointer(), sizeof(got), deviceOps::copyDirection::DeviceToHost),
            NIXL_SUCCESS);
        EXPECT_EQ(got[0], 0u);
        EXPECT_EQ(got[1], 0x123456789abcdef0ull);
        EXPECT_EQ(got[2], 0u);
    }

} // namespace host_published_mem
} // namespace gtest
