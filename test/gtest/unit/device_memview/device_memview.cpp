/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <new>

#include "device/device_allocator.h"
#include "device/device_memview.h"
#include "gpu/device_types.cuh"
#include "agent_data.h"

namespace {

class MockAllocator final : public nixlDeviceAllocator {
public:
    bool fail_copy = false;
    bool fail_sync = false;
    size_t allocations = 0;
    int active_device = 0;

    nixl_status_t
    copyHostToDevice(void *dst, const void *src, size_t size) noexcept override {
        if (fail_copy) {
            return NIXL_ERR_BACKEND;
        }
        std::memcpy(dst, src, size);
        return NIXL_SUCCESS;
    }

    nixl_status_t
    copyDeviceToHost(void *dst, const void *src, size_t size) noexcept override {
        std::memcpy(dst, src, size);
        return NIXL_SUCCESS;
    }

    nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
        std::memset(ptr, value, size);
        return NIXL_SUCCESS;
    }

    nixl_status_t
    synchronize() noexcept override {
        return fail_sync ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
    }

    nixl_status_t
    getActiveDevice(int &device_id) noexcept override {
        device_id = active_device;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    setActiveDevice(int device_id) noexcept override {
        active_device = device_id;
        return NIXL_SUCCESS;
    }

protected:
    nixl_status_t
    doAllocDeviceMem(void **ptr, size_t size) noexcept override {
        *ptr = new (std::nothrow) unsigned char[size];
        if (*ptr == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        ++allocations;
        return NIXL_SUCCESS;
    }

    void
    doFreeDeviceMem(void *ptr) noexcept override {
        delete[] static_cast<unsigned char *>(ptr);
        --allocations;
    }

    nixl_status_t
    doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t) noexcept override {
        *host_ptr = nullptr;
        *dev_ptr = nullptr;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    void
    doFreeMappedHostMem(void *) noexcept override {}
};

TEST(DeviceMemViewTest, UploadAndFreeUseInjectedAllocator) {
    MockAllocator allocator;
    nixlMemViewH wrapper = nullptr;
    void *backend_view = reinterpret_cast<void *>(0x1234);

    ASSERT_EQ(nixlDeviceMemViewAllocate(
                  nixl_device_exec_mode_t::UCX_DIRECT, backend_view, wrapper, &allocator),
              NIXL_SUCCESS);
    ASSERT_NE(wrapper, nullptr);
    EXPECT_EQ(allocator.allocations, 1U);

    nixlDeviceMemViewFree(wrapper, &allocator);
    EXPECT_EQ(allocator.allocations, 0U);
}

TEST(DeviceMemViewTest, CopyFailureRollsBackDeviceAllocation) {
    MockAllocator allocator;
    allocator.fail_copy = true;
    nixlMemViewH wrapper = reinterpret_cast<void *>(0x1);

    EXPECT_EQ(nixlDeviceMemViewAllocate(nixl_device_exec_mode_t::UCX_DIRECT,
                                        reinterpret_cast<void *>(0x1234),
                                        wrapper,
                                        &allocator),
              NIXL_ERR_BACKEND);
    EXPECT_EQ(wrapper, nullptr);
    EXPECT_EQ(allocator.allocations, 0U);
}

TEST(DeviceMemViewTest, SyncFailureRollsBackDeviceAllocation) {
    MockAllocator allocator;
    allocator.fail_sync = true;
    nixlMemViewH wrapper = reinterpret_cast<void *>(0x1);

    EXPECT_EQ(nixlDeviceMemViewAllocate(nixl_device_exec_mode_t::UCX_DIRECT,
                                        reinterpret_cast<void *>(0x1234),
                                        wrapper,
                                        &allocator),
              NIXL_ERR_BACKEND);
    EXPECT_EQ(wrapper, nullptr);
    EXPECT_EQ(allocator.allocations, 0U);
}

TEST(DeviceMemViewTest, InvalidInputsDoNotAllocate) {
    MockAllocator allocator;
    nixlMemViewH wrapper = reinterpret_cast<void *>(0x1);

    EXPECT_EQ(
        nixlDeviceMemViewAllocate(
            nixl_device_exec_mode_t::NONE, reinterpret_cast<void *>(0x1234), wrapper, &allocator),
        NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(wrapper, nullptr);
    EXPECT_EQ(allocator.allocations, 0U);

    wrapper = reinterpret_cast<void *>(0x1);
    EXPECT_EQ(nixlDeviceMemViewAllocate(
                  nixl_device_exec_mode_t::UCX_DIRECT, nullptr, wrapper, &allocator),
              NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(wrapper, nullptr);
    EXPECT_EQ(allocator.allocations, 0U);
}

TEST(DeviceMemViewTest, StatusTagReservesFourBytesOutsidePayload) {
    nixlGpuXferStatusH status{};
    uint32_t tag = 0;
    std::memcpy(&tag, status.storage + nixl::gpu::xfer_status_mode_offset, sizeof(tag));
    EXPECT_EQ(sizeof(status), 64U);
    EXPECT_EQ(nixl_gpu_xfer_status_payload_size, 60U);
    EXPECT_EQ(tag, 0U);

    tag = static_cast<uint32_t>(nixl_device_exec_mode_t::GPUNETIO_DIRECT);
    std::memcpy(status.storage + nixl::gpu::xfer_status_mode_offset, &tag, sizeof(tag));
    std::memcpy(&tag, status.storage + nixl::gpu::xfer_status_mode_offset, sizeof(tag));
    EXPECT_EQ(tag, 3U);
}

TEST(DeviceMemViewTest, NativeRemoteAdmissionIsNarrow) {
    EXPECT_TRUE(nixlRemoteBackendAdmissionAllowed(
        "GPUNETIO", true, false, nixl_device_exec_mode_t::GPUNETIO_DIRECT));
    EXPECT_FALSE(nixlRemoteBackendAdmissionAllowed(
        "UCX", true, false, nixl_device_exec_mode_t::GPUNETIO_DIRECT));
    EXPECT_FALSE(
        nixlRemoteBackendAdmissionAllowed("GPUNETIO", true, false, nixl_device_exec_mode_t::NONE));
    EXPECT_TRUE(
        nixlRemoteBackendAdmissionAllowed("UCX", true, true, nixl_device_exec_mode_t::NONE));
}

} // namespace
