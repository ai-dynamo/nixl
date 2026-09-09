/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device/device_memview.h"

#include "device/device_allocator.h"

nixl_status_t
nixlDeviceMemViewAllocate(nixl_device_exec_mode_t execution_mode,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out,
                          nixlDeviceAllocator *allocator_override) noexcept {
    wrapper_out = nullptr;
    if (execution_mode == nixl_device_exec_mode_t::NONE || backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlDeviceAllocator &allocator =
        allocator_override == nullptr ? nixlGetDeviceAllocator() : *allocator_override;
    nixlDeviceMem wrapper_mem;
    nixl_status_t status = allocator.allocDeviceMem(sizeof(nixlDeviceMemViewWrapper), wrapper_mem);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlDeviceMemViewWrapper host_wrapper{execution_mode, backend_memview};
    status = allocator.copyHostToDevice(wrapper_mem.get(), &host_wrapper, sizeof(host_wrapper));
    if (status != NIXL_SUCCESS) {
        return status;
    }
    status = allocator.synchronize();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    wrapper_out = wrapper_mem.release();
    return NIXL_SUCCESS;
}

void
nixlDeviceMemViewFree(nixlMemViewH wrapper, nixlDeviceAllocator *allocator_override) noexcept {
    if (wrapper != nullptr) {
        nixlDeviceAllocator &allocator =
            allocator_override == nullptr ? nixlGetDeviceAllocator() : *allocator_override;
        allocator.freeDeviceMem(wrapper);
    }
}
