/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device/device_memview.h"

#include <gpu/device_types.cuh>

#include "device/device_ops.h"

static_assert(sizeof(nixl_device_exec_mode_t) == 1);

nixl_status_t
nixlDeviceMemViewAllocate(nixl_device_exec_mode_t execution_mode,
                          nixlMemViewH backend_memview,
                          nixl::deviceViewHandle &wrapper_out) noexcept {
    wrapper_out = nullptr;
    if (backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    nixl::deviceMem wrapper_mem;
    auto status = ops->allocDeviceMem(sizeof(nixlDeviceMemViewWrapper), wrapper_mem);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlDeviceMemViewWrapper host_wrapper{
        execution_mode,
        backend_memview,
    };
    status = ops->copy(wrapper_mem.get(),
                       &host_wrapper,
                       sizeof(host_wrapper),
                       nixl::deviceOps::copyDirection::HostToDevice);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    /* Ownership crosses the public handle boundary; reclaimed in
     * nixlDeviceMemViewFree. */
    wrapper_out = wrapper_mem.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDeviceMemViewGetBackend(nixl::deviceViewHandle wrapper, nixlMemViewH &backend_out) noexcept {
    backend_out = nullptr;
    if (wrapper == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    nixlDeviceMemViewWrapper host_wrapper{};
    const auto status = ops->copy(
        &host_wrapper, wrapper, sizeof(host_wrapper), nixl::deviceOps::copyDirection::DeviceToHost);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    backend_out = host_wrapper.backend_memview;
    return NIXL_SUCCESS;
}

void
nixlDeviceMemViewFree(nixl::deviceViewHandle wrapper) noexcept {
    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        return;
    }
    nixl::deviceMem owner(wrapper, nixl::deviceMemDeleter{ops});
}
