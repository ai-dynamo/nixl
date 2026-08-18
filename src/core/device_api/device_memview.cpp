/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device_api/device_memview.h"

#include <gpu/common/nixl_device_types.cuh>

#include "device/device_buffer.h"

static_assert(sizeof(nixl_device_exec_mode_t) == 1);

nixl_status_t
nixlDeviceMemViewAllocate(bool use_proxy,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out) noexcept {
    wrapper_out = nullptr;
    if (backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixl_device_exec_mode_t execution_mode =
        use_proxy ? nixl_device_exec_mode_t::PROXY : nixl_device_exec_mode_t::UCX_DIRECT;

    void *device_wrapper = nullptr;
    auto status = nixlDeviceBufferAllocate(&device_wrapper, sizeof(nixlDeviceMemViewWrapper));
    if (status != NIXL_SUCCESS) {
        return status;
    }

    const nixlDeviceMemViewWrapper host_wrapper{
        execution_mode,
        backend_memview,
    };
    status = nixlDeviceBufferCopyHostToDevice(device_wrapper, &host_wrapper, sizeof(host_wrapper));
    if (status != NIXL_SUCCESS) {
        nixlDeviceBufferFree(device_wrapper);
        return status;
    }

    wrapper_out = device_wrapper;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlDeviceMemViewGetBackend(nixlMemViewH wrapper, nixlMemViewH &backend_out) noexcept {
    backend_out = nullptr;
    if (wrapper == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlDeviceMemViewWrapper host_wrapper{};
    const auto status =
        nixlDeviceBufferCopyDeviceToHost(&host_wrapper, wrapper, sizeof(host_wrapper));
    if (status != NIXL_SUCCESS) {
        return status;
    }

    backend_out = host_wrapper.backend_memview;
    return NIXL_SUCCESS;
}

void
nixlDeviceMemViewFree(nixlMemViewH wrapper) noexcept {
    nixlDeviceBufferFree(wrapper);
}
