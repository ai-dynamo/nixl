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
#ifndef NIXL_SRC_API_GPU_UCX_NIXL_DEVICE_IMPL_CUH
#define NIXL_SRC_API_GPU_UCX_NIXL_DEVICE_IMPL_CUH

#include "../common/nixl_device_types.cuh"

#include <ucp/api/device/ucp_device_impl.h>

#include <cassert>
#include <cstring>

namespace nixl::gpu::ucx_impl {

template<nixl_gpu_level_t level> struct UcpDeviceLevel;

template<> struct UcpDeviceLevel<nixl_gpu_level_t::THREAD> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_THREAD;
};

template<> struct UcpDeviceLevel<nixl_gpu_level_t::WARP> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_WARP;
};

template<> struct UcpDeviceLevel<nixl_gpu_level_t::BLOCK> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_BLOCK;
};

template<> struct UcpDeviceLevel<nixl_gpu_level_t::GRID> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_GRID;
};

__device__ inline uint64_t
to_ucp_flags(uint64_t nixl_flags) noexcept {
    constexpr uint64_t all_known_nixl_flags{nixl_gpu_flags::defer};
    assert((nixl_flags & ~all_known_nixl_flags) == 0);

    uint64_t ucp_flags{UCP_DEVICE_FLAG_NODELAY};
    if (nixl_flags & nixl_gpu_flags::defer) {
        ucp_flags &= ~UCP_DEVICE_FLAG_NODELAY;
    }
    return ucp_flags;
}

__device__ inline nixl_status_t
convert_status(ucs_status_t status) {
    if (!UCS_STATUS_IS_ERR(status)) {
        return NIXL_IN_PROG;
    }
    printf("UCX returned error: %d\n", status);
    return NIXL_ERR_BACKEND;
}

__device__ inline ucp_device_request_t *
request_ptr(nixlGpuXferStatusH *xfer_status) {
    static_assert(sizeof(ucp_device_request_t) <= sizeof(nixlGpuXferStatusH{}.storage),
                  "nixlGpuXferStatusH storage is too small for UCX device request");
    return xfer_status ? reinterpret_cast<ucp_device_request_t *>(xfer_status->storage) : nullptr;
}

__device__ inline ucp_device_local_mem_list_h
local_mem_list(nixlMemViewH mvh) {
    return static_cast<ucp_device_local_mem_list_h>(mvh);
}

__device__ inline ucp_device_remote_mem_list_h
remote_mem_list(nixlMemViewH mvh) {
    return static_cast<ucp_device_remote_mem_list_h>(mvh);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    const auto status =
        ucp_device_progress_req<UcpDeviceLevel<level>::value>(request_ptr(&xfer_status));

    switch (status) {
    case UCS_OK:
        return NIXL_SUCCESS;
    case UCS_INPROGRESS:
        return NIXL_IN_PROG;
    default:
        return NIXL_ERR_BACKEND;
    }
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    auto src_mem_list = local_mem_list(src.mvh);
    auto dst_mem_list = remote_mem_list(dst.mvh);
    const auto status = ucp_device_put<UcpDeviceLevel<level>::value>(src_mem_list,
                                                                     src.index,
                                                                     src.offset,
                                                                     dst_mem_list,
                                                                     dst.index,
                                                                     dst.offset,
                                                                     size,
                                                                     channel_id,
                                                                     to_ucp_flags(flags),
                                                                     request_ptr(xfer_status));
    return convert_status(status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    auto mem_list = remote_mem_list(counter.mvh);
    const auto status =
        ucp_device_counter_inc<UcpDeviceLevel<level>::value>(value,
                                                             mem_list,
                                                             counter.index,
                                                             counter.offset,
                                                             channel_id,
                                                             to_ucp_flags(flags),
                                                             request_ptr(xfer_status));
    return convert_status(status);
}

__device__ inline void *
get_ptr(nixlMemViewH mvh, size_t index) {
    auto mem_list = remote_mem_list(mvh);
    void *ptr = nullptr;
    ucp_device_get_ptr(mem_list, index, &ptr);
    return ptr;
}

} // namespace nixl::gpu::ucx_impl

#endif // NIXL_SRC_API_GPU_UCX_NIXL_DEVICE_IMPL_CUH
