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
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_UCX_DEVICE_UCX_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_UCX_DEVICE_UCX_CUH

#include "../../device_types.cuh"

#include <ucp/api/device/ucp_device_impl.h>

#include <cassert>
#include <cstring>

namespace nixl::gpu::impl::ucx {

template<level_t level> struct ucsDeviceLevel;

template<> struct ucsDeviceLevel<level_t::THREAD> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_THREAD;
};

template<> struct ucsDeviceLevel<level_t::WARP> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_WARP;
};

template<> struct ucsDeviceLevel<level_t::BLOCK> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_BLOCK;
};

template<> struct ucsDeviceLevel<level_t::GRID> {
    static constexpr ucs_device_level_t value = UCS_DEVICE_LEVEL_GRID;
};

__device__ inline uint64_t
toUcpFlags(uint64_t nixl_flags) noexcept {
    constexpr uint64_t all_known_nixl_flags{flags::defer};
    assert((nixl_flags & ~all_known_nixl_flags) == 0);

    uint64_t ucp_flags{UCP_DEVICE_FLAG_NODELAY};
    if (nixl_flags & flags::defer) {
        ucp_flags &= ~UCP_DEVICE_FLAG_NODELAY;
    }
    return ucp_flags;
}

__device__ inline nixl_status_t
convertStatus(ucs_status_t status) {
    switch (status) {
    case UCS_OK:
        return NIXL_SUCCESS;
    case UCS_INPROGRESS:
        return NIXL_IN_PROG;
    default:
        printf("UCX returned error: %d\n", status);
        return NIXL_ERR_BACKEND;
    }
}

__device__ inline ucp_device_request_t *
requestPtr(xferStatusH *xfer_status) {
    static_assert(sizeof(ucp_device_request_t) <= xfer_status_payload_size,
                  "transfer-status payload is too small for UCX device request");
    return xfer_status ? reinterpret_cast<ucp_device_request_t *>(xfer_status->storage) : nullptr;
}

__device__ inline ucp_device_local_mem_list_h
localMemList(nixlMemViewH mvh) {
    return static_cast<ucp_device_local_mem_list_h>(mvh);
}

__device__ inline ucp_device_remote_mem_list_h
remoteMemList(nixlMemViewH mvh) {
    return static_cast<ucp_device_remote_mem_list_h>(mvh);
}

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    const auto status =
        ucp_device_progress_req<ucsDeviceLevel<level>::value>(requestPtr(&xfer_status));

    switch (status) {
    case UCS_OK:
        return NIXL_SUCCESS;
    case UCS_INPROGRESS:
        return NIXL_IN_PROG;
    default:
        return NIXL_ERR_BACKEND;
    }
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    auto src_mem_list = localMemList(src.mvh);
    auto dst_mem_list = remoteMemList(dst.mvh);
    const auto status = ucp_device_put<ucsDeviceLevel<level>::value>(src_mem_list,
                                                                     src.index,
                                                                     src.offset,
                                                                     dst_mem_list,
                                                                     dst.index,
                                                                     dst.offset,
                                                                     size,
                                                                     channel_id,
                                                                     toUcpFlags(flags),
                                                                     requestPtr(xfer_status));
    return convertStatus(status);
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t flags,
          xferStatusH *xfer_status) {
    auto mem_list = remoteMemList(counter.mvh);
    const auto status =
        ucp_device_counter_inc<ucsDeviceLevel<level>::value>(value,
                                                             mem_list,
                                                             counter.index,
                                                             counter.offset,
                                                             channel_id,
                                                             toUcpFlags(flags),
                                                             requestPtr(xfer_status));
    return convertStatus(status);
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    auto mem_list = remoteMemList(mvh);
    void *ptr = nullptr;
    ucp_device_get_ptr(mem_list, index, &ptr);
    return ptr;
}

} // namespace nixl::gpu::impl::ucx

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_UCX_DEVICE_UCX_CUH
