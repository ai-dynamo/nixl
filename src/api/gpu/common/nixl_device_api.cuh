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
#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH

#include <cstring>

#include <gpu/nixl_device_config.h>

#include "nixl_device_types.cuh"

#include "../proxy/nixl_device_impl.cuh"

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
#include "../ucx/nixl_device_impl.cuh"
#else
namespace nixl::gpu::ucx_impl {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &,
    const nixlMemViewElem &,
    size_t,
    unsigned = 0,
    uint64_t = 0,
    nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t,
           const nixlMemViewElem &,
           unsigned = 0,
           uint64_t = 0,
           nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
get_ptr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::ucx_impl
#endif

namespace nixl::gpu::api {
namespace detail {

    __device__ __forceinline__ const nixlDeviceMemViewWrapper *
    as_device_memview(nixlMemViewH handle) {
        return static_cast<const nixlDeviceMemViewWrapper *>(handle);
    }

    template<nixl_gpu_level_t level>
    __device__ __forceinline__ bool
    execution_leader() {
        if constexpr (level == nixl_gpu_level_t::THREAD) {
            return true;
        } else if constexpr (level == nixl_gpu_level_t::WARP) {
            return threadIdx.x % warpSize == 0;
        } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
            return threadIdx.x == 0;
        } else if constexpr (level == nixl_gpu_level_t::GRID) {
            return blockIdx.x == 0 && threadIdx.x == 0;
        }
    }

    __device__ __forceinline__ nixl_device_exec_mode_t
    load_execution_mode(const nixlGpuXferStatusH &status) {
        uint32_t execution_mode = 0;
        memcpy(&execution_mode,
               status.storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE,
               sizeof(execution_mode));
        return static_cast<nixl_device_exec_mode_t>(execution_mode);
    }

    template<nixl_gpu_level_t level>
    __device__ __forceinline__ void
    write_execution_mode(nixlGpuXferStatusH *status,
                         nixl_status_t submission_status,
                         nixl_device_exec_mode_t execution_mode) {
        if (status == nullptr || submission_status != NIXL_IN_PROG || !execution_leader<level>()) {
            return;
        }
        const uint32_t mode = static_cast<uint32_t>(execution_mode);
        memcpy(status->storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE, &mode, sizeof(mode));
    }

} // namespace detail

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    switch (detail::load_execution_mode(xfer_status)) {
    case nixl_device_exec_mode_t::UCX_DIRECT:
        return ucx_impl::get_xfer_status<level>(xfer_status);
    case nixl_device_exec_mode_t::PROXY:
        return proxy_impl::get_xfer_status<level>(xfer_status);
    default:
        return NIXL_ERR_INVALID_PARAM;
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
    const auto *src_view = detail::as_device_memview(src.mvh);
    const auto *dst_view = detail::as_device_memview(dst.mvh);
    const nixlMemViewElem backend_src{src_view->backend_memview, src.index, src.offset};
    const nixlMemViewElem backend_dst{dst_view->backend_memview, dst.index, dst.offset};

    nixl_status_t status;
    if (dst_view->execution_mode == nixl_device_exec_mode_t::UCX_DIRECT) {
        status =
            ucx_impl::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    } else if (dst_view->execution_mode == nixl_device_exec_mode_t::PROXY) {
        status =
            proxy_impl::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    } else {
        status = NIXL_ERR_INVALID_PARAM;
    }
    detail::write_execution_mode<level>(xfer_status, status, dst_view->execution_mode);
    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    const auto *view = detail::as_device_memview(counter.mvh);
    const nixlMemViewElem backend_counter{view->backend_memview, counter.index, counter.offset};

    nixl_status_t status;
    if (view->execution_mode == nixl_device_exec_mode_t::UCX_DIRECT) {
        status =
            ucx_impl::atomic_add<level>(value, backend_counter, channel_id, flags, xfer_status);
    } else if (view->execution_mode == nixl_device_exec_mode_t::PROXY) {
        status =
            proxy_impl::atomic_add<level>(value, backend_counter, channel_id, flags, xfer_status);
    } else {
        status = NIXL_ERR_INVALID_PARAM;
    }
    detail::write_execution_mode<level>(xfer_status, status, view->execution_mode);
    return status;
}

__device__ inline void *
get_ptr(nixlMemViewH mvh, size_t index) {
    const auto *view = detail::as_device_memview(mvh);
    switch (view->execution_mode) {
    case nixl_device_exec_mode_t::UCX_DIRECT:
        return ucx_impl::get_ptr(view->backend_memview, index);
    case nixl_device_exec_mode_t::PROXY:
        return proxy_impl::get_ptr(view->backend_memview, index);
    default:
        return nullptr;
    }
}

} // namespace nixl::gpu::api

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
