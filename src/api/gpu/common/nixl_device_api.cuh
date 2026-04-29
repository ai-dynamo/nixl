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

#include "nixl_device_types.cuh"

#if defined(NIXL_GPU_DEVICE_BACKEND_PROXY)
#include "../proxy/nixl_device_impl.cuh"

namespace nixl::gpu {
namespace selected_impl = proxy_impl;
}
#elif defined(NIXL_GPU_DEVICE_BACKEND_UCX)
#include "../ucx/nixl_device_impl.cuh"

namespace nixl::gpu {
namespace selected_impl = ucx_impl;
}
#else
#error "No GPU device backend implementation selected"
#endif

namespace nixl::gpu::api {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    return selected_impl::get_xfer_status<level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    return selected_impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    return selected_impl::atomic_add<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
get_ptr(nixlMemViewH mvh, size_t index) {
    return selected_impl::get_ptr(mvh, index);
}

} // namespace nixl::gpu::api

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
