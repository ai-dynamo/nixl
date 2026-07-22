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

// Public __device__ wrapper functions for the NIXL GPU transfer API.
//
// This file must be included AFTER nixl::gpu::selected_impl has been aliased
// to the desired backend namespace (ucx_impl, proxy_impl, etc.).  It is
// included automatically by each backend-specific nixl_device.cuh facade and
// by the generic GPU API header; callers should not include it directly.

#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_WRAPPERS_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_WRAPPERS_CUH

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    return nixl::gpu::selected_impl::get_xfer_status<level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::selected_impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::selected_impl::atomic_add<level>(
        value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    return nixl::gpu::selected_impl::get_ptr(mvh, index);
}

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_WRAPPERS_CUH
