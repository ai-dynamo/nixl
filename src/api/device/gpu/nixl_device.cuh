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

// Generic entry point for the GPU Device API.

#ifndef NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
#define NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH

#include <gpu/impl/device_dispatch.cuh>

namespace nixl::gpu {

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    return impl::getXferStatus<level>(xfer_status);
}

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    xferStatusH *xfer_status = nullptr) {
    return impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id = 0,
          uint64_t flags = 0,
          xferStatusH *xfer_status = nullptr) {
    return impl::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    return impl::getPtr(mvh, index);
}

} // namespace nixl::gpu

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    return nixl::gpu::getXferStatus<level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    return nixl::gpu::getPtr(mvh, index);
}

#endif // NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
