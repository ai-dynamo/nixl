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
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH

#include "../device_types.cuh"

#if __has_include(<ucp/api/device/ucp_device_impl.h>)
#include "ucx/device_ucx.cuh"
#else
namespace nixl::gpu::impl::ucx {

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &, const memViewElem &, size_t, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t, const memViewElem &, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
getPtr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::ucx
#endif

namespace nixl::gpu::impl {

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    return ucx::getXferStatus<level>(xfer_status);
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    return ucx::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t flags,
          xferStatusH *xfer_status) {
    return ucx::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    return ucx::getPtr(mvh, index);
}

} // namespace nixl::gpu::impl

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH
