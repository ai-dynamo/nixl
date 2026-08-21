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
#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH

#include "nixl_device_proxy.cuh"
#include "nixl_types.h"

namespace nixl::gpu::proxy_impl {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
getXferStatus(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        status = ProxyDeviceContext::pollXferStatus(xfer_status);
    }

    if constexpr (level == nixl_gpu_level_t::WARP) {
        status = static_cast<nixl_status_t>(__shfl_sync(0xffffffff, static_cast<int>(status), 0));
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        __shared__ nixl_status_t s_status;
        if (threadIdx.x == 0) {
            s_status = status;
        }
        __syncthreads();
        status = s_status;
        __syncthreads();
    }

    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        const auto *src_memview = static_cast<const nixlProxyDeviceMemView *>(src.mvh);
        const auto *dst_memview = static_cast<const nixlProxyDeviceMemView *>(dst.mvh);
        const auto *ctx = reinterpret_cast<const ProxyDeviceContext *>(dst_memview->context);
        if (ctx == nullptr || src_memview->context != dst_memview->context) {
            status = NIXL_ERR_INVALID_PARAM;
        } else {
            status = ctx->enqueue(
                nixlProxySubmission{.srcOffset = static_cast<uint64_t>(src.offset),
                                    .dstOffset = static_cast<uint64_t>(dst.offset),
                                    .size = static_cast<uint64_t>(size),
                                    .opcode = nixl_proxy_opcode_t::PUT,
                                    .flags = static_cast<uint8_t>(flags),
                                    .channelId = static_cast<uint16_t>(channel_id),
                                    .srcIndex = static_cast<uint32_t>(src.index),
                                    .dstIndex = static_cast<uint32_t>(dst.index),
                                    .srcProxyMemViewId = proxyMemViewIdFromHandle(src.mvh),
                                    .dstProxyMemViewId = proxyMemViewIdFromHandle(dst.mvh)},
                xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
atomicAdd(uint64_t value,
          const nixlMemViewElem &counter,
          unsigned channel_id = 0,
          uint64_t flags = 0,
          nixlGpuXferStatusH *xfer_status = nullptr) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        const auto *memview = static_cast<const nixlProxyDeviceMemView *>(counter.mvh);
        const auto *ctx = reinterpret_cast<const ProxyDeviceContext *>(memview->context);
        if (ctx == nullptr) {
            status = NIXL_ERR_INVALID_PARAM;
        } else {
            status = ctx->enqueue(
                nixlProxySubmission{.value = value,
                                    .dstOffset = static_cast<uint64_t>(counter.offset),
                                    .size = static_cast<uint64_t>(sizeof(uint64_t)),
                                    .opcode = nixl_proxy_opcode_t::ATOMIC_ADD,
                                    .flags = static_cast<uint8_t>(flags),
                                    .channelId =
                                        static_cast<uint16_t>(channel_id % ctx->numChannels),
                                    .dstIndex = static_cast<uint32_t>(counter.index),
                                    .dstProxyMemViewId = proxyMemViewIdFromHandle(counter.mvh)},
                xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

__device__ __forceinline__ void *
getPtr(nixlMemViewH mvh, size_t index) {
    if (mvh == nullptr) {
        return nullptr;
    }

    const auto *memview = static_cast<const nixlProxyDeviceMemView *>(mvh);
    if (index >= memview->directPtrCount) {
        return nullptr;
    }

    return memview->directPtrs[index];
}

} // namespace nixl::gpu::proxy_impl

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
