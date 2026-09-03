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
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_CUH

#include "device_proxy_ring.cuh"
#include <nixl_types.h>

namespace nixl::gpu::impl::proxy {

template<level_t level>
__device__ __forceinline__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {

    nixl_status_t status = NIXL_IN_PROG;
    if (isLeader<level>()) {
        status = ProxyDeviceContext::pollXferStatus(xfer_status);
    }

    if constexpr (level == level_t::WARP) {
        status = static_cast<nixl_status_t>(__shfl_sync(0xffffffff, static_cast<int>(status), 0));
    } else if constexpr (level == level_t::BLOCK) {
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

template<level_t level>
__device__ __forceinline__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    nixl_status_t status = NIXL_IN_PROG;
    if (isLeader<level>()) {
        static_cast<void>(flags);
        const auto *src_memview = static_cast<const nixlProxyDeviceMemView *>(src.mvh);
        const auto *dst_memview = static_cast<const nixlProxyDeviceMemView *>(dst.mvh);
        const auto *ctx = proxyContextFromMemView(dst_memview);
        if (ctx == nullptr || src_memview->context != dst_memview->context) {
            status = NIXL_ERR_INVALID_PARAM;
        } else {
            status = ctx->enqueue(nixlProxyCommand{.operand = static_cast<uint64_t>(src.offset),
                                                   .dst_offset = static_cast<uint64_t>(dst.offset),
                                                   .size = static_cast<uint64_t>(size),
                                                   .src_view = proxyHostViewFromHandle(src.mvh),
                                                   .dst_view = proxyHostViewFromHandle(dst.mvh),
                                                   .src_index = static_cast<uint32_t>(src.index),
                                                   .dst_index = static_cast<uint32_t>(dst.index),
                                                   .opcode = nixl_proxy_opcode_t::PUT},
                                  channel_id,
                                  xfer_status);
        }
    }
    sync<level>();
    return status;
}

template<level_t level>
__device__ __forceinline__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t flags,
          xferStatusH *xfer_status) {
    nixl_status_t status = NIXL_IN_PROG;
    if (isLeader<level>()) {
        static_cast<void>(flags);
        const auto *memview = static_cast<const nixlProxyDeviceMemView *>(counter.mvh);
        const auto *ctx = proxyContextFromMemView(memview);
        if (ctx == nullptr) {
            status = NIXL_ERR_INVALID_PARAM;
        } else {
            status =
                ctx->enqueue(nixlProxyCommand{.operand = value,
                                              .dst_offset = static_cast<uint64_t>(counter.offset),
                                              .dst_view = proxyHostViewFromHandle(counter.mvh),
                                              .dst_index = static_cast<uint32_t>(counter.index),
                                              .opcode = nixl_proxy_opcode_t::ATOMIC_ADD},
                             channel_id,
                             xfer_status);
        }
    }
    sync<level>();
    return status;
}

__device__ __forceinline__ void *
getPtr(nixlMemViewH mvh, size_t index) {
    if (mvh == nullptr) {
        return nullptr;
    }

    const auto *memview = static_cast<const nixlProxyDeviceMemView *>(mvh);
    if (proxyContextFromMemView(memview) == nullptr || index >= memview->direct_ptr_count) {
        return nullptr;
    }

    return nixlProxyDeviceMemViewDirectPtrs(memview)[index];
}

} // namespace nixl::gpu::impl::proxy

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_CUH
