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
    if constexpr (level == level_t::GRID) {
        return proxyDeviceContext::pollXferStatus(xfer_status);
    }

    nixl_status_t status = NIXL_IN_PROG;
    if (isLeader<level>()) {
        status = proxyDeviceContext::pollXferStatus(xfer_status);
    }
    return broadcastStatus<level>(status);
}

__device__ __forceinline__ void *
getPtr(nixlMemViewH mvh, size_t index);

static __device__ nixlProxyCompletionSlot direct_completion_slot{};

__device__ __forceinline__ void
completeDirect(xferStatusH *xfer_status) {
    if (xfer_status != nullptr) {
        const proxyXferStatus pxs{&direct_completion_slot, 0};
        memcpy(xfer_status->storage, &pxs, sizeof(pxs));
    }
}

__device__ __forceinline__ void
putDirect(const char *src, char *dst, size_t size, xferStatusH *xfer_status) {
    memcpy(dst, src, size);
    __threadfence_system();
    completeDirect(xfer_status);
}

template<level_t level>
__device__ __forceinline__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t,
    xferStatusH *xfer_status) {
    nixl_status_t status = NIXL_IN_PROG;
    auto src_ptr = static_cast<const char *>(getPtr(src.mvh, src.index));
    auto dst_ptr = static_cast<char *>(getPtr(dst.mvh, dst.index));
    if (src_ptr != nullptr && dst_ptr != nullptr) {
        if (isLeader<level>()) {
            putDirect(src_ptr + src.offset, dst_ptr + dst.offset, size, xfer_status);
        }
    } else if (isLeader<level>()) {
        auto src_memview = static_cast<const nixlProxyDeviceMemView *>(src.mvh);
        auto dst_memview = static_cast<const nixlProxyDeviceMemView *>(dst.mvh);
        auto ctx = proxyContextFromMemView(dst_memview);
        assert(src_memview->context == dst_memview->context);
        status = ctx->enqueue(nixlProxyCommand{.operand = static_cast<uint64_t>(src.offset),
                                               .dst_offset = static_cast<uint64_t>(dst.offset),
                                               .size = static_cast<uint64_t>(size),
                                               .src_view = proxyHostViewFromHandle(src.mvh),
                                               .dst_view = proxyHostViewFromHandle(dst.mvh),
                                               .src_index = static_cast<uint32_t>(src.index),
                                               .dst_index = static_cast<uint32_t>(dst.index),
                                               .opcode = nixl_proxy_opcode_t::PUT},
                              nixlProxyDeviceMemViewPeer(dst_memview, dst.index),
                              channel_id,
                              xfer_status);
    }
    return broadcastStatus<level>(status);
}

template<level_t level>
__device__ __forceinline__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t,
          xferStatusH *xfer_status) {
    nixl_status_t status = NIXL_IN_PROG;
    auto counter_ptr = static_cast<char *>(getPtr(counter.mvh, counter.index));
    if (counter_ptr != nullptr) {
        if (isLeader<level>()) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(
                *reinterpret_cast<uint64_t *>(counter_ptr + counter.offset));
            ref.fetch_add(value, cuda::memory_order_release);
            completeDirect(xfer_status);
        }
    } else if (isLeader<level>()) {
        auto memview = static_cast<const nixlProxyDeviceMemView *>(counter.mvh);
        auto ctx = proxyContextFromMemView(memview);
        status = ctx->enqueue(nixlProxyCommand{.operand = value,
                                               .dst_offset = static_cast<uint64_t>(counter.offset),
                                               .dst_view = proxyHostViewFromHandle(counter.mvh),
                                               .dst_index = static_cast<uint32_t>(counter.index),
                                               .opcode = nixl_proxy_opcode_t::ATOMIC_ADD},
                              nixlProxyDeviceMemViewPeer(memview, counter.index),
                              channel_id,
                              xfer_status);
    }
    return broadcastStatus<level>(status);
}

__device__ __forceinline__ void *
getPtr(nixlMemViewH mvh, size_t index) {
    auto memview = static_cast<const nixlProxyDeviceMemView *>(mvh);
    assert(memview != nullptr && memview->context != nullptr);
    // A proxied view may have no direct pointer for an element; that is not an error.
    if (index >= memview->direct_ptr_count) {
        return nullptr;
    }

    return nixlProxyDeviceMemViewDirectPtrs(memview)[index];
}

} // namespace nixl::gpu::impl::proxy

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_PROXY_DEVICE_PROXY_CUH
