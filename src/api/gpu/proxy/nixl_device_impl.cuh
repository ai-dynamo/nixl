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
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);

    ProxyDeviceContext *ctx = load_proxy_context();

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        if (ctx == nullptr) {
            status = NIXL_ERR_BACKEND;
        } else {
            status = ctx->pollXferStatus(xfer_status);
        }
    }

    if constexpr (level == nixl_gpu_level_t::WARP) {
        status = static_cast<nixl_status_t>(
            __shfl_sync(0xffffffff, static_cast<int>(status), 0));
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
__device__ inline nixl_status_t
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
        ProxyDeviceContext *ctx = load_proxy_context();
        if (ctx == nullptr) {
            status = NIXL_ERR_BACKEND;
        } else {
            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::PUT;
            submission.channel_id = static_cast<uint32_t>(channel_id);
            submission.flags = flags;
            submission.src_proxy_memview_id = proxyMemViewIdFromHandle(src.mvh);
            submission.src_index = src.index;
            submission.src_offset = src.offset;
            submission.dst_proxy_memview_id = proxyMemViewIdFromHandle(dst.mvh);
            submission.dst_index = dst.index;
            submission.dst_offset = dst.offset;
            submission.size = size;
            status = ctx->enqueue(submission, xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        ProxyDeviceContext *ctx = load_proxy_context();
        if (ctx == nullptr) {
            status = NIXL_ERR_BACKEND;
        } else {
            nixlProxySubmission submission{};
            submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
            submission.channel_id = static_cast<uint32_t>(channel_id);
            submission.flags = flags;
            submission.dst_proxy_memview_id = proxyMemViewIdFromHandle(counter.mvh);
            submission.dst_index = counter.index;
            submission.dst_offset = counter.offset;
            submission.value = value;
            status = ctx->enqueue(submission, xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

__device__ inline void *
get_ptr(nixlMemViewH, size_t) {
    // TODO: Implement support for NVLink fast-path over proxy - NIX-1342
    return nullptr;
}

} // namespace nixl::gpu::proxy_impl

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
