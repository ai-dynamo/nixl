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
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        status = nixlProxyPollXferStatus(xfer_status);
    }

    return nixlProxyBroadcastStatus<level>(status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
put(const nixlProxyDeviceContextData &context,
    const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {

    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        status = nixlProxyEnqueue(
            context,
            nixlProxySubmission{.src_offset = static_cast<uint64_t>(src.offset),
                                .dst_offset = static_cast<uint64_t>(dst.offset),
                                .size = static_cast<uint64_t>(size),
                                .opcode = nixl_proxy_opcode_t::PUT,
                                .flags = static_cast<uint8_t>(flags),
                                .channel_id = static_cast<uint16_t>(channel_id),
                                .src_index = static_cast<uint32_t>(src.index),
                                .dst_index = static_cast<uint32_t>(dst.index),
                                .src_proxy_memview_id = proxyMemViewIdFromHandle(src.mvh),
                                .dst_proxy_memview_id = proxyMemViewIdFromHandle(dst.mvh)},
            xfer_status);
    }
    return nixlProxyBroadcastStatus<level>(status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
atomic_add(const nixlProxyDeviceContextData &context,
           uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    uint32_t lane_id;
    nixlProxyExecInit<level>(lane_id);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        status = nixlProxyEnqueue(
            context,
            nixlProxySubmission{.value = value,
                                .dst_offset = static_cast<uint64_t>(counter.offset),
                                .size = static_cast<uint64_t>(sizeof(uint64_t)),
                                .opcode = nixl_proxy_opcode_t::ATOMIC_ADD,
                                .flags = static_cast<uint8_t>(flags),
                                .channel_id = static_cast<uint16_t>(channel_id),
                                .dst_index = static_cast<uint32_t>(counter.index),
                                .dst_proxy_memview_id = proxyMemViewIdFromHandle(counter.mvh)},
            xfer_status);
    }
    return nixlProxyBroadcastStatus<level>(status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    if (src.mvh == nullptr || dst.mvh == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto *src_memview = static_cast<const nixlProxyDeviceMemView *>(src.mvh);
    const auto *dst_memview = static_cast<const nixlProxyDeviceMemView *>(dst.mvh);
    if (src_memview->version != NIXL_PROXY_MEM_LIST_VERSION_V1 ||
        dst_memview->version != NIXL_PROXY_MEM_LIST_VERSION_V1 ||
        src_memview->kind != nixlProxyMemViewKind::LOCAL ||
        dst_memview->kind != nixlProxyMemViewKind::REMOTE ||
        src.index >= src_memview->length || dst.index >= dst_memview->length ||
        src_memview->context.shutdown_word != dst_memview->context.shutdown_word) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return put<level>(
        dst_memview->context, src, dst, size, channel_id, flags, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    if (counter.mvh == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto *memview = static_cast<const nixlProxyDeviceMemView *>(counter.mvh);
    if (memview->version != NIXL_PROXY_MEM_LIST_VERSION_V1 ||
        memview->kind != nixlProxyMemViewKind::REMOTE || counter.index >= memview->length) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return atomic_add<level>(
        memview->context, value, counter, channel_id, flags, xfer_status);
}

__device__ __forceinline__ void *
get_ptr(nixlMemViewH mvh, size_t index) {
    if (mvh == nullptr) {
        return nullptr;
    }

    const auto *memview = static_cast<const nixlProxyDeviceMemView *>(mvh);
    if (memview->version != NIXL_PROXY_MEM_LIST_VERSION_V1 ||
        memview->kind != nixlProxyMemViewKind::REMOTE || index >= memview->length) {
        return nullptr;
    }
    return memview->mem_elements[index].direct_ptr;
}

} // namespace nixl::gpu::proxy_impl

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
