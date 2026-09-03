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

#include <gpu/device_types.cuh>
#include <gpu/impl/level_utils.cuh>

#if __has_include(<ucp/api/device/ucp_device_impl.h>)
#include <gpu/impl/ucx/device_ucx.cuh>
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

// The proxy implementation needs only CUDA, so unlike the UCX arm it is always
// present and needs no capability gate.
#include "proxy/device_proxy.cuh"

namespace nixl::gpu::impl {

namespace detail {

    __device__ __forceinline__ const memViewWrapper *
    asDeviceMemView(nixlMemViewH handle) {
        return static_cast<const memViewWrapper *>(handle);
    }

    __device__ __forceinline__ exec_mode_t
    loadExecutionMode(const xferStatusH &status) {
        uint32_t execution_mode = 0;
        memcpy(&execution_mode, status.storage + xfer_status_payload_size, sizeof(execution_mode));
        return static_cast<exec_mode_t>(execution_mode);
    }

    /** Tag accepted submissions, including inline success, for later dispatch by getXferStatus. */
    template<level_t level>
    __device__ __forceinline__ void
    writeExecutionMode(xferStatusH *status,
                       nixl_status_t submission_status,
                       exec_mode_t execution_mode) {
        if (status != nullptr && submission_status >= 0 && isLeader<level>()) {
            const uint32_t mode = static_cast<uint32_t>(execution_mode);
            memcpy(status->storage + xfer_status_payload_size, &mode, sizeof(mode));
        }
        // Collective pollers share the leader's tag; a grid-level call is not a barrier here.
        if constexpr (level != level_t::GRID) {
            sync<level>();
        }
    }

} // namespace detail

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    switch (detail::loadExecutionMode(xfer_status)) {
    case exec_mode_t::UCX_DIRECT:
        return ucx::getXferStatus<level>(xfer_status);
    case exec_mode_t::PROXY:
        return proxy::getXferStatus<level>(xfer_status);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    const auto *src_view = detail::asDeviceMemView(src.mvh);
    const auto *dst_view = detail::asDeviceMemView(dst.mvh);
    const memViewElem backend_src{src_view->backend_memview, src.index, src.offset};
    const memViewElem backend_dst{dst_view->backend_memview, dst.index, dst.offset};

    // The destination decides: it is the view that names the peer.
    nixl_status_t status;
    if (dst_view->execution_mode == exec_mode_t::UCX_DIRECT) {
        status = ucx::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    } else if (dst_view->execution_mode == exec_mode_t::PROXY) {
        status = proxy::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    } else {
        status = NIXL_ERR_INVALID_PARAM;
    }
    detail::writeExecutionMode<level>(xfer_status, status, dst_view->execution_mode);
    return status;
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t flags,
          xferStatusH *xfer_status) {
    const auto *view = detail::asDeviceMemView(counter.mvh);
    const memViewElem backend_counter{view->backend_memview, counter.index, counter.offset};

    nixl_status_t status;
    if (view->execution_mode == exec_mode_t::UCX_DIRECT) {
        status = ucx::atomicAdd<level>(value, backend_counter, channel_id, flags, xfer_status);
    } else if (view->execution_mode == exec_mode_t::PROXY) {
        status = proxy::atomicAdd<level>(value, backend_counter, channel_id, flags, xfer_status);
    } else {
        status = NIXL_ERR_INVALID_PARAM;
    }
    detail::writeExecutionMode<level>(xfer_status, status, view->execution_mode);
    return status;
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    const auto *view = detail::asDeviceMemView(mvh);
    switch (view->execution_mode) {
    case exec_mode_t::UCX_DIRECT:
        return ucx::getPtr(view->backend_memview, index);
    case exec_mode_t::PROXY:
        return proxy::getPtr(view->backend_memview, index);
    default:
        return nullptr;
    }
}

} // namespace nixl::gpu::impl

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH
