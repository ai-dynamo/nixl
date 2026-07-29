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

#include <cstring>

#include <nixl_device_config.h>

#include "nixl_device_types.cuh"
#include "../proxy/nixl_device_impl.cuh"

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
#include "../ucx/nixl_device_impl.cuh"
#else
namespace nixl::gpu::ucx_impl {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &,
    const nixlMemViewElem &,
    size_t,
    unsigned = 0,
    uint64_t = 0,
    nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t,
           const nixlMemViewElem &,
           unsigned = 0,
           uint64_t = 0,
           nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
get_ptr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::ucx_impl
#endif

namespace nixl::gpu::api {
namespace detail {

enum class MemViewBackend : uint8_t {
    INVALID,
    UCX,
    PROXY,
    UNSUPPORTED,
};

__device__ __forceinline__ MemViewBackend
memview_backend(nixlMemViewH handle) {
    if (handle == nullptr) {
        return MemViewBackend::INVALID;
    }
    const uint16_t version = *static_cast<const uint16_t *>(handle);
    if ((version & NIXL_PROXY_MEM_LIST_NAMESPACE) == 0) {
        return MemViewBackend::UCX;
    }
    return version == NIXL_PROXY_MEM_LIST_VERSION_V1 ? MemViewBackend::PROXY :
                                                       MemViewBackend::UNSUPPORTED;
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ bool
execution_leader() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return true;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        return threadIdx.x % warpSize == 0;
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        return threadIdx.x == 0;
    } else {
        return blockIdx.x == 0 && threadIdx.x == 0;
    }
}

__device__ __forceinline__ nixlDeviceXferStatusFooter
load_footer(const nixlGpuXferStatusH &status) {
    nixlDeviceXferStatusFooter footer{};
    memcpy(&footer, status.storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE, sizeof(footer));
    return footer;
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ void
write_footer(nixlGpuXferStatusH *status,
             nixl_status_t submission_status,
             nixlDeviceXferStatusBackend backend) {
    if (status == nullptr || submission_status != NIXL_IN_PROG || !execution_leader<level>()) {
        return;
    }
    const nixlDeviceXferStatusFooter footer{
        NIXL_DEVICE_XFER_STATUS_MAGIC,
        NIXL_DEVICE_XFER_STATUS_ABI_VERSION,
        static_cast<uint8_t>(backend),
        0,
    };
    memcpy(status->storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE, &footer, sizeof(footer));
}

__device__ __forceinline__ nixl_status_t
validate_proxy_put(const nixlMemViewElem &src,
                   const nixlMemViewElem &dst,
                   const nixlProxyDeviceMemView *&dst_memview) {
    const auto *src_memview = static_cast<const nixlProxyDeviceMemView *>(src.mvh);
    dst_memview = static_cast<const nixlProxyDeviceMemView *>(dst.mvh);
    if (src_memview->kind != nixlProxyMemViewKind::LOCAL ||
        dst_memview->kind != nixlProxyMemViewKind::REMOTE ||
        src.index >= src_memview->length || dst.index >= dst_memview->length ||
        src_memview->context.shutdown_word == nullptr ||
        src_memview->context.shutdown_word != dst_memview->context.shutdown_word) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return NIXL_SUCCESS;
}

__device__ __forceinline__ nixl_status_t
validate_proxy_counter(const nixlMemViewElem &counter,
                       const nixlProxyDeviceMemView *&memview) {
    memview = static_cast<const nixlProxyDeviceMemView *>(counter.mvh);
    if (memview->kind != nixlProxyMemViewKind::REMOTE || counter.index >= memview->length) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return NIXL_SUCCESS;
}

} // namespace detail

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    const auto footer = detail::load_footer(xfer_status);
    if (footer.magic != NIXL_DEVICE_XFER_STATUS_MAGIC ||
        footer.abi_version != NIXL_DEVICE_XFER_STATUS_ABI_VERSION || footer.reserved != 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    switch (static_cast<nixlDeviceXferStatusBackend>(footer.backend)) {
    case nixlDeviceXferStatusBackend::UCX:
        return ucx_impl::get_xfer_status<level>(xfer_status);
    case nixlDeviceXferStatusBackend::PROXY:
        if constexpr (level == nixl_gpu_level_t::GRID) {
            return NIXL_ERR_NOT_SUPPORTED;
        } else {
            return proxy_impl::get_xfer_status<level>(xfer_status);
        }
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    const auto dst_backend = detail::memview_backend(dst.mvh);
    const auto src_backend = detail::memview_backend(src.mvh);
    if (dst_backend == detail::MemViewBackend::UNSUPPORTED ||
        src_backend == detail::MemViewBackend::UNSUPPORTED) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (dst_backend == detail::MemViewBackend::INVALID ||
        src_backend == detail::MemViewBackend::INVALID || dst_backend != src_backend) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t status;
    nixlDeviceXferStatusBackend status_backend;
    if (dst_backend == detail::MemViewBackend::UCX) {
        status = ucx_impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
        status_backend = nixlDeviceXferStatusBackend::UCX;
    } else {
        const nixlProxyDeviceMemView *dst_memview = nullptr;
        status = detail::validate_proxy_put(src, dst, dst_memview);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        if constexpr (level == nixl_gpu_level_t::GRID) {
            return NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = proxy_impl::put<level>(
                dst_memview->context, src, dst, size, channel_id, flags, xfer_status);
        }
        status_backend = nixlDeviceXferStatusBackend::PROXY;
    }
    detail::write_footer<level>(xfer_status, status, status_backend);
    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    const auto backend = detail::memview_backend(counter.mvh);
    if (backend == detail::MemViewBackend::UNSUPPORTED) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (backend == detail::MemViewBackend::INVALID) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t status;
    nixlDeviceXferStatusBackend status_backend;
    if (backend == detail::MemViewBackend::UCX) {
        status =
            ucx_impl::atomic_add<level>(value, counter, channel_id, flags, xfer_status);
        status_backend = nixlDeviceXferStatusBackend::UCX;
    } else {
        const nixlProxyDeviceMemView *memview = nullptr;
        status = detail::validate_proxy_counter(counter, memview);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        if constexpr (level == nixl_gpu_level_t::GRID) {
            return NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = proxy_impl::atomic_add<level>(
                memview->context, value, counter, channel_id, flags, xfer_status);
        }
        status_backend = nixlDeviceXferStatusBackend::PROXY;
    }
    detail::write_footer<level>(xfer_status, status, status_backend);
    return status;
}

__device__ inline void *
get_ptr(nixlMemViewH mvh, size_t index) {
    switch (detail::memview_backend(mvh)) {
    case detail::MemViewBackend::UCX:
        return ucx_impl::get_ptr(mvh, index);
    case detail::MemViewBackend::PROXY:
        return proxy_impl::get_ptr(mvh, index);
    default:
        return nullptr;
    }
}

} // namespace nixl::gpu::api

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
