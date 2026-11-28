// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if defined(__clang__) && defined(__CUDA__)
// CuTe 4.5.1 libNVVM cannot lower Clang's newer block-barrier intrinsic.
// __syncthreads() is a full CTA barrier on barrier ID zero.
static __device__ __forceinline__ __attribute__((convergent)) void
cuteNixlBlockBarrier() {
    asm volatile("bar.sync 0;" ::: "memory");
}

#pragma push_macro("__syncthreads")
#define __syncthreads() cuteNixlBlockBarrier()

// UCX's device qualifiers and CUDA-IPC specializations currently assume NVCC.
#pragma push_macro("__NVCC__")
#pragma push_macro("__builtin_ia32_prefetch")
#define __NVCC__
#include <ucs/sys/device_code.h>
#pragma pop_macro("__builtin_ia32_prefetch")
#pragma pop_macro("__NVCC__")
#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <ucs/type/status.h>
#include <cuda/atomic>
#pragma clang attribute push(__attribute__((device)), apply_to = function)
#include <uct/cuda/cuda_ipc/cuda_ipc.cuh>
#pragma clang attribute pop
#endif

#include <gpu/nixl_device.cuh>
#if defined(__clang__) && defined(__CUDA__)
#pragma pop_macro("__syncthreads")
#endif

// Source-matched private shims, not a stable ABI.
static_assert(sizeof(nixlGpuXferStatusH) == 64);
static_assert(alignof(nixlGpuXferStatusH) == 16);
static_assert(NIXL_SUCCESS == 0 && NIXL_IN_PROG == 1);
static_assert(nixl_gpu_flags::defer == 1);
static_assert(static_cast<int>(nixl_gpu_level_t::THREAD) == 0);
static_assert(static_cast<int>(nixl_gpu_level_t::WARP) == 1);
static_assert(static_cast<int>(nixl_gpu_level_t::BLOCK) == 2);

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_put(uint64_t local,
              uint64_t remote,
              uint64_t size,
              uint64_t local_index,
              uint64_t remote_index,
              uint64_t local_offset,
              uint64_t remote_offset,
              unsigned channel,
              uint64_t flags,
              uint64_t request,
              int level) {
    const nixlMemViewElem src{reinterpret_cast<nixlMemViewH>(local), local_index, local_offset};
    const nixlMemViewElem dst{reinterpret_cast<nixlMemViewH>(remote), remote_index, remote_offset};
    auto *status = reinterpret_cast<nixlGpuXferStatusH *>(request);
    switch (level) {
    case 0:
        return nixlPut<nixl_gpu_level_t::THREAD, nixl_device_exec_mode_t::DYNAMIC>(
            src, dst, size, channel, flags, status);
    case 1:
        return nixlPut<nixl_gpu_level_t::WARP, nixl_device_exec_mode_t::DYNAMIC>(
            src, dst, size, channel, flags, status);
    case 2:
        return nixlPut<nixl_gpu_level_t::BLOCK, nixl_device_exec_mode_t::DYNAMIC>(
            src, dst, size, channel, flags, status);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_atomic_add(uint64_t value,
                     uint64_t remote,
                     uint64_t index,
                     uint64_t offset,
                     unsigned channel,
                     uint64_t flags,
                     uint64_t request,
                     int level) {
    const nixlMemViewElem counter{reinterpret_cast<nixlMemViewH>(remote), index, offset};
    auto *status = reinterpret_cast<nixlGpuXferStatusH *>(request);
    switch (level) {
    case 0:
        return nixlAtomicAdd<nixl_gpu_level_t::THREAD, nixl_device_exec_mode_t::DYNAMIC>(
            value, counter, channel, flags, status);
    case 1:
        return nixlAtomicAdd<nixl_gpu_level_t::WARP, nixl_device_exec_mode_t::DYNAMIC>(
            value, counter, channel, flags, status);
    case 2:
        return nixlAtomicAdd<nixl_gpu_level_t::BLOCK, nixl_device_exec_mode_t::DYNAMIC>(
            value, counter, channel, flags, status);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_flush(uint64_t remote, unsigned channel, int level) {
    using namespace nixl::gpu;
    auto view = reinterpret_cast<nixlMemViewH>(remote);
    switch (level) {
    case 0:
        return flush<level_t::THREAD, exec_mode_t::DYNAMIC>(view, channel);
    case 1:
        return flush<level_t::WARP, exec_mode_t::DYNAMIC>(view, channel);
    case 2:
        return flush<level_t::BLOCK, exec_mode_t::DYNAMIC>(view, channel);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

extern "C" __device__ __attribute__((always_inline)) int
cute_nixl_get_xfer_status(uint64_t request, int level) {
    if (request == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto &status = *reinterpret_cast<nixlGpuXferStatusH *>(request);
    switch (level) {
    case 0:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD, nixl_device_exec_mode_t::DYNAMIC>(
            status);
    case 1:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::WARP, nixl_device_exec_mode_t::DYNAMIC>(
            status);
    case 2:
        return nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK, nixl_device_exec_mode_t::DYNAMIC>(
            status);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}
