/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU-side nixlPut after host-side prepMemView (see xferBenchNixlWorker::prepareGPULocalView
 * and prepareGPURemoteView in nixl_worker.cpp).
 *
 * Flattening order (must match prepMemView):
 *   for each inner list in local_trans_lists / remote_trans_lists:
 *     for each IOV:
 *       addDesc -> consecutive indices 0 .. numRegions-1
 */

#include "nixlbench_device_launch.cuh"

#include <gpu/ucx/nixl_device.cuh>

#include <cstdio>
#include <iostream>

namespace {

/** Lane-0 xfer status slots per warp when blockDim.x <= 1024 (32 warps). */
constexpr unsigned nixlbench_max_warps = 32u;

template<nixl_gpu_level_t Level>
__device__ bool
nixlbenchPutLevel(nixlbenchDeviceXferParams params,
                  size_t region_idx,
                  nixlGpuXferStatusH &xfer_status) {
    const nixlMemViewElem src{params.localMvh, region_idx, 0};
    const nixlMemViewElem dst{params.remoteMvh, region_idx, 0};
    nixl_status_t status = nixlPut<Level>(src, dst, params.regionSize, 0, 0, &xfer_status);
    if (status != NIXL_IN_PROG) {
        printf("[nixlbenchPutLevel] nixlPut did not return NIXL_IN_PROG: "
               "region=%zu threadIdx.x=%u blockIdx.x=%u blockDim.x=%u status=%d\n",
               region_idx,
               threadIdx.x,
               blockIdx.x,
               blockDim.x,
               static_cast<int>(status));
        return false;
    }

    do {
        status = nixlGpuGetXferStatus<Level>(xfer_status);
    } while (status == NIXL_IN_PROG);

    if (status != NIXL_SUCCESS) {
        printf("[nixlbenchPutLevel] transfer did not complete: region=%zu "
               "threadIdx.x=%u blockIdx.x=%u blockDim.x=%u final_status=%d\n",
               region_idx,
               threadIdx.x,
               blockIdx.x,
               blockDim.x,
               static_cast<int>(status));
        return false;
    }

    return true;
}

template<nixl_gpu_level_t Level>
__device__ bool
nixlbenchSignalCounter(nixlbenchDeviceXferParams params,
                       size_t counter_offset,
                       uint64_t value,
                       const char *counter_name) {
    if (!params.signalRemoteCompletion || params.remoteMvh == nullptr) {
        return true;
    }
    const nixlMemViewElem counter{params.remoteMvh, params.numRegions, counter_offset};
    nixlGpuXferStatusH xfer_status{};
    nixl_status_t status = nixlAtomicAdd<Level>(value, counter, 0, 0, &xfer_status);
    if (status != NIXL_IN_PROG) {
        printf(
            "[nixlbenchSignalCounter] nixlAtomicAdd(%s) did not return NIXL_IN_PROG: status=%d\n",
            counter_name,
            static_cast<int>(status));
        return false;
    }
    do {
        status = nixlGpuGetXferStatus<Level>(xfer_status);
    } while (status == NIXL_IN_PROG);
    if (status != NIXL_SUCCESS) {
        printf("[nixlbenchSignalCounter] nixlAtomicAdd(%s) did not complete: final_status=%d\n",
               counter_name,
               static_cast<int>(status));
        return false;
    }
    return true;
}

template<nixl_gpu_level_t Level>
__device__ bool
nixlbenchSignalCompletion(nixlbenchDeviceXferParams params) {
    return nixlbenchSignalCounter<Level>(
        params, params.completionCounterOffsetBytes, 1ull, "completion");
}

template<nixl_gpu_level_t Level>
__device__ bool
nixlbenchSignalError(nixlbenchDeviceXferParams params) {
    return nixlbenchSignalCounter<Level>(params, params.errorCounterOffsetBytes, 1ull, "error");
}

/**
 * If blockDim.x <= warpSize: THREAD-level nixlPut; each thread uses regions
 * region_idx = threadIdx.x, threadIdx.x + blockDim.x, ...
 * Else: WARP-level nixlPut; each warp strides by num_warps and all lanes in a warp participate.
 * After all puts complete, thread 0 increments done counter by 1. If any put fails, thread 0
 * increments error counter by 1.
 */
__global__ void
nixlbenchPutKernel(nixlbenchDeviceXferParams params) {
    __shared__ unsigned put_fail_count;
    if (threadIdx.x == 0) {
        put_fail_count = 0;
    }
    __syncthreads();

    const bool use_thread_level = blockDim.x <= static_cast<unsigned>(warpSize);
    const unsigned lane = use_thread_level ? 0 : (threadIdx.x % warpSize);
    const unsigned group_id = use_thread_level ? threadIdx.x : (threadIdx.x / warpSize);
    const unsigned num_groups =
        use_thread_level ? blockDim.x : ((blockDim.x + warpSize - 1) / warpSize);
    if (group_id >= num_groups || group_id >= nixlbench_max_warps) {
        printf("[nixlbenchPutKernel] group_id out of range: "
               "group_id=%u num_groups=%u max_warps=%u "
               "threadIdx.x=%u blockDim.x=%u\n",
               group_id,
               num_groups,
               nixlbench_max_warps,
               threadIdx.x,
               blockDim.x);
        if (lane == 0) {
            atomicAdd(&put_fail_count, 1u);
        }
    } else {
        nixlGpuXferStatusH xfer_status{};
        bool put_failed = false;
        for (size_t region_idx = group_id; region_idx < params.numRegions;
             region_idx += num_groups) {
            if (use_thread_level) {
                put_failed =
                    !nixlbenchPutLevel<nixl_gpu_level_t::THREAD>(params, region_idx, xfer_status);
            } else {
                put_failed =
                    !nixlbenchPutLevel<nixl_gpu_level_t::WARP>(params, region_idx, xfer_status);
            }
            if (put_failed) {
                break;
            }
        }

        if (use_thread_level) {
            if (put_failed && lane == 0) {
                atomicAdd(&put_fail_count, 1u);
            }
        } else if (__any_sync(__activemask(), put_failed) && lane == 0) {
            atomicAdd(&put_fail_count, 1u);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        if (put_fail_count > 0) {
            if (!nixlbenchSignalError<nixl_gpu_level_t::THREAD>(params)) {
                printf("[nixlbenchPutKernel] error nixlAtomicAdd failed\n");
            }
            return;
        }

        if (!nixlbenchSignalCompletion<nixl_gpu_level_t::THREAD>(params)) {
            printf("[nixlbenchPutKernel] completion nixlAtomicAdd failed\n");
            if (!nixlbenchSignalError<nixl_gpu_level_t::THREAD>(params)) {
                printf("[nixlbenchPutKernel] error nixlAtomicAdd failed\n");
            }
        }
    }
}

} // namespace

nixl_status_t
nixlbenchLaunchDevicePut(const nixlbenchDeviceXferParams &params,
                         unsigned block_threads,
                         cudaStream_t stream) {
    constexpr unsigned kWarpSize = 32;

    if (block_threads == 0 || block_threads > 1024u) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: invalid block_threads=" << block_threads
                  << " (must be 1..1024)\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (block_threads > kWarpSize && (block_threads % kWarpSize) != 0) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: block_threads (" << block_threads
                  << " ) must be <= or a multiple of " << kWarpSize
                  << " (WARP-level nixlPut requires full warps)\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlbenchPutKernel<<<1, block_threads, 0, stream>>>(params);

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: cudaGetLastError after launch: "
                  << cudaGetErrorString(cuda_err) << '\n';
        return NIXL_ERR_BACKEND;
    }

    if (stream == nullptr) {
        cuda_err = cudaDeviceSynchronize();
    } else {
        cuda_err = cudaStreamSynchronize(stream);
    }
    if (cuda_err != cudaSuccess) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: synchronize failed: "
                  << cudaGetErrorString(cuda_err) << '\n';
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
