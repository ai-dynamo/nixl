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

constexpr unsigned kWarpSize = 32; // Assumed equal to device warpSize (CUDA guarantee);

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchPollXferStatus(nixl_status_t status, nixlGpuXferStatusH &xfer_status) {
    while (status == NIXL_IN_PROG) {
        status = nixlGpuGetXferStatus<Level>(xfer_status);
    }
    return status;
}

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchPutLevel(const nixlbenchDeviceXferParams &params,
                  size_t region_idx,
                  nixlGpuXferStatusH &xfer_status) {
    const nixlMemViewElem src{params.localMvh, region_idx, 0};
    const nixlMemViewElem dst{params.remoteMvh, region_idx, 0};
    nixl_status_t status = nixlPut<Level>(src, dst, params.regionSize, 0, 0, &xfer_status);
    status = nixlbenchPollXferStatus<Level>(status, xfer_status);

    if (status != NIXL_SUCCESS) {
        printf("[nixlbenchPutLevel] transfer did not complete: region=%zu "
               "threadIdx.x=%u blockIdx.x=%u blockDim.x=%u final_status=%d\n",
               region_idx,
               threadIdx.x,
               blockIdx.x,
               blockDim.x,
               static_cast<int>(status));
    }
    return status;
}

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchSignalCounter(const nixlbenchDeviceXferParams &params,
                       size_t counter_offset,
                       uint64_t value,
                       const char *counter_name) {
    const nixlMemViewElem counter{params.remoteMvh, params.numRegions, counter_offset};
    nixlGpuXferStatusH xfer_status;
    nixl_status_t status = nixlAtomicAdd<Level>(value, counter, 0, 0, &xfer_status);
    status = nixlbenchPollXferStatus<Level>(status, xfer_status);

    if (status != NIXL_SUCCESS) {
        printf("[nixlbenchSignalCounter] nixlAtomicAdd(%s) did not complete: final_status=%d\n",
               counter_name,
               static_cast<int>(status));
    }
    return status;
}

__device__ nixl_status_t
nixlbenchSignalCompletion(nixlbenchDeviceXferParams params) {
    return nixlbenchSignalCounter<nixl_gpu_level_t::THREAD>(
        params, params.completionCounterOffsetBytes, 1ull, "completion");
}

__device__ nixl_status_t
nixlbenchSignalError(nixlbenchDeviceXferParams params) {
    return nixlbenchSignalCounter<nixl_gpu_level_t::THREAD>(
        params, params.errorCounterOffsetBytes, 1ull, "error");
}

/**
 * Performs device-initiated NIXL PUT transfers and optionally reports completion or errors
 * through remote counters.
 */
template<nixl_gpu_level_t Level>
__global__ void
nixlbenchPutKernel(nixlbenchDeviceXferParams params) {
    unsigned group_id, num_groups;
    if constexpr (Level == nixl_gpu_level_t::THREAD) {
        group_id = threadIdx.x;
        num_groups = blockDim.x;
    } else { // CUDA warpSize == kWarpSize == 32
        group_id = threadIdx.x / warpSize;
        num_groups = (blockDim.x + warpSize - 1) / warpSize;
    }

    nixlGpuXferStatusH xfer_status;
    nixl_status_t put_status = NIXL_SUCCESS;
    for (size_t region_idx = group_id; region_idx < params.numRegions; region_idx += num_groups) {
        put_status = nixlbenchPutLevel<Level>(params, region_idx, xfer_status);
        if (put_status != NIXL_SUCCESS) {
            break;
        }
    }

    const bool any_put_failed = __syncthreads_or(put_status != NIXL_SUCCESS);
    if (threadIdx.x == 0) {
        if (any_put_failed) {
            if (nixlbenchSignalError(params) != NIXL_SUCCESS) {
                printf("[nixlbenchPutKernel] error nixlAtomicAdd failed\n");
            }
            return;
        }

        if (nixlbenchSignalCompletion(params) != NIXL_SUCCESS) {
            printf("[nixlbenchPutKernel] completion nixlAtomicAdd failed\n");
            if (nixlbenchSignalError(params) != NIXL_SUCCESS) {
                printf("[nixlbenchPutKernel] error nixlAtomicAdd failed\n");
            }
        }
    }
}

} // namespace

nixl_status_t
nixlbenchLaunchDevicePut(const nixlbenchDeviceXferParams &params, unsigned block_threads) {
    if (params.localMvh == nullptr || params.remoteMvh == nullptr) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: valid local and remote memory views "
                     "are required\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!params.signalRemoteCompletion) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: remote completion signaling is "
                     "required\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (block_threads == 0 || block_threads > 1024u) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: invalid block_threads=" << block_threads
                  << " (must be 1..1024)\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (block_threads <= kWarpSize) {
        nixlbenchPutKernel<nixl_gpu_level_t::THREAD><<<1, block_threads, 0, nullptr>>>(params);
    } else {
        if (block_threads % kWarpSize != 0) {
            std::cerr << "nixlbench: nixlbenchLaunchDevicePut: block_threads (" << block_threads
                      << ") must be a multiple of " << kWarpSize
                      << " (WARP-level nixlPut requires full warps)\n";
            return NIXL_ERR_INVALID_PARAM;
        }
        nixlbenchPutKernel<nixl_gpu_level_t::WARP><<<1, block_threads, 0, nullptr>>>(params);
    }

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: cudaGetLastError after launch: "
                  << cudaGetErrorString(cuda_err) << '\n';
        return NIXL_ERR_BACKEND;
    }

    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: synchronize failed: "
                  << cudaGetErrorString(cuda_err) << '\n';
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
