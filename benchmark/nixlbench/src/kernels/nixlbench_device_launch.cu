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
constexpr unsigned kMaxGroups = 32; // 32 threads or max 1024 / 32 warps per block

__device__ __forceinline__ uint64_t
nixlbenchGetTimeNs() {
    uint64_t global_timer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(global_timer));
    return global_timer;
}

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
nixlbenchPostPut(const nixlbenchDeviceXferParams &params,
                 size_t region_idx,
                 unsigned channel_id,
                 nixlGpuXferStatusH &xfer_status) {
    const nixlMemViewElem src{params.localMvh, region_idx, 0};
    const nixlMemViewElem dst{params.remoteMvh, region_idx, 0};
    return nixlPut<Level>(src, dst, params.regionSize, channel_id, 0, &xfer_status);
}

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchSignalCounter(const nixlbenchDeviceXferParams &params,
                       size_t counter_offset,
                       uint64_t value,
                       unsigned channel_id,
                       nixlGpuXferStatusH &xfer_status,
                       const char *counter_name) {
    const nixlMemViewElem counter{params.remoteMvh, params.counterIndex, counter_offset};
    nixl_status_t status = nixlAtomicAdd<Level>(value, counter, channel_id, 0, &xfer_status);
    status = nixlbenchPollXferStatus<Level>(status, xfer_status);

    if (status != NIXL_SUCCESS) {
        printf("[nixlbenchSignalCounter] nixlAtomicAdd(%s) did not complete: final_status=%d\n",
               counter_name,
               static_cast<int>(status));
    }
    return status;
}

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchSignalCompletion(const nixlbenchDeviceXferParams &params,
                          uint64_t num_iterations,
                          unsigned channel_id,
                          nixlGpuXferStatusH &xfer_status) {
    return nixlbenchSignalCounter<Level>(params,
                                         params.completionCounterOffsetBytes,
                                         num_iterations,
                                         channel_id,
                                         xfer_status,
                                         "completion");
}

template<nixl_gpu_level_t Level>
__device__ nixl_status_t
nixlbenchSignalError(const nixlbenchDeviceXferParams &params,
                     unsigned channel_id,
                     nixlGpuXferStatusH &xfer_status) {
    return nixlbenchSignalCounter<Level>(
        params, params.errorCounterOffsetBytes, 1ull, channel_id, xfer_status, "error");
}

/**
 * Performs device-initiated NIXL PUT transfers and reports completion or errors
 * through remote counters.
 *
 * Every group runs @c numIterations iterations of the complete region list,
 * so the block as a whole performs @c numIterations * num_groups transfers of the list.
 * Groups keep independent transfer status and timing samples,
 * signal its own share of the completion counter, so no group synchronization is needed.
 */
template<nixl_gpu_level_t Level>
__global__ void
nixlbenchPutKernel(nixlbenchDeviceXferParams params) {
    __shared__ nixlGpuXferStatusH xfer_statuses[kMaxGroups];
    unsigned group_id, num_groups;
    if constexpr (Level == nixl_gpu_level_t::THREAD) {
        group_id = threadIdx.x;
        num_groups = blockDim.x;
    } else { // CUDA warpSize == kWarpSize == 32
        group_id = threadIdx.x / warpSize;
        num_groups = (blockDim.x + warpSize - 1) / warpSize;
    }
    nixlGpuXferStatusH &xfer_status = xfer_statuses[group_id];
    const unsigned channel_id = group_id % params.channelNum;
    const bool group_leader = Level == nixl_gpu_level_t::THREAD || threadIdx.x % warpSize == 0;
    const size_t region_base = group_id * params.numRegions;
    bool group_failed = false;

    for (uint64_t iter = 0; iter < params.numIterations && !group_failed; ++iter) {
        if constexpr (Level == nixl_gpu_level_t::WARP) {
            __syncwarp();
        }
        const uint64_t post_start_ns = nixlbenchGetTimeNs();

        nixl_status_t put_status = NIXL_SUCCESS;
        for (size_t region_idx = 0; region_idx < params.numRegions; ++region_idx) {
            put_status =
                nixlbenchPostPut<Level>(params, region_base + region_idx, channel_id, xfer_status);
            if (put_status != NIXL_IN_PROG) {
                break;
            }
        }
        if constexpr (Level == nixl_gpu_level_t::WARP) {
            __syncwarp();
        }
        const uint64_t post_end_ns = nixlbenchGetTimeNs();

        if (put_status == NIXL_IN_PROG) {
            put_status = nixlbenchPollXferStatus<Level>(put_status, xfer_status);
        }
        if constexpr (Level == nixl_gpu_level_t::WARP) {
            __syncwarp();
        }
        const uint64_t xfer_end_ns = nixlbenchGetTimeNs();

        if (group_leader) {
            const size_t sample_idx = iter * num_groups + group_id;
            params.postDurationNs[sample_idx] = post_end_ns - post_start_ns;
            params.xferDurationNs[sample_idx] = xfer_end_ns - post_end_ns;
        }

        if (put_status != NIXL_SUCCESS) {
            if (group_leader) {
                printf("[nixlbenchPutKernel] transfer did not complete: "
                       "threadIdx.x=%u blockIdx.x=%u blockDim.x=%u final_status=%d\n",
                       threadIdx.x,
                       blockIdx.x,
                       blockDim.x,
                       static_cast<int>(put_status));
            }
            group_failed = true;
        }
    }

    if (group_failed) {
        (void)nixlbenchSignalError<Level>(params, channel_id, xfer_status);
    } else if (nixlbenchSignalCompletion<Level>(
                   params, params.numIterations, channel_id, xfer_status) != NIXL_SUCCESS) {
        (void)nixlbenchSignalError<Level>(params, channel_id, xfer_status);
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

    if (params.postDurationNs == nullptr || params.xferDurationNs == nullptr) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: duration output buffers are required "
                     "(postDurationNs and xferDurationNs must hold numIterations * num_groups "
                     "entries)\n";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (params.channelNum == 0) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: channelNum must be greater than zero\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (block_threads == 0 || block_threads > 1024u) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: invalid block_threads=" << block_threads
                  << " (must be 1..1024)\n";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (block_threads > kWarpSize && block_threads % kWarpSize != 0) {
        std::cerr << "nixlbench: nixlbenchLaunchDevicePut: block_threads (" << block_threads
                  << ") must be a multiple of " << kWarpSize
                  << " (WARP-level nixlPut requires full warps)\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (block_threads <= kWarpSize) {
        nixlbenchPutKernel<nixl_gpu_level_t::THREAD><<<1, block_threads, 0, nullptr>>>(params);
    } else {
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
