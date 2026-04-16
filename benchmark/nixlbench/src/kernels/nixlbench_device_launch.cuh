/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH
#define NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH

#include <nixl_types.h>
#include <cuda_runtime.h>
#include <stddef.h>

/**
 * @brief Parameters for @ref nixlbenchPutKernel (passed by value to the device).
 *
 * @a localMvh and @a remoteMvh must come from nixlAgent::prepMemView using the same flattening
 * order as xferBenchNixlWorker::prepareGPULocalView / prepareGPURemoteView (outer vector = thread
 * lists, inner vector = IOVs for that thread).
 *
 * @a numRegions is the **data** region count (put loop uses indices @c 0 .. @a numRegions-1). When
 * @a signalRemoteCompletion is true, the host must have appended a counter buffer as the last
 * remote descriptor so the view has @a numRegions + 1 regions. The counter buffer stores:
 * - done counter at byte offset @a completionCounterOffsetBytes
 * - error counter at byte offset @a errorCounterOffsetBytes
 *
 * Kernel uses @c nixlAtomicAdd on @c { remoteMvh, numRegions, offset }.
 * When @a signalRemoteCompletion is false, @a remoteMvh has exactly @a numRegions regions and
 * no counter atomic is issued.
 */
struct nixlbenchDeviceXferParams {
    nixlMemViewH localMvh;   ///< Local memory view from prepMemView
    nixlMemViewH remoteMvh;  ///< Remote memory view from prepMemView
    size_t numRegions;       ///< Data region count (puts); completion index when signaling
    size_t regionSize;       ///< Bytes per region for this transfer pattern
    bool signalRemoteCompletion; ///< If true, counter region exists at index @a numRegions
    size_t completionCounterOffsetBytes; ///< Done counter offset in the counter region
    size_t errorCounterOffsetBytes;      ///< Error counter offset in the counter region
};

/**
 * @brief Launches @ref nixlbenchPutKernel with a 1-D block of @a block_threads threads.
 *
 * If @a block_threads is less than or equal to the GPU warp size (32),
 * @c nixl_gpu_level_t::THREAD is used;
 * otherwise @c nixl_gpu_level_t::WARP is used (each warp strides over regions and all lanes in
 * the warp participate in each device API call). Typical
 * @a block_threads matches nixlbench @c --num_threads.
 *
 * Requires UCX with GPU device API (gpu/ucx/nixl_device.cuh). @a block_threads must be in
 * [1, 1024] (CUDA block limit).
 *
 * On failure, logs to stderr. Synchronizes @a stream (or the device if null) so device printf
 * output from the kernel is flushed before returning.
 *
 * @param params        Transfer parameters (handles, counts, size).
 * @param block_threads CUDA block dimension in the x direction.
 * @param stream        CUDA stream, or nullptr for the default stream.
 * @return NIXL_SUCCESS on success; NIXL_ERR_INVALID_PARAM for invalid @a block_threads;
 *         NIXL_ERR_BACKEND for CUDA launch or synchronize failures.
 */
nixl_status_t
nixlbenchLaunchDevicePut(
    const nixlbenchDeviceXferParams &params,
    unsigned block_threads,
    cudaStream_t stream = nullptr);

#endif  // NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH
