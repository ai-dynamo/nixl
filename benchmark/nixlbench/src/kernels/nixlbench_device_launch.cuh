/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH
#define NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH

#include <nixl_types.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Parameters for @ref nixlbenchPutKernel (passed by value to the device).
 *
 * @a localMvh and @a remoteMvh must come from nixlAgent::prepMemView using the same flattening
 * order as xferBenchNixlWorker::prepareGPULocalView / prepareGPURemoteView (outer vector = group
 * lists, inner vector = IOVs for that group).
 *
 * @a numRegions is data region count: group @c g owns indices @c g*numRegions .. @c
 * (g+1)*numRegions-1. The host must append a counter buffer after all data descriptors, at index @a
 * counterIndex. The counter buffer stores:
 * - done counter at byte offset @a completionCounterOffsetBytes
 * - error counter at byte offset @a errorCounterOffsetBytes
 *
 * Every group in the launched block transfers @a numIterations times.
 * The block as a whole performs @c numIterations * num_groups list transfers.
 *
 * Each group signals independently using @c nixlAtomicAdd on @c { remoteMvh, counterIndex, offset }
 * over its own channel to add @a numIterations to the done counter.
 * Duration outputs contain @c numIterations * num_groups entries in iteration-major order.
 */
struct nixlbenchDeviceXferParams {
    nixlMemViewH localMvh; ///< Local memory view from prepMemView
    nixlMemViewH remoteMvh; ///< Remote memory view from prepMemView
    size_t numRegions; ///< Data region count (puts)
    size_t counterIndex; ///< Index of counter buffer (= numRegions * num_groups)
    size_t regionSize; ///< Bytes per region for this transfer pattern
    uint64_t numIterations; ///< Per-group number of complete region-list transfers
    uint64_t *postDurationNs; ///< Per-iteration, per-group PUT posting duration output
    uint64_t *xferDurationNs; ///< Per-iteration, per-group completion polling duration output
    size_t completionCounterOffsetBytes; ///< Done counter offset in the counter region
    size_t errorCounterOffsetBytes; ///< Error counter offset in the counter region
};

/**
 * @brief Launches @ref nixlbenchPutKernel with a 1-D block of @a block_threads threads.
 *
 * If @a block_threads is less than or equal to the GPU warp size (32),
 * @c nixl_gpu_level_t::THREAD is used (one group per thread);
 * otherwise @c nixl_gpu_level_t::WARP is used (one group per warp).
 * Typical @a block_threads matches nixlbench @c --num_threads.
 *
 * Requires NIXL UCX GPU Device API support. @a block_threads must be in [1, 1024];
 * values greater than 32 must be a multiple of 32.
 *
 * On failure, logs to stderr. Synchronizes the device so device printf output from the kernel is
 * flushed before returning.
 *
 * @param params        Transfer parameters (handles, counts, size).
 * @param block_threads CUDA block dimension in the x direction.
 * @return NIXL_SUCCESS if the kernel launches and synchronizes without CUDA runtime errors;
 *         NIXL_ERR_INVALID_PARAM for invalid parameters;
 *         NIXL_ERR_BACKEND for CUDA launch or synchronization failures.
 *         Device transfer failures are reported separately through the remote error counter.
 */
nixl_status_t
nixlbenchLaunchDevicePut(const nixlbenchDeviceXferParams &params, unsigned block_threads);

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_KERNELS_NIXLBENCH_DEVICE_LAUNCH_CUH
