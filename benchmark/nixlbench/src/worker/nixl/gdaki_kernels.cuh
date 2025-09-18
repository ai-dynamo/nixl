/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GDAKI_KERNELS_CUH
#define GDAKI_KERNELS_CUH

#include <cuda_runtime.h>
#include <nixl_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch GDAKI kernel for device-side transfer execution (full transfers)
nixl_status_t launchGdakiKernel(
    nixlGpuXferReqH* req_handle,
    int num_iterations,
    const std::string& coordination_level = "block",
    int threads_per_block = 256,
    int blocks_per_grid = 1,
    cudaStream_t stream = 0);

// Launch GDAKI kernel for partial transfers (supports thread/warp/block coordination)
nixl_status_t launchGdakiPartialKernel(
    nixlGpuXferReqH* req_handle,
    int num_iterations,
    const std::string& coordination_level = "block",
    int threads_per_block = 256,
    int blocks_per_grid = 1,
    cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif // GDAKI_KERNELS_CUH
