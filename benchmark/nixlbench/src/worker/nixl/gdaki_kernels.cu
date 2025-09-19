/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nixl_types.h>
#include <utils/ucx/nixl_gdaki_device.cuh>
#include "gdaki_kernels.cuh"

// Helper function to get request index based on coordination level (from gtest)
template<nixl_gpu_xfer_coordination_level_t level>
__device__ constexpr size_t
getRequestIndex() {
    switch (level) {
    case NIXL_GPU_XFER_COORDINATION_THREAD:
        return threadIdx.x;
    case NIXL_GPU_XFER_COORDINATION_WARP:
        return threadIdx.x / warpSize;
    case NIXL_GPU_XFER_COORDINATION_BLOCK:
        return 0;
    default:
        return 0;
    }
}

// GDAKI kernel for full transfers (block coordination only)
__global__ void
gdakiFullTransferKernel(nixlGpuXferReqH *req_handle, int num_iterations) {
    __shared__ nixlGpuXferStatusH xfer_status;

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Post the GPU transfer request with signal increment of 1
        nixl_status_t status =
            nixlPostGpuXferReq<NIXL_GPU_XFER_COORDINATION_BLOCK>(req_handle, 1ULL, &xfer_status);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGetGpuXferReqStatus<NIXL_GPU_XFER_COORDINATION_BLOCK>(&xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

// GDAKI kernel for partial transfers (supports thread/warp/block coordination)
template<nixl_gpu_xfer_coordination_level_t level>
__global__ void
gdakiPartialTransferKernel(nixlGpuXferReqH *req_handle, int num_iterations) {
    constexpr size_t MAX_THREADS = 1024;
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr = &xfer_status[getRequestIndex<level>()];

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Use partial transfer API which supports all coordination levels
        nixl_status_t status = nixlPostPartialGpuXferReq<level>(
            req_handle, 1ULL, 0, nullptr, nullptr, nullptr, nullptr, xfer_status_ptr);
        if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGetGpuXferReqStatus<level>(xfer_status_ptr);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

// Host-side launcher
extern "C" {

nixl_status_t
launchGdakiKernel(nixlGpuXferReqH *req_handle,
                  int num_iterations,
                  const std::string &coordination_level,
                  int threads_per_block,
                  int blocks_per_grid,
                  cudaStream_t stream) {

    // Validate parameters
    if (num_iterations <= 0 || req_handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (coordination_level != "thread" && coordination_level != "warp" &&
        coordination_level != "block") {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > 1024) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Use full transfer kernel for block coordination only
    if (coordination_level == "block") {
        gdakiFullTransferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                                   num_iterations);
    } else {
        // For thread/warp coordination, fall back to block coordination for full transfers
        gdakiFullTransferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                                   num_iterations);
    }

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
launchGdakiPartialKernel(nixlGpuXferReqH *req_handle,
                         int num_iterations,
                         const std::string &coordination_level,
                         int threads_per_block,
                         int blocks_per_grid,
                         cudaStream_t stream) {

    // Validate parameters
    if (num_iterations <= 0 || req_handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (coordination_level != "thread" && coordination_level != "warp" &&
        coordination_level != "block") {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > 1024) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Launch partial transfer kernel based on coordination level
    if (coordination_level == "thread") {
        gdakiPartialTransferKernel<NIXL_GPU_XFER_COORDINATION_THREAD>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations);
    } else if (coordination_level == "warp") {
        gdakiPartialTransferKernel<NIXL_GPU_XFER_COORDINATION_WARP>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations);
    } else if (coordination_level == "block") {
        gdakiPartialTransferKernel<NIXL_GPU_XFER_COORDINATION_BLOCK>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle, num_iterations);
    }

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

} // extern "C"
