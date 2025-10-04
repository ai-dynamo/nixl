/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gpu/ucx/nixl_device.cuh>
#include "gdaki_kernels.cuh"

// Helper function to get request index based on coordination level (from gtest)
template<nixl_gpu_level_t level>
__device__ constexpr size_t
getRequestIndex() {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        return threadIdx.x;
    case nixl_gpu_level_t::WARP:
        return threadIdx.x / warpSize;
    case nixl_gpu_level_t::BLOCK:
        return 0;
    default:
        return 0;
    }
}

nixl_gpu_level_t
stringToGpuLevel(const char *gdaki_level) {
    if (!strcmp(gdaki_level, "WARP")) return nixl_gpu_level_t::WARP;
    if (!strcmp(gdaki_level, "BLOCK")) return nixl_gpu_level_t::BLOCK;
    return nixl_gpu_level_t::THREAD;
}

// GDAKI kernel for full transfers (block coordination only)
__global__ void
gdakiFullTransferKernel(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        const size_t *lens,
                        void *const *local_addrs,
                        const uint64_t *remote_addrs,
                        const uint64_t signal_inc,
                        const uint64_t remote_addr) {
    __shared__ nixlGpuXferStatusH xfer_status;
    nixlGpuSignal signal = {signal_inc, remote_addr};

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Post the GPU transfer request with signal increment of 1
        nixl_status_t status = nixlGpuPostWriteXferReq<nixl_gpu_level_t::BLOCK>(
            req_handle, lens, local_addrs, remote_addrs, signal, true, &xfer_status);
        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

// GDAKI kernel for partial transfers (supports thread/warp/block coordination)
template<nixl_gpu_level_t level>
__global__ void
gdakiPartialTransferKernel(nixlGpuXferReqH *req_handle,
                           int num_iterations,
                           const size_t count,
                           const size_t *lens,
                           void *const *local_addrs,
                           const uint64_t *remote_addrs,
                           const uint64_t signal_inc,
                           const uint64_t remote_addr) {
    nixlGpuXferStatusH xfer_status;
    nixlGpuSignal signal = {signal_inc, remote_addr};

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Use partial transfer API which supports all coordination levels
        nixl_status_t status = nixlGpuPostPartialWriteXferReq<level>(req_handle,
                                                                     count,
                                                                     nullptr,
                                                                     lens,
                                                                     local_addrs,
                                                                     remote_addrs,
                                                                     signal,
                                                                     1,
                                                                     true,
                                                                     &xfer_status);
        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<level>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            return; // Early exit on error
        }
    }
}

template<nixl_gpu_level_t level>
__global__ void
gdakiReadSignalKernel(const void *signal_addr, uint64_t *count) {
    *count = nixlGpuReadSignal<level>(signal_addr);
}

// Host-side launcher
nixl_status_t
checkDeviceKernelParams(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        int threads_per_block,
                        int blocks_per_grid) {
    // Validate parameters
    if (num_iterations <= 0 || req_handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > MAX_THREADS) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
launchDeviceKernel(nixlGpuXferReqH *req_handle,
                   int num_iterations,
                   const char *level,
                   const size_t *lens,
                   void *const *local_addrs,
                   const uint64_t *remote_addrs,
                   int threads_per_block,
                   int blocks_per_grid,
                   cudaStream_t stream,
                   const uint64_t signal_inc,
                   const uint64_t remote_addr) {

    nixl_gpu_level_t gpulevel = stringToGpuLevel(level);
    nixl_status_t ret =
        checkDeviceKernelParams(req_handle, num_iterations, threads_per_block, blocks_per_grid);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to validate kernel launch parameters" << std::endl;
        return ret;
    }

    // Use full transfer kernel for block coordination only
    if (gpulevel != nixl_gpu_level_t::BLOCK) {
        std::cout << "Falling back to block coordination for full transfers" << std::endl;
    }

    gdakiFullTransferKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        req_handle, num_iterations, lens, local_addrs, remote_addrs, signal_inc, remote_addr);

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "Failed to launch device full transfer kernel: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
launchDevicePartialKernel(nixlGpuXferReqH *req_handle,
                          int num_iterations,
                          const char *level,
                          const size_t count,
                          const size_t *lens,
                          void *const *local_addrs,
                          const uint64_t *remote_addrs,
                          int threads_per_block,
                          int blocks_per_grid,
                          cudaStream_t stream,
                          const uint64_t signal_inc,
                          const uint64_t remote_addr) {

    nixl_gpu_level_t gpulevel = stringToGpuLevel(level);
    nixl_status_t ret =
        checkDeviceKernelParams(req_handle, num_iterations, threads_per_block, blocks_per_grid);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to validate kernel launch parameters" << std::endl;
        return ret;
    }

    // Launch partial transfer kernel based on coordination level
    if (gpulevel == nixl_gpu_level_t::THREAD) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::THREAD>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                num_iterations,
                                                                count,
                                                                lens,
                                                                local_addrs,
                                                                remote_addrs,
                                                                signal_inc,
                                                                remote_addr);
    } else if (gpulevel == nixl_gpu_level_t::WARP) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::WARP>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(req_handle,
                                                                num_iterations,
                                                                count,
                                                                lens,
                                                                local_addrs,
                                                                remote_addrs,
                                                                signal_inc,
                                                                remote_addr);
    } else {
        std::cerr << "Invalid GPU level selected for partial transfers: " << level << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check for launch errors
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::cerr << "Failed to launch device partial transfer kernel: "
                  << cudaGetErrorString(launch_error) << std::endl;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

uint64_t
readNixlGpuSignal(const void *signal_addr, const char *gpulevel) {
    const nixl_gpu_level_t level = stringToGpuLevel(gpulevel);
    uint64_t count = 0;
    uint64_t *d_count = nullptr;

    // Allocate device memory for the result
    cudaError_t err = cudaMalloc(&d_count, sizeof(uint64_t));
    if (err != cudaSuccess || !d_count) {
        return 0;
    }

    // Launch kernel with single thread/block configuration
    if (level == nixl_gpu_level_t::THREAD) {
        gdakiReadSignalKernel<nixl_gpu_level_t::THREAD><<<1, 1>>>(signal_addr, d_count);
    } else if (level == nixl_gpu_level_t::WARP) {
        gdakiReadSignalKernel<nixl_gpu_level_t::WARP><<<1, 1>>>(signal_addr, d_count);
    } else if (level == nixl_gpu_level_t::BLOCK) {
        gdakiReadSignalKernel<nixl_gpu_level_t::BLOCK><<<1, 1>>>(signal_addr, d_count);
    }

    // Wait for kernel completion and copy result back
    cudaDeviceSynchronize();
    cudaMemcpy(&count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_count);

    return count;
}
