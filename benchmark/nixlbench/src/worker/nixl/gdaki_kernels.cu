/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gpu/ucx/nixl_device.cuh>
#include "gdaki_levels.h"
#include "gdaki_kernels.cuh"

// Helper function to get request index based on coordination level (from gtest)
template<nixl_gpu_level_t level>
inline __device__ constexpr size_t
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

static inline nixl_gpu_level_t
stringToGpuLevel(std::string_view gdaki_level) {
    if (gdaki_level == gdaki_levels::kWarp) {
        return nixl_gpu_level_t::WARP;
    }
    if (gdaki_level == gdaki_levels::kBlock) {
        return nixl_gpu_level_t::BLOCK;
    }
    return nixl_gpu_level_t::THREAD;
}

// GDAKI kernel for full transfers (block coordination only)
__global__ void
gdakiFullTransferKernel(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        const uint64_t signal_inc) {
    __shared__ nixlGpuXferStatusH xfer_status;

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Post the GPU transfer request with signal increment of 1
        nixl_status_t status = nixlGpuPostSignalXferReq<nixl_gpu_level_t::BLOCK>(
            req_handle, 0, signal_inc, 0, 0, true, &xfer_status);
        if (status != NIXL_SUCCESS) {
            printf("Failed to post transfer iteration %d request: %d\n", i, status);
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<nixl_gpu_level_t::BLOCK>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            printf("Failed to get transfer status: %d\n", status);
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
                           const uint64_t signal_inc) {
    __shared__ nixlGpuXferStatusH xfer_status;

    // Execute transfers for the specified number of iterations
    for (int i = 0; i < num_iterations; i++) {
        // Use partial transfer API which supports all coordination levels
        nixl_status_t status = nixlGpuPostPartialWriteXferReq<level>(req_handle,
                                                                     count,
                                                                     nullptr,
                                                                     lens,
                                                                     nullptr,
                                                                     nullptr,
                                                                     0,
                                                                     signal_inc,
                                                                     0,
                                                                     0,
                                                                     true,
                                                                     &xfer_status);
        if (status != NIXL_SUCCESS) {
            printf("Failed to post partial transfer iteration %d request: %d\n", i, status);
            return; // Early exit on error
        }

        // Wait for transfer completion
        do {
            status = nixlGpuGetXferStatus<level>(xfer_status);
        } while (status == NIXL_IN_PROG);

        if (status != NIXL_SUCCESS) {
            printf("Failed to get transfer status: %d\n", status);
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
    if (num_iterations <= 0) {
        std::cerr << "Invalid number of iterations, must be greater than 0" << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (req_handle == nullptr) {
        std::cerr << "Invalid request handle, must be non-null" << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (threads_per_block < 1 || threads_per_block > MAX_THREADS) {
        std::cerr << "Invalid threads per block, must be between 1 and " << MAX_THREADS
                  << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (blocks_per_grid < 1) {
        std::cerr << "Invalid blocks per grid, must be greater than 0" << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
launchDeviceKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params) {

    nixl_gpu_level_t gpulevel = stringToGpuLevel(params.level);
    nixl_status_t ret = checkDeviceKernelParams(
        req_handle, params.num_iterations, params.threads_per_block, params.blocks_per_grid);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to validate kernel launch parameters" << std::endl;
        return ret;
    }

    // Use full transfer kernel for block coordination only
    if (gpulevel != nixl_gpu_level_t::BLOCK) {
        std::cout << "Falling back to block coordination for full transfers" << std::endl;
    }

    // Allocate device memory for address arrays (RAII-managed)
    void **d_local_addrs_raw = nullptr;
    size_t *d_lens_raw = nullptr;

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMalloc,
                     &d_local_addrs_raw,
                     params.count * sizeof(void *));

    device_ptr<void *> d_local_addrs(d_local_addrs_raw);

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMalloc,
                     &d_lens_raw,
                     params.count * sizeof(size_t));

    device_ptr<size_t> d_lens(d_lens_raw);

    // Copy host arrays to device
    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMemcpy,
                     d_local_addrs.get(),
                     params.local_addrs,
                     params.count * sizeof(void *),
                     cudaMemcpyHostToDevice);

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMemcpy,
                     d_lens.get(),
                     params.lens,
                     params.count * sizeof(size_t),
                     cudaMemcpyHostToDevice);

    gdakiFullTransferKernel<<<1, params.threads_per_block>>>(
        req_handle, params.num_iterations, params.signal_inc);
    // Check for launch errors
    CUDA_CALL(return NIXL_ERR_BACKEND, UCS_LOG_LEVEL_ERROR, cudaGetLastError);

    // Wait for kernel completion before freeing device memory
    CUDA_CALL(/*no-op*/, UCS_LOG_LEVEL_ERROR, cudaStreamSynchronize, params.stream);

    return NIXL_SUCCESS;
}

nixl_status_t
launchDevicePartialKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params) {

    nixl_gpu_level_t gpulevel = stringToGpuLevel(params.level);
    nixl_status_t ret = checkDeviceKernelParams(
        req_handle, params.num_iterations, params.threads_per_block, params.blocks_per_grid);

    if (ret != NIXL_SUCCESS) {
        std::cerr << "Failed to validate kernel launch parameters" << std::endl;
        return ret;
    }

    // Allocate device memory for address arrays (RAII-managed)
    void **d_local_addrs_raw = nullptr;
    size_t *d_lens_raw = nullptr;

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMalloc,
                     &d_local_addrs_raw,
                     params.count * sizeof(void *));

    device_ptr<void *> d_local_addrs(d_local_addrs_raw);

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMalloc,
                     &d_lens_raw,
                     params.count * sizeof(size_t));

    device_ptr<size_t> d_lens(d_lens_raw);

    // Copy host arrays to device
    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMemcpy,
                     d_local_addrs.get(),
                     params.local_addrs,
                     params.count * sizeof(void *),
                     cudaMemcpyHostToDevice);

    CUDA_CALL(return NIXL_ERR_BACKEND,
                     UCS_LOG_LEVEL_ERROR,
                     cudaMemcpy,
                     d_lens.get(),
                     params.lens,
                     params.count * sizeof(size_t),
                     cudaMemcpyHostToDevice);

    // Launch partial transfer kernel based on coordination level
    if (gpulevel == nixl_gpu_level_t::THREAD) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::THREAD><<<1, params.threads_per_block>>>(
            req_handle, params.num_iterations, params.count, d_lens.get(), params.signal_inc);
    } else if (gpulevel == nixl_gpu_level_t::WARP) {
        gdakiPartialTransferKernel<nixl_gpu_level_t::WARP><<<1, params.threads_per_block>>>(
            req_handle, params.num_iterations, params.count, d_lens.get(), params.signal_inc);
    } else {
        std::cerr << "Invalid GPU level selected for partial transfers: " << params.level
                  << std::endl;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Check for launch errors
    CUDA_CALL(return NIXL_ERR_BACKEND, UCS_LOG_LEVEL_ERROR, cudaGetLastError);

    // Wait for kernel completion before freeing device memory
    CUDA_CALL(/*no-op*/, UCS_LOG_LEVEL_ERROR, cudaStreamSynchronize, params.stream);

    return NIXL_SUCCESS;
}

uint64_t
readNixlGpuSignal(const void *signal_addr, std::string_view gpulevel) {
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
