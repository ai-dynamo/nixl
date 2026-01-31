/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GDAKI_KERNELS_CUH
#define GDAKI_KERNELS_CUH

#include <iostream>
#include <string_view>
#include <memory>
#include <nixl_types.h>

#define MAX_THREADS 1024

#ifndef CUDA_CALL
#define CUDA_CALL(_handler, _log_level, _func, ...)                                                \
    do {                                                                                           \
        cudaError_t _cerr = _func(__VA_ARGS__);                                                    \
        if (_cerr != cudaSuccess) {                                                                \
            std::cout << #_func << " failed: " << (int)_cerr << " : " << cudaGetErrorString(_cerr) \
                      << std::endl;                                                                \
            _handler;                                                                              \
        }                                                                                          \
    } while (0)
#endif

nixl_status_t
checkDeviceKernelParams(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        int threads_per_block,
                        int blocks_per_grid);

// RAII helpers for CUDA device allocations
struct device_free {
    void
    operator()(void *p) const noexcept {
        if (p) cudaFree(p);
    }
};

template<class T> using device_ptr = std::unique_ptr<T, device_free>;

nixl_status_t
checkDeviceKernelParams(nixlGpuXferReqH *req_handle,
                        int num_iterations,
                        int threads_per_block,
                        int blocks_per_grid);

// Common parameters for device kernel launches
struct deviceKernelParams {
    int num_iterations;
    std::string_view level;
    size_t count;
    const size_t *lens;
    void *const *local_addrs;
    const uint64_t *remote_addrs;
    size_t threads_per_block = 256;
    size_t blocks_per_grid = 1;
    cudaStream_t stream = 0;
    uint64_t signal_inc = 0;
};

// Launch Device kernel for device-side transfer execution (full transfers)
nixl_status_t
launchDeviceKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params);

// Launch Device kernel for partial transfers (supports thread/warp/block coordination)
nixl_status_t
launchDevicePartialKernel(nixlGpuXferReqH *req_handle, const deviceKernelParams &params);

uint64_t
readNixlGpuSignal(const void *signal_addr, std::string_view gpulevel);

#endif // GDAKI_KERNELS_CUH
