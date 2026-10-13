// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NIXL_TEST_GTEST_GPU_UTILS_H
#define NIXL_TEST_GTEST_GPU_UTILS_H

#include <iostream>
#include <cstdlib>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>

inline void
checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#if defined(HAVE_ROCM)
#include <hip/hip_runtime.h>

inline void
checkHipError(hipError_t result, const char *message) {
    if (result != hipSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << hipGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

inline void
gpuSetDevice(int device, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaSetDevice(device), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipSetDevice(device), message);
#else
    (void)device;
    (void)message;
#endif
}

inline void
gpuMalloc(void **ptr, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMalloc(ptr, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMalloc(ptr, size), message);
#else
    (void)ptr;
    (void)size;
    (void)message;
#endif
}

inline void
gpuFree(void *ptr, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaFree(ptr), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipFree(ptr), message);
#else
    (void)ptr;
    (void)message;
#endif
}

inline void
gpuGetDeviceCount(int *count, const char *message) {
#if defined(HAVE_CUDA)
    cudaError_t result = cudaGetDeviceCount(count);
    if (result != cudaSuccess) {
        std::cout << message << " (" << cudaGetErrorString(result) << "), setting count to 0"
                  << std::endl;
        *count = 0;
    }
#elif defined(HAVE_ROCM)
    hipError_t result = hipGetDeviceCount(count);
    if (result != hipSuccess) {
        std::cout << message << " (" << hipGetErrorString(result) << "), setting count to 0"
                  << std::endl;
        *count = 0;
    }
#else
    std::cout << message << " (GPU support not compiled in), setting count to 0" << std::endl;
    *count = 0;
#endif
}

#endif // NIXL_TEST_GTEST_GPU_UTILS_H
