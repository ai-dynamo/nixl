// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <cstdlib>

#ifdef HAVE_GPU

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>

static void
checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

static void
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
#ifdef HAVE_CUDA
    checkCudaError(cudaSetDevice(device), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipSetDevice(device), message);
#endif
}

inline void
gpuMalloc(void **ptr, size_t size, const char *message) {
#ifdef HAVE_CUDA
    checkCudaError(cudaMalloc(ptr, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMalloc(ptr, size), message);
#endif
}

inline void
gpuFree(void *ptr, const char *message) {
#ifdef HAVE_CUDA
    checkCudaError(cudaFree(ptr), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipFree(ptr), message);
#endif
}

inline void
gpuMemset(void *ptr, int value, size_t size, const char *message) {
#ifdef HAVE_CUDA
    checkCudaError(cudaMemset(ptr, value, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMemset(ptr, value, size), message);
    checkHipError(hipStreamSynchronize(0), "Failed to synchronize stream");
#endif
}

inline void
gpuGetDeviceCount(int *count, const char *message) {
#ifdef HAVE_CUDA
    checkCudaError(cudaGetDeviceCount(count), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipGetDeviceCount(count), message);
#endif
}

inline void
gpuMemcpyD2H(void *dst, const void *src, size_t size, const char *message) {
#ifdef HAVE_CUDA
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost), message);
#endif
}

#endif // HAVE_GPU
