// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <cstdlib>
#include "test_utils.h"

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
    nixl_exit_on_failure(false, message);
#endif
}

inline void
gpuMalloc(void **ptr, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMalloc(ptr, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMalloc(ptr, size), message);
#else
    nixl_exit_on_failure(false, message);
#endif
}

inline void
gpuFree(void *ptr, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaFree(ptr), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipFree(ptr), message);
#else
    nixl_exit_on_failure(false, message);
#endif
}

inline void
gpuMemset(void *ptr, int value, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMemset(ptr, value, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMemset(ptr, value, size), message);
    checkHipError(hipStreamSynchronize(0), "Failed to synchronize stream");
#else
    nixl_exit_on_failure(false, message);
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

inline void
gpuMemcpyD2H(void *dst, const void *src, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost), message);
#else
    nixl_exit_on_failure(false, message);
#endif
}

inline int
gpuQueryAddr(void *address, bool &is_dev, int &dev) {
#if defined(HAVE_CUDA)
    CUdevice cu_dev;
    CUcontext cu_ctx;
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
    constexpr int num_attrs = 4;
    CUpointer_attribute attr_type[num_attrs];
    void *attr_data[num_attrs];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &cu_dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &cu_ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    if (result == CUDA_SUCCESS) {
        is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);
        dev = cu_dev;

        std::cout << "CUDA addr: " << std::hex << address << " dev=" << std::dec << dev
                  << " ctx=" << std::hex << cu_ctx << std::dec << std::endl;
        return 0;
    }
    return 1;
#elif defined(HAVE_ROCM)
    hipPointerAttribute_t attrs;
    const hipError_t result = hipPointerGetAttributes(&attrs, address);

    if (result == hipSuccess) {
        is_dev = (attrs.type == hipMemoryTypeDevice);
        dev = attrs.device;
        std::cout << "HIP addr: " << std::hex << address << " dev=" << std::dec << dev << std::endl;
        return 0;
    }
    return 1;
#else
    nixl_exit_on_failure(false, "GPU support not available");
    return 1;
#endif
}
