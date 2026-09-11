/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "device/device_allocator.h"

#include <cuda_runtime.h>

#include "common/nixl_log.h"

namespace {

void
logCudaFailure(const char *operation, cudaError_t error) {
    NIXL_ERROR << operation << " failed: " << cudaGetErrorString(error);
}

class nixlCudaDeviceAllocator final : public nixlDeviceAllocator {
public:
    nixl_status_t
    doAllocDeviceMem(void **ptr, size_t size) noexcept override {
        if (ptr == nullptr || size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        *ptr = nullptr;
        const cudaError_t error = cudaMalloc(ptr, size);
        if (error != cudaSuccess) {
            logCudaFailure("cudaMalloc", error);
            *ptr = nullptr;
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    void
    doFreeDeviceMem(void *ptr) noexcept override {
        // cudaFree is a no-op on nullptr and resolves the allocation's owning
        // device from the pointer itself, so neither a null guard nor a device
        // switch belongs here. The caller must pass a pointer that
        // allocDeviceMem produced.
        const cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            logCudaFailure("cudaFree", error);
        }
    }

    nixl_status_t
    doAllocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept override {
        if (host_ptr == nullptr || dev_ptr == nullptr || size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        *host_ptr = nullptr;
        *dev_ptr = nullptr;
        // Mapped guarantees cudaHostGetDevicePointer; portable permits use from other GPUs.
        cudaError_t error =
            cudaHostAlloc(host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
        if (error != cudaSuccess) {
            logCudaFailure("cudaHostAlloc", error);
            *host_ptr = nullptr;
            return NIXL_ERR_BACKEND;
        }
        error = cudaHostGetDevicePointer(dev_ptr, *host_ptr, 0);
        if (error != cudaSuccess) {
            logCudaFailure("cudaHostGetDevicePointer", error);
            const cudaError_t free_error = cudaFreeHost(*host_ptr);
            if (free_error != cudaSuccess) {
                logCudaFailure("cudaFreeHost after cudaHostGetDevicePointer", free_error);
            }
            *host_ptr = nullptr;
            *dev_ptr = nullptr;
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    void
    doFreeMappedHostMem(void *host_ptr) noexcept override {
        if (host_ptr == nullptr) {
            return;
        }
        const cudaError_t error = cudaFreeHost(host_ptr);
        if (error != cudaSuccess) {
            logCudaFailure("cudaFreeHost", error);
        }
    }

    nixl_status_t
    copyHostToDevice(void *dst, const void *src, size_t size) noexcept override {
        if (dst == nullptr || src == nullptr || size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            logCudaFailure("cudaMemcpy host to device", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    copyDeviceToHost(void *dst, const void *src, size_t size) noexcept override {
        if (dst == nullptr || src == nullptr || size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            logCudaFailure("cudaMemcpy device to host", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
        if (ptr == nullptr || size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const cudaError_t error = cudaMemset(ptr, value, size);
        if (error != cudaSuccess) {
            logCudaFailure("cudaMemset", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    synchronize() noexcept override {
        const cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            logCudaFailure("cudaDeviceSynchronize", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    getActiveDevice(int &device_id) noexcept override {
        const cudaError_t error = cudaGetDevice(&device_id);
        if (error != cudaSuccess) {
            logCudaFailure("cudaGetDevice", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    setActiveDevice(int device_id) noexcept override {
        const cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess) {
            logCudaFailure("cudaSetDevice", error);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }
};

} // namespace

extern "C" __attribute__((visibility("default"))) nixlDeviceAllocator *
nixlCreateCudaDeviceAllocatorV1() noexcept {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return nullptr;
    }

    static nixlCudaDeviceAllocator allocator;
    return &allocator;
}
