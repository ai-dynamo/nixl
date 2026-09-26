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
#include "device/device_ops.h"

#include <cuda_runtime.h>
#include <string>

#include "common/nixl_log.h"

namespace {

std::string
cudaFailureMsg(const char *operation, cudaError_t error) {
    return std::string(operation) + " failed: " + cudaGetErrorString(error);
}

nixl_status_t
cudaStatus(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        NIXL_ERROR << cudaFailureMsg(operation, error);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

class cudaDeviceOps final : public nixl::deviceOps {
public:
    nixl_status_t
    doAllocDeviceMem(void *&ptr, size_t size) noexcept override {
        void *allocation;
        const nixl_status_t status = cudaStatus(cudaMalloc(&allocation, size), "cudaMalloc");
        if (status != NIXL_SUCCESS) {
            return status;
        }
        ptr = allocation;
        return NIXL_SUCCESS;
    }

    void
    doFreeDeviceMem(void *ptr) noexcept override {
        // cudaFree recovers the owning device and accepts nullptr.
        const cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            NIXL_WARN << cudaFailureMsg("cudaFree", error);
        }
    }

    nixl_status_t
    doAllocMappedHostMem(void *&host_ptr, void *&dev_ptr, size_t size) noexcept override {
        void *host_allocation;
        void *device_alias;
        // Mapped guarantees cudaHostGetDevicePointer; portable permits use from other GPUs.
        nixl_status_t status = cudaStatus(
            cudaHostAlloc(&host_allocation, size, cudaHostAllocMapped | cudaHostAllocPortable),
            "cudaHostAlloc");
        if (status != NIXL_SUCCESS) {
            return status;
        }
        status = cudaStatus(cudaHostGetDevicePointer(&device_alias, host_allocation, 0),
                            "cudaHostGetDevicePointer");
        if (status != NIXL_SUCCESS) {
            doFreeMappedHostMem(host_allocation);
            return status;
        }
        host_ptr = host_allocation;
        dev_ptr = device_alias;
        return NIXL_SUCCESS;
    }

    void
    doFreeMappedHostMem(void *host_ptr) noexcept override {
        // cudaFreeHost is a no-op on nullptr, matching doFreeDeviceMem.
        const cudaError_t error = cudaFreeHost(host_ptr);
        if (error != cudaSuccess) {
            NIXL_WARN << cudaFailureMsg("cudaFreeHost", error);
        }
    }

    nixl_status_t
    copy(void *dst, const void *src, size_t size, copyDirection direction) noexcept override {
        if (size == 0 || dst == nullptr || src == nullptr) {
            NIXL_ERROR << "Invalid device copy: dst=" << dst << " src=" << src << " size=" << size;
            return NIXL_ERR_INVALID_PARAM;
        }
        cudaMemcpyKind kind;
        switch (direction) {
        case copyDirection::HostToDevice:
            kind = cudaMemcpyHostToDevice;
            break;
        case copyDirection::DeviceToHost:
            kind = cudaMemcpyDeviceToHost;
            break;
        default:
            NIXL_ERROR << "Invalid device copy direction: " << static_cast<int>(direction);
            return NIXL_ERR_INVALID_PARAM;
        }
        return cudaStatus(cudaMemcpy(dst, src, size, kind),
                          direction == copyDirection::HostToDevice ? "cudaMemcpy host to device" :
                                                                     "cudaMemcpy device to host");
    }

    nixl_status_t
    memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
        if (size == 0 || ptr == nullptr) {
            NIXL_ERROR << "Invalid device memset: ptr=" << ptr << " size=" << size;
            return NIXL_ERR_INVALID_PARAM;
        }
        return cudaStatus(cudaMemset(ptr, value, size), "cudaMemset");
    }

    nixl_status_t
    synchronize() noexcept override {
        return cudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    nixl_status_t
    getActiveDevice(int &device_id) noexcept override {
        return cudaStatus(cudaGetDevice(&device_id), "cudaGetDevice");
    }

    nixl_status_t
    setActiveDevice(int device_id) noexcept override {
        return cudaStatus(cudaSetDevice(device_id), "cudaSetDevice");
    }
};

} // namespace

extern "C" NIXL_DEVICE_OPS_EXPORT nixl::deviceOps *
nixlCreateCudaDeviceOps() noexcept {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaErrorNoDevice) {
        NIXL_INFO << "No CUDA-capable GPU is available";
        return nullptr;
    }
    if (cudaStatus(error, "cudaGetDeviceCount") != NIXL_SUCCESS) {
        return nullptr;
    }
    if (device_count == 0) {
        NIXL_INFO << "CUDA reported zero available GPUs";
        return nullptr;
    }

    static cudaDeviceOps ops;
    return &ops;
}
