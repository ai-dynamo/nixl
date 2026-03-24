/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>

#include "config.hpp"
#include "cuda_utils.hpp"
#include "vmm.hpp"

namespace {
constexpr const char *k_vmm_ctx = "vmm_region";
}

void
vmm_region::release() noexcept {
    if (is_cuda_malloc_) {
        if (ptr_) {
            nixl_ep::warn_cuda_api(cudaFree(reinterpret_cast<void *>(ptr_)), k_vmm_ctx, "cudaFree");
        }
        ptr_ = 0;
        size_ = 0;
        is_cuda_malloc_ = false;
        return;
    }

    if (vmm_mapped_) {
        nixl_ep::warn_cu_api(cuMemUnmap(ptr_, size_), k_vmm_ctx, "cuMemUnmap");
        vmm_mapped_ = false;
    }
    if (vmm_addr_reserved_ && ptr_) {
        nixl_ep::warn_cu_api(cuMemAddressFree(ptr_, size_), k_vmm_ctx, "cuMemAddressFree");
        ptr_ = 0;
        vmm_addr_reserved_ = false;
    }
    if (handle_) {
        nixl_ep::warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
        handle_ = 0;
    }
    size_ = 0;
}

vmm_region::~vmm_region() {
    release();
}

vmm_region::vmm_region(size_t size, CUdevice device) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: size must be non-zero");
    }

    struct cuda_alloc_ctx {
        bool fabric_supported;
        CUmemAllocationProp prop;
        size_t granularity;

        cuda_alloc_ctx(CUdevice dev) : fabric_supported(false), prop({}), granularity(0) {
            int version;
            if (cuDriverGetVersion(&version) != CUDA_SUCCESS) {
                throw std::runtime_error("Failed to get CUDA driver version");
            }

            if (version < 11000) {
                return; /* too old — fall back to cudaMalloc */
            }

            int fab = 0;
            if ((cuDeviceGetAttribute(&fab,
                                      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                      dev) != CUDA_SUCCESS) ||
                (!fab)) {
                return; /* no fabric — fall back to cudaMalloc */
            }

            int rdma_vmm_supported = 0;
            if (cuDeviceGetAttribute(&rdma_vmm_supported,
                                     CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                     dev) != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "Failed to query GPUDirect RDMA with VMM support attribute");
            }

            if (!rdma_vmm_supported) {
                throw std::runtime_error(
                    "GPUDirect RDMA with CUDA VMM is not supported on this device");
            }

            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = dev;
            prop.allocFlags.gpuDirectRDMACapable = 1;
            prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

            if (cuMemGetAllocationGranularity(
                    &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
                throw std::runtime_error("Failed to get CUDA allocation granularity");
            }

            fabric_supported = true;
        }
    };

    static cuda_alloc_ctx ctx(device);

    if (!ctx.fabric_supported) {
        size_ = size;
        is_cuda_malloc_ = true;
        if (cudaMalloc(reinterpret_cast<void **>(&ptr_), size) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc fallback failed (fabric not supported)");
        }
        return;
    }

    CUmemAccessDesc access_desc = {};

    size_ = nixl_ep::align_up<size_t>(size, ctx.granularity);

    if (cuMemCreate(&handle_, size_, &ctx.prop, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create CUDA VMM allocation");
    }

    if (cuMemAddressReserve(&ptr_, size_, 0, 0, 0) != CUDA_SUCCESS) {
        release();
        throw std::runtime_error("Failed to reserve CUDA virtual address");
    }
    vmm_addr_reserved_ = true;

    if (cuMemMap(ptr_, size_, 0, handle_, 0) != CUDA_SUCCESS) {
        release();
        throw std::runtime_error("Failed to map CUDA VMM memory");
    }
    vmm_mapped_ = true;

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(ptr_, size_, &access_desc, 1) != CUDA_SUCCESS) {
        release();
        throw std::runtime_error("Failed to set CUDA memory access");
    }
}
