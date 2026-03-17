/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>

#include "config.hpp"
#include "vmm.hpp"

vmm_region
vmm_init(size_t size, CUdevice device) {
    if (size == 0) {
        throw std::invalid_argument("vmm_init: size must be non-zero");
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
        vmm_region region = {};
        region.size = size;
        region.is_cuda_malloc = true;
        if (cudaMalloc(reinterpret_cast<void **>(&region.ptr), size) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc fallback failed (fabric not supported)");
        }
        return region;
    }

    vmm_region region = {};
    CUmemAccessDesc access_desc = {};
    const char *err_msg;

    region.size = align_up<size_t>(size, ctx.granularity);

    if (cuMemCreate(&region.handle, region.size, &ctx.prop, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to create CUDA VMM allocation");
    }

    if (cuMemAddressReserve(&region.ptr, region.size, 0, 0, 0) != CUDA_SUCCESS) {
        err_msg = "Failed to reserve CUDA virtual address";
        goto err_release;
    }

    if (cuMemMap(region.ptr, region.size, 0, region.handle, 0) != CUDA_SUCCESS) {
        err_msg = "Failed to map CUDA VMM memory";
        goto err_free;
    }

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(region.ptr, region.size, &access_desc, 1) != CUDA_SUCCESS) {
        err_msg = "Failed to set CUDA memory access";
        goto err_unmap;
    }

    return region;

err_unmap:
    cuMemUnmap(region.ptr, region.size);
err_free:
    cuMemAddressFree(region.ptr, region.size);
    region.ptr = 0;
err_release:
    cuMemRelease(region.handle);
    region.handle = 0;
    throw std::runtime_error(err_msg);
}

void
vmm_free(vmm_region &region) {
    if (!region.ptr) {
        return;
    }
    if (region.is_cuda_malloc) {
        cudaFree(reinterpret_cast<void *>(region.ptr));
        region.ptr = 0;
        return;
    }
    cuMemUnmap(region.ptr, region.size);
    cuMemAddressFree(region.ptr, region.size);
    region.ptr = 0;
    if (region.handle) {
        cuMemRelease(region.handle);
        region.handle = 0;
    }
}
