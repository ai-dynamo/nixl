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

#include <iostream>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "vmm.hpp"

namespace {

constexpr const char *k_vmm_ctx = "vmm_region";

const char *
cu_error_string(CUresult status) noexcept {
    const char *msg = nullptr;
    if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
        msg = "unknown CUDA driver error";
    }
    return msg;
}

/** Log a non-fatal warning if a CUDA driver API call failed (e.g. during teardown). */
void
warn_cu_api(CUresult status, const char *context, const char *operation) noexcept {
    if (status != CUDA_SUCCESS) {
        std::cerr << "WARNING: " << context << " failed to " << operation << ": "
                  << cu_error_string(status) << '\n';
    }
}

struct cuda_alloc_ctx {
    bool fabric_supported;
    CUmemAllocationProp prop;
    size_t granularity;
    CUdevice device;
    CUmemAccessDesc access_desc = {};

    cuda_alloc_ctx() : fabric_supported(false), prop({}), granularity(0) {
        int version;

        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
            throw std::runtime_error("CUDA device should be set before creating a vmm_region");
        }

        if (cuDriverGetVersion(&version) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA driver version");
        }

        if (version < 11000) {
            return; /* too old — fall back to cudaMalloc */
        }

        int fab = 0;
        if ((cuDeviceGetAttribute(&fab,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                  device) != CUDA_SUCCESS) ||
            (!fab)) {
            return; /* no fabric — fall back to cudaMalloc */
        }

        int rdma_vmm_supported = 0;
        if (cuDeviceGetAttribute(&rdma_vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                 device) != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to query GPUDirect RDMA with VMM support attribute");
        }

        if (!rdma_vmm_supported) {
            std::cerr << "DIAG: " << k_vmm_ctx
                      << " - GPUDirect RDMA with CUDA VMM not supported; falling back to "
                         "cuMemAlloc\n";
            return;
        }

        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        if (cuMemGetAllocationGranularity(
                &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA allocation granularity");
        }

        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        fabric_supported = true;
    }
};

/** The allocation properties, probed once per process. */
cuda_alloc_ctx &
alloc_ctx() {
    static cuda_alloc_ctx ctx;
    return ctx;
}

/** A region that is not exportable gets no export handle type. */
CUmemAllocationProp
alloc_prop(bool exportable) {
    CUmemAllocationProp prop = alloc_ctx().prop;
    if (!exportable) {
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    }
    return prop;
}

} // namespace

namespace nixl_ep {

void
vmm_region::release() noexcept {
    if (is_cuda_malloc_) {
        if (ptr_) {
            warn_cu_api(cuMemFree(ptr_), k_vmm_ctx, "cuMemFree");
        }
        ptr_ = 0;
        return;
    }

    if (vmm_mapped_) {
        warn_cu_api(cuMemUnmap(ptr_, size_), k_vmm_ctx, "cuMemUnmap");
        vmm_mapped_ = false;
    }
    if (ptr_) {
        warn_cu_api(cuMemAddressFree(ptr_, size_), k_vmm_ctx, "cuMemAddressFree");
        ptr_ = 0;
    }
    if (handle_) {
        warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
        handle_ = 0;
    }
}

vmm_region::~vmm_region() {
    release();
}

vmm_region::vmm_region(size_t size, bool exportable) : exportable_(exportable) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: size must be non-zero");
    }

    cuda_alloc_ctx &ctx = alloc_ctx();

    if (ctx.fabric_supported) {
        size_ = nixl_ep::align_up<size_t>(size, ctx.granularity);

        const CUmemAllocationProp prop = alloc_prop(exportable_);
        const CUresult mem_create_status = cuMemCreate(&handle_, size_, &prop, 0);
        if (mem_create_status != CUDA_SUCCESS) {
            handle_ = 0;
            ctx.fabric_supported = false;
        } else {
            if (cuMemAddressReserve(&ptr_, size_, 0, 0, 0) != CUDA_SUCCESS) {
                release();
                throw std::runtime_error("Failed to reserve CUDA virtual address");
            }

            if (cuMemMap(ptr_, size_, 0, handle_, 0) != CUDA_SUCCESS) {
                release();
                throw std::runtime_error("Failed to map CUDA VMM memory");
            }
            vmm_mapped_ = true;

            if (cuMemSetAccess(ptr_, size_, &ctx.access_desc, 1) != CUDA_SUCCESS) {
                release();
                throw std::runtime_error("Failed to set CUDA memory access");
            }

            return;
        }
    }

    size_ = size;
    is_cuda_malloc_ = true;
    if (cuMemAlloc(&ptr_, size) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc fallback failed");
    }
}

bool
vmm_region::make_exportable() {
    if (is_cuda_malloc_ || exportable_) {
        return false;
    }
    // Back the same virtual addresses with new physical memory, so that
    // pointers captured in CUDA graphs stay valid. Contents are not kept.
    if (cuMemUnmap(ptr_, size_) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to unmap CUDA VMM memory to make it exportable");
    }
    vmm_mapped_ = false;
    warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
    handle_ = 0;
    const CUmemAllocationProp prop = alloc_prop(true);
    const CUresult status = cuMemCreate(&handle_, size_, &prop, 0);
    if (status != CUDA_SUCCESS) {
        handle_ = 0;
        // In a restored process, creating FABRIC memory needs a checkpoint
        // layer that allows it and a subscription to the IMEX channel.
        throw std::runtime_error(std::string("Failed to create exportable CUDA VMM memory: ") +
                                 cu_error_string(status));
    }
    if (cuMemMap(ptr_, size_, 0, handle_, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to map exportable CUDA VMM memory");
    }
    vmm_mapped_ = true;
    if (cuMemSetAccess(ptr_, size_, &alloc_ctx().access_desc, 1) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to set CUDA memory access");
    }
    exportable_ = true;
    return true;
}

} // namespace nixl_ep
