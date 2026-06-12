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
#include <memory>
#include <optional>
#include <stdexcept>

#include "config.hpp"
#include "vmm.hpp"

namespace {

constexpr const char *k_vmm_ctx = "vmm_region";

/** Log a non-fatal warning if a CUDA driver API call failed (e.g. during teardown). */
void
warn_cu_api(CUresult status, const char *context, const char *operation) noexcept {
    if (status != CUDA_SUCCESS) {
        const char *msg = nullptr;
        if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
            msg = "unknown CUDA driver error";
        }
        std::cerr << "WARNING: " << context << " failed to " << operation << ": " << msg << '\n';
    }
}

} // namespace

namespace nixl_ep {

struct vmm_region::cuda_alloc_ctx {
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
            return;
        }

        int vmm_supported = 0;
        if (cuDeviceGetAttribute(&vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                 device) != CUDA_SUCCESS ||
            !vmm_supported) {
            return;
        }

        int fab = 0;
        if ((cuDeviceGetAttribute(&fab,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                  device) != CUDA_SUCCESS) ||
            (!fab)) {
            return;
        }

        int rdma_vmm_supported = 0;
        if (cuDeviceGetAttribute(&rdma_vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                 device) != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to query GPUDirect RDMA with VMM support attribute");
        }

        if (!rdma_vmm_supported) {
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

const vmm_region::cuda_alloc_ctx &
vmm_region::get_cuda_alloc_ctx() {
    static cuda_alloc_ctx ctx;
    return ctx;
}

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

void
vmm_region::map_handle(std::optional<CUdeviceptr> fixed_addr) {
    const auto &ctx = get_cuda_alloc_ctx();
    if (!ctx.fabric_supported) {
        throw std::runtime_error("CUDA VMM is not available");
    }
    if (!handle_) {
        throw std::runtime_error("CUDA VMM allocation handle is not available");
    }

    if (!fixed_addr.has_value()) {
        if (cuMemAddressReserve(&ptr_, size_, 0, 0, 0) != CUDA_SUCCESS) {
            if (handle_) {
                warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
                handle_ = 0;
            }
            throw std::runtime_error("Failed to reserve CUDA virtual address");
        }
    } else {
        ptr_ = fixed_addr.value();
    }

    if (cuMemMap(ptr_, size_, 0, handle_, 0) != CUDA_SUCCESS) {
        const bool release_va = !fixed_addr.has_value();
        if (handle_) {
            warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
            handle_ = 0;
        }
        if (release_va && ptr_) {
            warn_cu_api(cuMemAddressFree(ptr_, size_), k_vmm_ctx, "cuMemAddressFree");
            ptr_ = 0;
        }
        throw std::runtime_error("Failed to map CUDA VMM memory");
    }
    vmm_mapped_ = true;

    if (cuMemSetAccess(ptr_, size_, &ctx.access_desc, 1) != CUDA_SUCCESS) {
        if (vmm_mapped_) {
            warn_cu_api(cuMemUnmap(ptr_, size_), k_vmm_ctx, "cuMemUnmap");
            vmm_mapped_ = false;
        }
        if (handle_) {
            warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
            handle_ = 0;
        }
        if (!fixed_addr.has_value() && ptr_) {
            warn_cu_api(cuMemAddressFree(ptr_, size_), k_vmm_ctx, "cuMemAddressFree");
            ptr_ = 0;
        }
        throw std::runtime_error("Failed to set CUDA memory access");
    }
}

void
vmm_region::reserve_and_map(std::optional<CUdeviceptr> fixed_addr) {
    const auto &ctx = get_cuda_alloc_ctx();
    if (!ctx.fabric_supported) {
        throw std::runtime_error("CUDA VMM fabric memory is required");
    }

    const CUresult mem_create_status = cuMemCreate(&handle_, size_, &ctx.prop, 0);
    if (mem_create_status != CUDA_SUCCESS) {
        handle_ = 0;
        throw std::runtime_error("Failed to create CUDA VMM allocation");
    }

    map_handle(fixed_addr);
}

vmm_region::vmm_region(size_t size) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: size must be non-zero");
    }

    const auto &ctx = get_cuda_alloc_ctx();

    if (!ctx.fabric_supported) {
        throw std::runtime_error(
            "CUDA VMM fabric memory is required for NIXL-EP buffers");
    }

    size_ = nixl_ep::align_up<size_t>(size, ctx.granularity);
    reserve_and_map(std::nullopt);
}

CUmemFabricHandle
vmm_region::export_fabric_handle() const {
    if (!handle_) {
        throw std::runtime_error("Cannot export an unmapped CUDA VMM allocation");
    }

    CUmemFabricHandle fabric_handle{};
    if (cuMemExportToShareableHandle(
            &fabric_handle, handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to export CUDA VMM fabric handle");
    }
    return fabric_handle;
}

std::unique_ptr<vmm_region>
vmm_region::import_fabric_handle(const CUmemFabricHandle &fabric_handle, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: imported size must be non-zero");
    }

    const auto &ctx = get_cuda_alloc_ctx();
    if (!ctx.fabric_supported) {
        throw std::runtime_error(
            "CUDA VMM fabric memory is required for NIXL-EP buffers");
    }

    auto region = std::unique_ptr<vmm_region>(new vmm_region());
    region->size_ = nixl_ep::align_up<size_t>(size, ctx.granularity);

    CUmemFabricHandle handle_copy = fabric_handle;
    if (cuMemImportFromShareableHandle(
            &region->handle_, &handle_copy, CU_MEM_HANDLE_TYPE_FABRIC) !=
        CUDA_SUCCESS) {
        region->handle_ = 0;
        throw std::runtime_error("Failed to import CUDA VMM fabric handle");
    }

    try {
        region->map_handle(std::nullopt);
    } catch (...) {
        region->release();
        throw;
    }
    return region;
}

void
vmm_region::release_physical_preserve_va() {
    if (!can_preserve_va()) {
        throw std::runtime_error("CUDA VMM address preservation is not available");
    }
    if (vmm_mapped_) {
        if (cuMemUnmap(ptr_, size_) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to unmap CUDA VMM memory");
        }
        vmm_mapped_ = false;
    }
    if (handle_) {
        if (cuMemRelease(handle_) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to release CUDA VMM allocation");
        }
        handle_ = 0;
    }
}

void
vmm_region::remap_physical_at_preserved_va() {
    if (!can_preserve_va()) {
        throw std::runtime_error("CUDA VMM address preservation is not available");
    }
    if (vmm_mapped_) {
        return;
    }
    reserve_and_map(ptr_);
}

} // namespace nixl_ep
