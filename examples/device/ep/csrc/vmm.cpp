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
#include <string>

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

std::string
cu_error_string(CUresult status) {
    const char *name = nullptr;
    const char *msg = nullptr;
    if (cuGetErrorName(status, &name) != CUDA_SUCCESS || name == nullptr) {
        name = "CUDA_ERROR_UNKNOWN";
    }
    if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
        msg = "unknown CUDA driver error";
    }
    return std::string(name) + ": " + msg;
}

} // namespace

namespace nixl_ep {

struct vmm_region::cuda_alloc_ctx {
    bool vmm_supported;
    bool rdma_vmm_supported;
    bool fabric_supported;
    CUmemAllocationProp rdma_prop;
    CUmemAllocationProp fabric_prop;
    size_t rdma_granularity;
    size_t fabric_granularity;
    CUdevice device;
    CUmemAccessDesc access_desc = {};

    cuda_alloc_ctx()
        : vmm_supported(false),
          rdma_vmm_supported(false),
          fabric_supported(false),
          rdma_prop({}),
          fabric_prop({}),
          rdma_granularity(0),
          fabric_granularity(0) {
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

        int vmm_attr = 0;
        if (cuDeviceGetAttribute(&vmm_attr,
                                 CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                 device) != CUDA_SUCCESS ||
            !vmm_attr) {
            return;
        }
        vmm_supported = true;

        rdma_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        rdma_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        rdma_prop.location.id = device;
        rdma_prop.allocFlags.gpuDirectRDMACapable = 1;

        int rdma_vmm_attr = 0;
        if (cuDeviceGetAttribute(&rdma_vmm_attr,
                                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                 device) != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to query GPUDirect RDMA with VMM support attribute");
        }

        if (!rdma_vmm_attr) {
            return;
        }
        rdma_vmm_supported = true;

        CUresult status = cuMemGetAllocationGranularity(
            &rdma_granularity, &rdma_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to get CUDA RDMA allocation granularity: " +
                cu_error_string(status));
        }

        int fab = 0;
        if ((cuDeviceGetAttribute(&fab,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                  device) != CUDA_SUCCESS) ||
            (!fab)) {
            access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            return;
        }

        fabric_prop = rdma_prop;
        fabric_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        status = cuMemGetAllocationGranularity(
            &fabric_granularity, &fabric_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to get CUDA fabric allocation granularity: " +
                cu_error_string(status));
        }
        fabric_supported = true;

        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
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
    if (!ctx.vmm_supported) {
        throw std::runtime_error("CUDA VMM is not available");
    }
    if (!handle_) {
        throw std::runtime_error("CUDA VMM allocation handle is not available");
    }

    if (!fixed_addr.has_value()) {
        CUresult status = cuMemAddressReserve(&ptr_, size_, 0, 0, 0);
        if (status != CUDA_SUCCESS) {
            if (handle_) {
                warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
                handle_ = 0;
            }
            throw std::runtime_error(
                "Failed to reserve CUDA virtual address: " +
                cu_error_string(status));
        }
    } else {
        ptr_ = fixed_addr.value();
    }

    CUresult status = cuMemMap(ptr_, size_, 0, handle_, 0);
    if (status != CUDA_SUCCESS) {
        const bool release_va = !fixed_addr.has_value();
        if (handle_) {
            warn_cu_api(cuMemRelease(handle_), k_vmm_ctx, "cuMemRelease");
            handle_ = 0;
        }
        if (release_va && ptr_) {
            warn_cu_api(cuMemAddressFree(ptr_, size_), k_vmm_ctx, "cuMemAddressFree");
            ptr_ = 0;
        }
        throw std::runtime_error(
            "Failed to map CUDA VMM memory: " + cu_error_string(status));
    }
    vmm_mapped_ = true;

    status = cuMemSetAccess(ptr_, size_, &ctx.access_desc, 1);
    if (status != CUDA_SUCCESS) {
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
        throw std::runtime_error(
            "Failed to set CUDA memory access: " + cu_error_string(status));
    }
}

void
vmm_region::reserve_and_map(std::optional<CUdeviceptr> fixed_addr) {
    const auto &ctx = get_cuda_alloc_ctx();
    if (!ctx.rdma_vmm_supported) {
        throw std::runtime_error("CUDA VMM GPUDirect RDMA memory is required");
    }
    if (fabric_shareable_ && !ctx.fabric_supported) {
        throw std::runtime_error("CUDA VMM fabric memory is required");
    }

    const auto &prop = fabric_shareable_ ? ctx.fabric_prop : ctx.rdma_prop;
    const CUresult mem_create_status = cuMemCreate(&handle_, size_, &prop, 0);
    if (mem_create_status != CUDA_SUCCESS) {
        handle_ = 0;
        throw std::runtime_error(
            std::string("Failed to create CUDA VMM allocation") +
            (fabric_shareable_ ? " with fabric handle: " : ": ") +
            cu_error_string(mem_create_status));
    }

    map_handle(fixed_addr);
}

vmm_region::vmm_region(size_t size) : vmm_region(size, false) {}

vmm_region::vmm_region(size_t size, bool fabric_shareable) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: size must be non-zero");
    }

    const auto &ctx = get_cuda_alloc_ctx();

    if (!ctx.rdma_vmm_supported) {
        throw std::runtime_error(
            "CUDA VMM GPUDirect RDMA memory is required for NIXL-EP buffers");
    }
    if (fabric_shareable && !ctx.fabric_supported) {
        throw std::runtime_error(
            "CUDA VMM fabric memory is required for NIXL-EP NVL buffers");
    }

    fabric_shareable_ = fabric_shareable;
    const size_t granularity =
        fabric_shareable_ ? ctx.fabric_granularity : ctx.rdma_granularity;
    size_ = nixl_ep::align_up<size_t>(size, granularity);
    reserve_and_map(std::nullopt);
}

CUmemFabricHandle
vmm_region::export_fabric_handle() const {
    if (!handle_) {
        throw std::runtime_error("Cannot export an unmapped CUDA VMM allocation");
    }
    if (!fabric_shareable_) {
        throw std::runtime_error("Cannot export a non-fabric CUDA VMM allocation");
    }

    CUmemFabricHandle fabric_handle{};
    CUresult status = cuMemExportToShareableHandle(
        &fabric_handle, handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(
            "Failed to export CUDA VMM fabric handle: " + cu_error_string(status));
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
    region->fabric_shareable_ = true;
    region->size_ = nixl_ep::align_up<size_t>(size, ctx.fabric_granularity);

    CUmemFabricHandle handle_copy = fabric_handle;
    CUresult status = cuMemImportFromShareableHandle(
        &region->handle_, &handle_copy, CU_MEM_HANDLE_TYPE_FABRIC);
    if (status != CUDA_SUCCESS) {
        region->handle_ = 0;
        throw std::runtime_error(
            "Failed to import CUDA VMM fabric handle: " + cu_error_string(status));
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
