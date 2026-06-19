/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vmm.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string cu_error_string(CUresult status) {
    const char* msg = nullptr;
    if (cuGetErrorString(status, &msg) != CUDA_SUCCESS || msg == nullptr) {
        return "unknown CUDA driver error";
    }
    return msg;
}

void check_cu(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + cu_error_string(status));
    }
}

void warn_cu(CUresult status, const char* operation) noexcept {
    if (status != CUDA_SUCCESS) {
        std::cerr << "WARNING: " << operation << " failed: " << cu_error_string(status) << '\n';
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

CUdevice current_or_explicit_device(int device_id) {
    CUdevice device = 0;
    if (device_id >= 0) {
        check_cu(cuDeviceGet(&device, device_id), "cuDeviceGet");
        return device;
    }
    check_cu(cuCtxGetDevice(&device), "cuCtxGetDevice");
    return device;
}

} // namespace

namespace nixl::cuda {

bool is_fabric_vmm_supported(int device_id) {
#if defined(HAVE_CUDA_FABRIC)
    try {
        const CUdevice device = current_or_explicit_device(device_id);
        int fabric_supported = 0;
        if (cuDeviceGetAttribute(&fabric_supported,
                                 CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                 device) != CUDA_SUCCESS ||
            !fabric_supported) {
            return false;
        }

        int rdma_vmm_supported = 0;
        if (cuDeviceGetAttribute(&rdma_vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                 device) != CUDA_SUCCESS) {
            return false;
        }
        return rdma_vmm_supported != 0;
    } catch (...) {
        return false;
    }
#else
    (void)device_id;
    return false;
#endif
}

fabric_vmm_region::fabric_vmm_region(std::size_t size,
                                     int device_id,
                                     bool require_fabric,
                                     std::uintptr_t address_hint)
    : requested_size_(size) {
    if (size == 0) {
        throw std::invalid_argument("fabric_vmm_region: size must be non-zero");
    }

    device_ = current_or_explicit_device(device_id);
    check_cu(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    primary_context_retained_ = true;
    set_current_context();

#if defined(HAVE_CUDA_FABRIC)
    if (is_fabric_vmm_supported(device_id)) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        std::size_t granularity = 0;
        check_cu(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
        mapped_size_ = align_up(size, granularity);

        check_cu(cuMemCreate(&handle_, mapped_size_, &prop, 0), "cuMemCreate");
        try {
            check_cu(cuMemAddressReserve(&ptr_,
                                         mapped_size_,
                                         0,
                                         static_cast<CUdeviceptr>(address_hint),
                                         0),
                     "cuMemAddressReserve");
            check_cu(cuMemMap(ptr_, mapped_size_, 0, handle_, 0), "cuMemMap");
            mapped_ = true;

            CUmemAccessDesc access_desc = {};
            access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id = device_;
            access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            check_cu(cuMemSetAccess(ptr_, mapped_size_, &access_desc, 1), "cuMemSetAccess");
            is_fabric_ = true;
            return;
        } catch (...) {
            release();
            throw;
        }
    }
#endif

    if (require_fabric) {
        throw std::runtime_error("CUDA fabric VMM allocation is not supported on this device");
    }

    mapped_size_ = size;
    check_cu(cuMemAlloc(&ptr_, size), "cuMemAlloc");
}

fabric_vmm_region::~fabric_vmm_region() {
    release();
}

void fabric_vmm_region::set_current_context() const {
    if (context_ != nullptr) {
        check_cu(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    }
}

void fabric_vmm_region::release() noexcept {
    if (context_ != nullptr) {
        warn_cu(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    }

    if (is_fabric_) {
        if (mapped_) {
            warn_cu(cuMemUnmap(ptr_, mapped_size_), "cuMemUnmap");
            mapped_ = false;
        }
        if (ptr_) {
            warn_cu(cuMemAddressFree(ptr_, mapped_size_), "cuMemAddressFree");
            ptr_ = 0;
        }
        if (handle_) {
            warn_cu(cuMemRelease(handle_), "cuMemRelease");
            handle_ = 0;
        }
        is_fabric_ = false;
    } else if (ptr_) {
        warn_cu(cuMemFree(ptr_), "cuMemFree");
        ptr_ = 0;
    }

    if (primary_context_retained_) {
        warn_cu(cuDevicePrimaryCtxRelease(device_), "cuDevicePrimaryCtxRelease");
        primary_context_retained_ = false;
        context_ = nullptr;
    }
}

} // namespace nixl::cuda
