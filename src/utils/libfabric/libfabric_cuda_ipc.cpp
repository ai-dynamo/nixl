/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "libfabric_cuda_ipc.h"

#ifdef NIXL_HAVE_CUDA_DRIVER_API
#include <cuda.h>

#include <cstring>
#include <fstream>
#include <string>
#endif

namespace LibfabricUtils {

#ifdef NIXL_HAVE_CUDA_DRIVER_API

static_assert(sizeof(CUipcMemHandle) == sizeof(CudaIpcExport::handle));

namespace {

    const std::string &
    hostId() {
        static const std::string host_id = [] {
            std::string id;
            std::ifstream("/proc/sys/kernel/random/boot_id") >> id;
            return id;
        }();
        return host_id;
    }

} // namespace

bool
cudaIpcExportRange(const void *buffer, CudaIpcExport &out) {
    CUdeviceptr base;
    CUipcMemHandle handle;
    CUresult status = cuMemGetAddressRange(&base, nullptr, reinterpret_cast<CUdeviceptr>(buffer));
    if (status == CUDA_SUCCESS) {
        status = cuIpcGetMemHandle(&handle, base);
    }
    if (status != CUDA_SUCCESS) {
        return false;
    }

    out = CudaIpcExport{};
    std::memcpy(out.handle, handle.reserved, sizeof(out.handle));
    out.offset = reinterpret_cast<CUdeviceptr>(buffer) - base;
    hostId().copy(out.host_id, sizeof(out.host_id) - 1);
    return true;
}

std::shared_ptr<void>
cudaIpcImport(const CudaIpcExport &in) {
    if (hostId() != in.host_id) {
        return nullptr;
    }

    CUipcMemHandle handle;
    std::memcpy(handle.reserved, in.handle, sizeof(in.handle));
    CUdeviceptr base;
    const CUresult status = cuIpcOpenMemHandle(&base, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (status != CUDA_SUCCESS) {
        return nullptr;
    }
    return std::shared_ptr<void>(reinterpret_cast<void *>(base), [](void *ptr) {
        cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(ptr));
    });
}

#else // !NIXL_HAVE_CUDA_DRIVER_API

bool
cudaIpcExportRange(const void *, CudaIpcExport &) {
    return false;
}

std::shared_ptr<void>
cudaIpcImport(const CudaIpcExport &) {
    return nullptr;
}

#endif // NIXL_HAVE_CUDA_DRIVER_API

} // namespace LibfabricUtils
