/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

// The CUDA build alone defines NIXL_HAVE_CUDA_DRIVER_API, which guards the CUDA
// driver-API calls below; both the CUDA and the ROCm build define HAVE_CUDA.

#include "libfabric_cuda_dmabuf.h"
#include "common/nixl_log.h"

#include <unistd.h>

#ifdef NIXL_HAVE_CUDA_DRIVER_API
#include <cuda.h>
#include <sys/utsname.h>
#include <cstdio>
#endif

namespace LibfabricUtils {

#ifdef NIXL_HAVE_CUDA_DRIVER_API

namespace {

// Whether the kernel provides the RDMA dmabuf import ioctls that register an exported fd.
//
// The ioctls arrive in kernel 5.12 (torvalds/linux bfe0cc6), and the check runs once per
// process. A host on an older kernel registers by virtual address.
bool
kernelSupportsRdmaDmabuf() {
    static const bool supported = [] {
        struct utsname uts = {};
        if (uname(&uts) != 0) {
            NIXL_WARN << "uname() failed; assuming no kernel support for RDMA dmabuf import";
            return false;
        }
        int major = 0;
        int minor = 0;
        if (sscanf(uts.release, "%d.%d", &major, &minor) != 2) {
            NIXL_WARN << "Could not parse kernel release '" << uts.release
                      << "'; assuming no support for RDMA dmabuf import";
            return false;
        }
        const bool ok = (major > 5) || (major == 5 && minor >= 12);
        NIXL_DEBUG << "Kernel " << uts.release << (ok ? " supports" : " does not support")
                   << " RDMA dmabuf import (needs 5.12+)";
        return ok;
    }();
    return supported;
}

const char *
cudaErrorName(CUresult status) {
    const char *name = nullptr;
    if (cuGetErrorName(status, &name) != CUDA_SUCCESS || name == nullptr) {
        return "unknown CUDA error";
    }
    return name;
}

} // namespace

bool
cudaDmabufExportSupported(int device_id) {
    if (!kernelSupportsRdmaDmabuf()) {
        return false;
    }

    CUdevice device;
    CUresult status = cuDeviceGet(&device, device_id);
    if (status != CUDA_SUCCESS) {
        NIXL_DEBUG << "cuDeviceGet(" << device_id << ") failed: " << cudaErrorName(status)
                   << "; not using a dmabuf registration";
        return false;
    }

    // The attribute belongs to the device holding the memory being registered.
    int dmabuf_supported = 0;
    status = cuDeviceGetAttribute(&dmabuf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device);
    if (status != CUDA_SUCCESS) {
        NIXL_DEBUG << "Could not query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED on device "
                   << device_id << ": " << cudaErrorName(status)
                   << "; not using a dmabuf registration";
        return false;
    }

    return dmabuf_supported == 1;
}

bool
cudaDmabufExportRange(void *buffer,
                      size_t length,
                      int device_id,
                      bool pcie_mapping,
                      CudaDmabufExport &out) {
    out = CudaDmabufExport{};

    if (buffer == nullptr) {
        return false;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        NIXL_WARN << "sysconf(_SC_PAGESIZE) returned " << page_size
                  << "; cannot align a dmabuf export";
        return false;
    }

    uintptr_t aligned_base = 0;
    size_t aligned_length = 0;
    uint64_t offset = 0;
    cudaDmabufAlignRange(
        buffer, length, static_cast<size_t>(page_size), aligned_base, aligned_length, offset);

    unsigned long long flags = 0;
#ifdef HAVE_CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE
    if (pcie_mapping) {
        flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
    }
#else
    if (pcie_mapping) {
        // A build whose CUDA headers predate the flag exports the platform default
        // mapping, which is the single window on a platform offering one.
        NIXL_DEBUG << "CUDA headers predate CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE; "
                      "exporting with the platform default mapping";
    }
#endif

    int fd = -1;
    CUresult status = cuMemGetHandleForAddressRange(&fd,
                                                   static_cast<CUdeviceptr>(aligned_base),
                                                   aligned_length,
                                                   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                                   flags);

    bool obtained_pcie_mapping = (flags != 0);
    if ((status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_NOT_SUPPORTED) && flags != 0) {
        // A platform offering a single mapping rejects the PCIe flag, and its default
        // mapping is that same window, so the export retries with the default.
        NIXL_DEBUG << "cuMemGetHandleForAddressRange rejected the PCIe mapping flag ("
                   << cudaErrorName(status) << "); retrying with the default mapping";
        status = cuMemGetHandleForAddressRange(&fd,
                                              static_cast<CUdeviceptr>(aligned_base),
                                              aligned_length,
                                              CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                              0);
        obtained_pcie_mapping = false;
    }

    if (status != CUDA_SUCCESS || fd < 0) {
        NIXL_DEBUG << "cuMemGetHandleForAddressRange failed for buffer " << buffer << " length "
                   << length << " on device " << device_id << ": " << cudaErrorName(status)
                   << "; falling back to a virtual-address registration";
        return false;
    }

    out.fd = fd;
    out.offset = offset;
    out.base_addr = reinterpret_cast<void *>(aligned_base);
    out.pcie_mapping = obtained_pcie_mapping;

    NIXL_DEBUG << "Exported dmabuf fd " << out.fd << " for buffer " << buffer << " length "
               << length << " (aligned base " << out.base_addr << ", offset " << out.offset
               << ", aligned length " << aligned_length << ", "
               << (out.pcie_mapping ? "PCIe (BAR1) mapping" : "default mapping") << ")";
    return true;
}

#else // !NIXL_HAVE_CUDA_DRIVER_API

bool
cudaDmabufExportSupported(int device_id) {
    (void)device_id;
    return false;
}

bool
cudaDmabufExportRange(void *buffer,
                      size_t length,
                      int device_id,
                      bool pcie_mapping,
                      CudaDmabufExport &out) {
    (void)buffer;
    (void)length;
    (void)device_id;
    (void)pcie_mapping;
    out = CudaDmabufExport{};
    return false;
}

#endif // NIXL_HAVE_CUDA_DRIVER_API

void
cudaDmabufExportClose(CudaDmabufExport &to_close) {
    if (to_close.fd >= 0) {
        close(to_close.fd);
    }
    to_close = CudaDmabufExport{};
}

} // namespace LibfabricUtils
