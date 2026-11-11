/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */

// The CUDA build alone defines NIXL_HAVE_CUDA_DRIVER_API, which guards the CUDA
// driver-API calls below. Any other build (ROCm, no GPU runtime) reports UNKNOWN, and
// topology discovery then keeps its behaviour for hosts of unknown coherence.

#include "libfabric_cuda_caps.h"
#include "common/nixl_log.h"

#ifdef NIXL_HAVE_CUDA_DRIVER_API
#include <cuda.h>
#endif

#ifdef NIXL_HAVE_CUDA_DRIVER_API
namespace {

const char *
cudaCapsErrorName(CUresult status) {
    const char *name = nullptr;
    if (cuGetErrorName(status, &name) != CUDA_SUCCESS || name == nullptr) {
        return "unknown CUDA error";
    }
    return name;
}

} // namespace
#endif

extern "C" nixlAccelCoherence
nixlCudaAccelCoherence(const char *pci_bus_id) {
#ifdef NIXL_HAVE_CUDA_DRIVER_API
    if (pci_bus_id == nullptr) {
        return nixlAccelCoherence::UNKNOWN;
    }

    // cuInit is idempotent and creates no context.
    CUresult status = cuInit(0);
    if (status != CUDA_SUCCESS) {
        NIXL_DEBUG << "cuInit failed (" << cudaCapsErrorName(status) << "); coherence of GPU "
                   << pci_bus_id << " is unknown";
        return nixlAccelCoherence::UNKNOWN;
    }

    CUdevice device = 0;
    status = cuDeviceGetByPCIBusId(&device, pci_bus_id);
    if (status != CUDA_SUCCESS) {
        NIXL_DEBUG << "cuDeviceGetByPCIBusId(" << pci_bus_id << ") failed ("
                   << cudaCapsErrorName(status) << "); coherence is unknown";
        return nixlAccelCoherence::UNKNOWN;
    }

    int uses_host_page_tables = 0;
    status = cuDeviceGetAttribute(&uses_host_page_tables,
                                  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                                  device);
    if (status != CUDA_SUCCESS) {
        NIXL_DEBUG << "Coherence query for GPU " << pci_bus_id << " failed ("
                   << cudaCapsErrorName(status) << ")";
        return nixlAccelCoherence::UNKNOWN;
    }

    NIXL_DEBUG << "GPU " << pci_bus_id << " is "
               << (uses_host_page_tables ? "coherent" : "not coherent") << " with the CPU";
    return uses_host_page_tables ? nixlAccelCoherence::COHERENT : nixlAccelCoherence::NOT_COHERENT;
#else
    (void)pci_bus_id;
    return nixlAccelCoherence::UNKNOWN;
#endif
}
