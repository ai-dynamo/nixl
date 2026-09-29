// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#elif defined(HAVE_ROCM)
#include <hip/hip_runtime.h>
#endif

// Fill out (formatted "domain:bus:device.0") with the PCI bus ID of GPU gpu_id.
// Returns true on success. On non-GPU builds, or when the query fails, returns
// false and leaves out untouched.
inline bool
gpuGetPciBusId(int gpu_id, char *out, size_t out_len) {
#if defined(HAVE_CUDA)
    cudaDeviceProp prop;
    const cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess) {
        return false;
    }
    snprintf(out, out_len, "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    return true;
#elif defined(HAVE_ROCM)
    hipDeviceProp_t prop;
    const hipError_t err = hipGetDeviceProperties(&prop, gpu_id);
    if (err != hipSuccess) {
        return false;
    }
    snprintf(out, out_len, "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    return true;
#else
    (void)gpu_id;
    (void)out;
    (void)out_len;
    return false;
#endif
}
