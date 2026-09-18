/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include "cuda_common.h"

#include <array>
#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "common/nixl_log.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/oss/status_macros.h" // for ASSIGN_OR_RETURN, RETURN_IF_ERROR
#else
#include "rxdm_dxs_stub.h"
#endif
#include "tcpxo_common.h"

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>
#endif

namespace tcpxo {

absl::Status
InitCuda() {
#ifdef HAVE_CUDA
    // cuInit(0) initializes the driver APIs.
    return CudaDriverApiCall(cuInit(0));
#else
    return absl::UnavailableError("CUDA not available");
#endif
}

absl::StatusOr<GpuDev>
InitGpuDev(absl::string_view pci_addr) {
#ifdef HAVE_CUDA
    GpuDev gpu;
    gpu.pci_addr = std::string(pci_addr);
    const cudaError_t cuda_err = cudaDeviceGetByPCIBusId(&gpu.dev, pci_addr.data());
    if (cuda_err == cudaErrorInvalidDevice) {
        return absl::NotFoundError(
            absl::StrFormat("GPU PCI address %s is not visible to CUDA", pci_addr));
    }
    RETURN_IF_ERROR(CudaRuntimeApiCall(cuda_err));
    RETURN_IF_ERROR(CudaDriverApiCall(
        cuDeviceGetAttribute(&gpu.freq, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, gpu.dev)));
    return gpu;
#else
    return absl::UnavailableError("CUDA not available");
#endif
}

absl::StatusOr<int>
GetDeviceCount() {
#ifdef HAVE_CUDA
    int num_devices = -1;
    RETURN_IF_ERROR(CudaDriverApiCall(cuDeviceGetCount(&num_devices)));
    return num_devices;
#else
    return absl::UnavailableError("CUDA not available");
#endif
}

absl::StatusOr<std::string>
GetPciBusIdFromCudaDevId(int cuda_dev_id) {
#ifdef HAVE_CUDA
    // [domain]:[bus]:[device].[function] (13 characters + NUL)
    std::array<char, 14> pci_bus_id{};
    RETURN_IF_ERROR(CudaRuntimeApiCall(
        cudaDeviceGetPCIBusId(pci_bus_id.data(), pci_bus_id.size(), cuda_dev_id)));
    NIXL_DEBUG << "PCI address for CUDA device ID " << cuda_dev_id << ": " << pci_bus_id.data();
    return std::string(pci_bus_id.data());
#else
    return absl::UnavailableError("CUDA not available");
#endif
}

absl::Status
CheckDeviceCount(int max_devices) {
#ifndef TCPXO_STUB_RXDM_DXS
    ASSIGN_OR_RETURN(int num_devices, GetDeviceCount());
    if (num_devices > max_devices) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "Number of GPUs (%d) is larger than maximum number: %d", num_devices, max_devices));
    }
#endif
    return absl::OkStatus();
}

// Get DMA-BUF fd from memory.
absl::StatusOr<int>
GetDmabufFd(uint64_t cuda_dev_id, void *ptr, size_t size) {
    int fd = -1;
#ifdef HAVE_CUDA
    int curr_device = -1;

    RETURN_IF_ERROR(CudaRuntimeApiCall(cudaGetDevice(&curr_device)));

    if (static_cast<uint64_t>(curr_device) != cuda_dev_id) {
        RETURN_IF_ERROR(CudaRuntimeApiCall(cudaSetDevice(cuda_dev_id)));
    }
    absl::Cleanup device_cleanup = [&] { cudaSetDevice(curr_device); };

    RETURN_IF_ERROR(CudaDriverApiCall(cuMemGetHandleForAddressRange(
        &fd, reinterpret_cast<CUdeviceptr>(ptr), size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)));
#endif
    return fd;
}

// Get the base address and size of the underlying memory allocation.
absl::StatusOr<std::pair<void *, size_t>>
GetDmabufBase(uint64_t cuda_dev_id, void *ptr) {
#ifdef HAVE_CUDA
    CUdeviceptr base_ptr = 0;
    size_t base_size = 0;
    int curr_device = -1;

    RETURN_IF_ERROR(CudaRuntimeApiCall(cudaGetDevice(&curr_device)));

    if (static_cast<uint64_t>(curr_device) != cuda_dev_id) {
        RETURN_IF_ERROR(CudaRuntimeApiCall(cudaSetDevice(cuda_dev_id)));
    }
    absl::Cleanup device_cleanup = [&] { cudaSetDevice(curr_device); };

    RETURN_IF_ERROR(CudaDriverApiCall(
        cuMemGetAddressRange(&base_ptr, &base_size, reinterpret_cast<CUdeviceptr>(ptr))));
    return std::make_pair(reinterpret_cast<void *>(base_ptr), base_size);
#else
    return absl::UnavailableError("CUDA not available");
#endif
}

#ifdef HAVE_CUDA
absl::Status
CudaDriverApiCall(CUresult err) {
    if (err != CUDA_SUCCESS) {
        const char *name = nullptr;
        const char *reason = nullptr;
        if (cuGetErrorName(err, &name)) {
            return absl::InternalError(
                absl::StrFormat("Error: error getting error name from CU error %d", err));
        }
        if (cuGetErrorString(err, &reason)) {
            return absl::InternalError(
                absl::StrFormat("Error: error getting error string from CU error %d", err));
        }
        return absl::InternalError(
            absl::StrFormat("CUDA error detected! name: %s; string: %s", name, reason));
    }
    return absl::OkStatus();
}

absl::Status
CudaRuntimeApiCall(cudaError_t err) {
    if (err != cudaSuccess) {
        const char *name = cudaGetErrorName(err);
        const char *reason = cudaGetErrorString(err);
        if (name == nullptr || reason == nullptr) {
            return absl::InternalError(
                absl::StrFormat("Failed to get error name and reason from CUDA error %d", err));
        }
        return absl::InternalError(
            absl::StrFormat("CUDA error detected! name: %s; string: %s", name, reason));
    }
    return absl::OkStatus();
}
#endif

} // namespace tcpxo
