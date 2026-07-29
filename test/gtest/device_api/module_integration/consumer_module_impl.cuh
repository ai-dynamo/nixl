/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_DEVICE_API_CONSUMER_MODULE_IMPL_CUH
#define NIXL_TEST_DEVICE_API_CONSUMER_MODULE_IMPL_CUH

#include <cstring>

#include <cuda_runtime.h>
#include <nixl_device.cuh>

#include "consumer_module.h"

#ifndef NIXL_TEST_MODULE_CALL
#error "NIXL_TEST_MODULE_CALL must name this consumer module's host entry point"
#endif

static_assert(sizeof(DeviceApiTransferStatus) == sizeof(nixlGpuXferStatusH));
static_assert(alignof(DeviceApiTransferStatus) == alignof(nixlGpuXferStatusH));

namespace {

__global__ void
callDeviceApi(DeviceApiModuleRequest request, DeviceApiModuleResult *result) {
    *result = {};
    switch (request.action) {
    case DeviceApiModuleAction::DESCRIBE: {
        if (request.src == nullptr) {
            result->status = NIXL_ERR_INVALID_PARAM;
            return;
        }
        const auto *prefix = static_cast<const nixlProxyDeviceMemView *>(request.src);
        result->version = prefix->version;
        result->length = prefix->length;
        if (prefix->version == NIXL_PROXY_MEM_LIST_VERSION_V1) {
            result->runtime_identity = prefix->context.shutdown_word;
        }
        result->direct_ptr = nixlGetPtr(request.src, request.index);
        result->status = NIXL_SUCCESS;
        return;
    }
    case DeviceApiModuleAction::PUT: {
        nixlGpuXferStatusH transfer_status{};
        result->status = nixlPut(nixlMemViewElem{request.src, 0, 0},
                                 nixlMemViewElem{request.dst, 0, 0},
                                 0,
                                 0,
                                 0,
                                 &transfer_status);
        memcpy(&result->transfer_status, &transfer_status, sizeof(transfer_status));
        return;
    }
    case DeviceApiModuleAction::POLL: {
        nixlGpuXferStatusH transfer_status{};
        memcpy(&transfer_status, &request.transfer_status, sizeof(transfer_status));
        result->status = nixlGpuGetXferStatus(transfer_status);
        return;
    }
    }
    result->status = NIXL_ERR_INVALID_PARAM;
}

} // namespace

extern "C" cudaError_t
NIXL_TEST_MODULE_CALL(const DeviceApiModuleRequest *request, DeviceApiModuleResult *result) {
    if (request == nullptr || result == nullptr) {
        return cudaErrorInvalidValue;
    }

    DeviceApiModuleResult *device_result = nullptr;
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(&device_result), sizeof(*device_result));
    if (status != cudaSuccess) {
        return status;
    }

    callDeviceApi<<<1, 1>>>(*request, device_result);
    status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(result, device_result, sizeof(*result), cudaMemcpyDeviceToHost);
    }

    const cudaError_t free_status = cudaFree(device_result);
    return status == cudaSuccess ? free_status : status;
}

#endif // NIXL_TEST_DEVICE_API_CONSUMER_MODULE_IMPL_CUH
