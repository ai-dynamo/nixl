/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_DEVICE_API_CONSUMER_MODULE_H
#define NIXL_TEST_DEVICE_API_CONSUMER_MODULE_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "nixl_types.h"

enum class DeviceApiModuleAction : uint32_t {
    DESCRIBE,
    PUT,
    POLL,
};

struct alignas(16) DeviceApiTransferStatus {
    unsigned char storage[64];
};

struct DeviceApiModuleRequest {
    DeviceApiModuleAction action;
    nixlMemViewH src;
    nixlMemViewH dst;
    size_t index;
    DeviceApiTransferStatus transfer_status;
};

struct DeviceApiModuleResult {
    nixl_status_t status;
    uint16_t version;
    uint32_t length;
    const void *runtime_identity;
    void *direct_ptr;
    DeviceApiTransferStatus transfer_status;
};

using DeviceApiModuleCallFn =
    cudaError_t (*)(const DeviceApiModuleRequest *, DeviceApiModuleResult *);

extern "C" cudaError_t
nixlTestCallExecutableTuA(const DeviceApiModuleRequest *, DeviceApiModuleResult *);

extern "C" cudaError_t
nixlTestCallExecutableTuB(const DeviceApiModuleRequest *, DeviceApiModuleResult *);

extern "C" cudaError_t
nixlTestCallEarlyDsoA(const DeviceApiModuleRequest *, DeviceApiModuleResult *);

extern "C" cudaError_t
nixlTestCallEarlyDsoB(const DeviceApiModuleRequest *, DeviceApiModuleResult *);

#endif // NIXL_TEST_DEVICE_API_CONSUMER_MODULE_H
