/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_API_CPP_NIXL_DEVICE_BACKEND_TYPES_H
#define NIXL_SRC_API_CPP_NIXL_DEVICE_BACKEND_TYPES_H

#include <cstdint>

#include "nixl_types.h"

enum class nixl_device_exec_mode_t : uint8_t {
    NONE = 0,
    UCX_DIRECT = 1,
    PROXY = 2,
    GPUNETIO_DIRECT = 3,
};

struct nixlDeviceMemViewWrapper {
    // Deliberately a two-field POD: the installed header/consumer ABI is the
    // contract; release uses the host ownership record and never copies this
    // wrapper back from device memory.
    nixl_device_exec_mode_t execution_mode;
    nixlMemViewH backend_memview;
};

static_assert(sizeof(nixl_device_exec_mode_t) == 1);
static_assert(sizeof(nixlDeviceMemViewWrapper) == 2 * sizeof(void *));

#endif // NIXL_SRC_API_CPP_NIXL_DEVICE_BACKEND_TYPES_H
