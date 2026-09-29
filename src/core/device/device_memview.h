/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_DEVICE_DEVICE_MEMVIEW_H
#define NIXL_SRC_CORE_DEVICE_DEVICE_MEMVIEW_H

#include "nixl_device_backend_types.h"

class nixlDeviceAllocator;

[[nodiscard]] nixl_status_t
nixlDeviceMemViewAllocate(nixl_device_exec_mode_t execution_mode,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out,
                          nixlDeviceAllocator *allocator_override = nullptr) noexcept;

void
nixlDeviceMemViewFree(nixlMemViewH wrapper,
                      nixlDeviceAllocator *allocator_override = nullptr) noexcept;

#endif // NIXL_SRC_CORE_DEVICE_DEVICE_MEMVIEW_H
