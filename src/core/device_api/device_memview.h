/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H
#define NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H

#include <nixl_types.h>

[[nodiscard]] nixl_status_t
nixlDeviceMemViewAllocate(bool use_proxy,
                          nixlMemViewH backend_memview,
                          nixlMemViewH &wrapper_out) noexcept;

[[nodiscard]] nixl_status_t
nixlDeviceMemViewGetBackend(nixlMemViewH wrapper, nixlMemViewH &backend_out) noexcept;

void
nixlDeviceMemViewFree(nixlMemViewH wrapper) noexcept;

#endif // NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H
