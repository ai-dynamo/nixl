/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_BUFFER_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_BUFFER_H

#include <cstddef>

#include <nixl_types.h>

[[nodiscard]] nixl_status_t
nixlDeviceBufferAllocate(void **ptr, size_t size) noexcept;

[[nodiscard]] nixl_status_t
nixlDeviceBufferCopyHostToDevice(void *dst, const void *src, size_t size) noexcept;

[[nodiscard]] nixl_status_t
nixlDeviceBufferCopyDeviceToHost(void *dst, const void *src, size_t size) noexcept;

void
nixlDeviceBufferFree(void *ptr) noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_BUFFER_H
