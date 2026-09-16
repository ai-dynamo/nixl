/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SPDK_SHARED_MEMORY_H
#define NIXL_SPDK_SHARED_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private backend/env interface, not installed or called by applications.
 * Discover and retain the backing FD through SPDK unregister notifications.
 * Caller must serialize mapping lifetime against registration and I/O.
 */
int nixl_spdk_register_shared(void *addr, size_t len);
int nixl_spdk_unregister_shared(void *addr, size_t len);

#ifdef __cplusplus
}
#endif
#endif
