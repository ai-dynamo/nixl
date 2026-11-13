/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Small helpers used across this DPDK-free implementation of SPDK's env
 * interface: an aligned allocation wrapper and an error-logging macro that
 * routes into SPDK's own logging.
 */

#ifndef NIXL_SPDK_ENV_COMPAT_H
#define NIXL_SPDK_ENV_COMPAT_H

#include <stdlib.h>

#include "spdk/log.h"

#define ENV_ERRLOG(fmt, ...) SPDK_ERRLOG(fmt "\n", ##__VA_ARGS__)

static inline void *env_aligned_alloc(size_t alignment, size_t size)
{
	void *ptr = NULL;

	if (posix_memalign(&ptr, alignment, size) != 0) {
		return NULL;
	}
	return ptr;
}

#endif /* NIXL_SPDK_ENV_COMPAT_H */
