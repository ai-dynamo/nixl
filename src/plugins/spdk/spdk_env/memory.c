/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"
#include "spdk/memory.h"

#include "env_internal.h"
#include "env_mem_map.h"



#ifdef __linux__
#include <numaif.h>
#endif

/* This environment has no IOMMU and does no PCI probing, so nothing can perform
 * physical-address DMA. Every translation fails; callers that need a physical
 * address (local PCIe devices) are unsupported by design.
 */
uint64_t spdk_vtophys(const void *buf, uint64_t *size)
{
	(void)buf;

	if (size) {
		*size = UINT64_MAX;
	}
	return SPDK_VTOPHYS_ERROR;
}

bool spdk_iommu_is_enabled(void)
{
	return false;
}

int32_t spdk_mem_get_numa_id(const void *buf, uint64_t *size)
{
	if (size) {
		*size = UINT64_MAX;
	}
	return SPDK_ENV_NUMA_ID_ANY;
}

int spdk_mem_get_fd_and_offset(void *vaddr, uint64_t *offset)
{
	(void)vaddr;
	(void)offset;

	return -ENOTSUP;
}

void mem_enforce_numa(void)
{
}

static int mem_reg_map_check_contiguous(uint64_t addr1, uint64_t addr2)
{
	assert(addr1 & REG_MAP_REGISTERED);
	if (!(addr2 & REG_MAP_REGISTERED)) {
		return 0;
	}

	/* addr2 is the start of a new registration */
	return !(addr2 & REG_MAP_NOTIFY_START);
}

int mem_map_init(void)
{
	const struct spdk_mem_map_ops reg_map_ops = {
		.notify_cb = NULL,
		.are_contiguous = mem_reg_map_check_contiguous,
	};

	g_mem_reg_map = spdk_mem_map_alloc(0, &reg_map_ops, NULL);
	if (g_mem_reg_map == NULL) {
		ENV_ERRLOG("Failed to allocate g_mem_reg_map");
		return -ENOMEM;
	}

	return 0;
}

void mem_map_fini(void)
{
	spdk_mem_map_free(&g_mem_reg_map);
}
