/*
 * SPDX-FileCopyrightText: Copyright (c) 2017 Intel Corporation. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the SPDK project, licensed under the
 * BSD-3-Clause License. The modifications made by NVIDIA are licensed under the
 * Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_MEM_MAP_H
#define NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_MEM_MAP_H

#include "spdk/stdinc.h"
#include "spdk/env.h"
#include "spdk/queue.h"
#include "spdk/memory.h"

/* SHIFT_256TB, MASK_256TB, SHIFT_1GB, VALUE_1GB and MASK_1GB. */
#include "env_internal.h"

#define VFN_2MB(vaddr) ((vaddr) >> SHIFT_2MB)
#define VFN_4KB(vaddr) ((vaddr) >> SHIFT_4KB)

#define FN_2MB_TO_4KB(fn) ((fn) << (SHIFT_2MB - SHIFT_4KB))
#define FN_4KB_TO_2MB(fn) ((fn) >> (SHIFT_2MB - SHIFT_4KB))

#define MAP_256TB_IDX(vfn_2mb) ((vfn_2mb) >> (SHIFT_1GB - SHIFT_2MB))
#define MAP_1GB_IDX(vfn_2mb) ((vfn_2mb) & ((1ULL << (SHIFT_1GB - SHIFT_2MB)) - 1))
#define MAP_2MB_IDX(vfn_4kb) ((vfn_4kb) & ((1ULL << (SHIFT_2MB - SHIFT_4KB)) - 1))

#define MAP_256TB_SIZE (1ULL << (SHIFT_256TB - SHIFT_1GB))
#define MAP_1GB_SIZE (1ULL << (SHIFT_1GB - SHIFT_2MB))
#define MAP_2MB_SIZE (1ULL << (SHIFT_2MB - SHIFT_4KB))

#define ADDR_FROM_IDX(idx_256tb, idx_1gb, idx_2mb) \
	(((idx_256tb) << SHIFT_1GB) | ((idx_1gb) << SHIFT_2MB) | ((idx_2mb) << SHIFT_4KB))

#define REG_MAP_REGISTERED (1ULL << 62)
#define REG_MAP_NOTIFY_START (1ULL << 63)

#define VTOPHYS_4KB (1ULL << 63)
#define VTOPHYS_ADDR(paddr) ((paddr) & ~VTOPHYS_4KB)

struct map_2mb4kb {
	uint64_t translation_4kb[MAP_2MB_SIZE];
};

struct map_1gb2mb {
	uint64_t translation_2mb[MAP_1GB_SIZE];
};

struct map_1gb4kb {
	struct map_2mb4kb *map[MAP_1GB_SIZE];
};

struct map_256tb {
	struct {
		struct map_1gb2mb *map_1gb2mb;
		struct map_1gb4kb *map_1gb4kb;
	} map[MAP_256TB_SIZE];
};

struct spdk_mem_map {
	struct map_256tb map_256tb;
	pthread_mutex_t mutex;
	uint64_t default_translation;
	struct spdk_mem_map_ops ops;
	void *cb_ctx;
	TAILQ_ENTRY(spdk_mem_map) tailq;
};

TAILQ_HEAD(spdk_mem_map_head, spdk_mem_map);

extern struct spdk_mem_map *g_mem_reg_map;
extern struct spdk_mem_map_head g_spdk_mem_maps;
extern pthread_mutex_t g_spdk_mem_map_mutex;

#endif /* NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_MEM_MAP_H */
