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
#include "spdk/stdinc.h"
#include "spdk/assert.h"
#include "spdk/likely.h"
#include "spdk/util.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"

#include "env_mem_map.h"



#if DEBUG
#define DEBUG_PRINT(...) ENV_ERRLOG(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#define ADDR_INVALID ((uint64_t)-1)
#define _4KB_OFFSET(ptr) (((uintptr_t)(ptr)) & MASK_4KB)

struct spdk_mem_map *g_mem_reg_map;
struct spdk_mem_map_head g_spdk_mem_maps = TAILQ_HEAD_INITIALIZER(g_spdk_mem_maps);
pthread_mutex_t g_spdk_mem_map_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline uint64_t mem_map_translate(const struct spdk_mem_map *map, uint64_t vaddr, int *page_size)
{
	const struct map_1gb2mb *map_1gb2mb;
	const struct map_1gb4kb *map_1gb4kb;
	const struct map_2mb4kb *map_2mb4kb;
	uint64_t translation, vfn_4kb, vfn_2mb, idx_2mb, idx_1gb, idx_256tb;

	vfn_2mb = VFN_2MB(vaddr);
	idx_256tb = MAP_256TB_IDX(vfn_2mb);
	idx_1gb = MAP_1GB_IDX(vfn_2mb);

	map_1gb2mb = map->map_256tb.map[idx_256tb].map_1gb2mb;
	if (spdk_likely(map_1gb2mb != NULL)) {
		translation = map_1gb2mb->translation_2mb[idx_1gb];
		if (spdk_likely(translation != map->default_translation)) {
			*page_size = VALUE_2MB;
			return translation;
		}
	}

	map_1gb4kb = map->map_256tb.map[idx_256tb].map_1gb4kb;
	if (spdk_likely(map_1gb4kb != NULL)) {
		map_2mb4kb = map_1gb4kb->map[idx_1gb];
		if (spdk_likely(map_2mb4kb != NULL)) {
			vfn_4kb = VFN_4KB(vaddr);
			idx_2mb = MAP_2MB_IDX(vfn_4kb);
			*page_size = VALUE_4KB;

			return map_2mb4kb->translation_4kb[idx_2mb];
		}
	}

	*page_size = VALUE_2MB;
	return map->default_translation;
}

static bool mem_map_is_4kb_mapping(struct spdk_mem_map *map, uint64_t vaddr)
{
	int page_size;

	mem_map_translate(map, vaddr, &page_size);
	return page_size == VALUE_4KB;
}

static int mem_map_walk_region(struct spdk_mem_map *map,
			       uint64_t vaddr,
			       size_t size,
			       int (*callback)(struct spdk_mem_map *map, uint64_t addr, size_t sz, void *ctx),
			       void *ctx)
{
	uint64_t vfn_4kb, vfn_2mb;
	uint64_t vfn_4kb_end, vfn_2mb_end;
	int rc;

	vfn_4kb = VFN_4KB(vaddr);
	vfn_4kb_end = spdk_min(FN_2MB_TO_4KB(VFN_2MB(vaddr + MASK_2MB)), VFN_4KB(vaddr + size));
	while (vfn_4kb < vfn_4kb_end) {
		rc = callback(map, vaddr, VALUE_4KB, ctx);
		if (rc != 0) {
			return rc;
		}
		vaddr += VALUE_4KB;
		size -= VALUE_4KB;
		vfn_4kb++;
	}

	vfn_2mb = VFN_2MB(vaddr);
	vfn_2mb_end = VFN_2MB(vaddr + size);
	while (vfn_2mb < vfn_2mb_end) {
		rc = callback(map, vaddr, VALUE_2MB, ctx);
		if (rc != 0) {
			return rc;
		}
		vaddr += VALUE_2MB;
		size -= VALUE_2MB;
		vfn_2mb++;
	}

	vfn_4kb = VFN_4KB(vaddr);
	vfn_4kb_end = VFN_4KB(vaddr + size);
	while (vfn_4kb < vfn_4kb_end) {
		rc = callback(map, vaddr, VALUE_4KB, ctx);
		if (rc != 0) {
			return rc;
		}
		vaddr += VALUE_4KB;
		size -= VALUE_4KB;
		vfn_4kb++;
	}

	return 0;
}

static uint64_t mem_reg_map_next_region(uint64_t addr)
{
	uint64_t idx_256tb, idx_1gb, idx_2mb;
	uint64_t reg, vfn_2mb, vfn_4kb;
	int page_size;

	vfn_2mb = VFN_2MB(addr);
	vfn_4kb = VFN_4KB(addr);
	idx_256tb = MAP_256TB_IDX(vfn_2mb);
	idx_1gb = MAP_1GB_IDX(vfn_2mb);
	idx_2mb = MAP_2MB_IDX(vfn_4kb);
	for (; idx_256tb < MAP_256TB_SIZE; idx_256tb++) {
		if (!g_mem_reg_map->map_256tb.map[idx_256tb].map_1gb2mb &&
		    !g_mem_reg_map->map_256tb.map[idx_256tb].map_1gb4kb) {
			goto next_256tb;
		}

		for (; idx_1gb < MAP_1GB_SIZE; idx_1gb++) {
			addr = ADDR_FROM_IDX(idx_256tb, idx_1gb, idx_2mb);
			reg = mem_map_translate(g_mem_reg_map, addr, &page_size);

			if (reg & REG_MAP_NOTIFY_START) {
				assert(reg & REG_MAP_REGISTERED);
				return addr;
			}

			if (page_size == VALUE_4KB) {
				for (; idx_2mb < MAP_2MB_SIZE; idx_2mb++) {
					addr = ADDR_FROM_IDX(idx_256tb, idx_1gb, idx_2mb);
					reg = mem_map_translate(g_mem_reg_map, addr, &page_size);

					if (reg & REG_MAP_NOTIFY_START) {
						assert(reg & REG_MAP_REGISTERED);
						return addr;
					}
				}
			}

			idx_2mb = 0;
		}
next_256tb:
		idx_1gb = 0;
	}

	return ADDR_INVALID;
}

static int mem_map_notify_walk(struct spdk_mem_map *map, enum spdk_mem_map_notify_action action)
{
	uint64_t addr, fail_addr, size;
	int rc;

	if (!g_mem_reg_map) {
		return -EINVAL;
	}

	pthread_mutex_lock(&g_mem_reg_map->mutex);
	for (addr = mem_reg_map_next_region(0); addr != ADDR_INVALID; addr = mem_reg_map_next_region(addr)) {
		size = UINT64_MAX;
		spdk_mem_map_translate(g_mem_reg_map, addr, &size);
		rc = map->ops.notify_cb(map->cb_ctx, map, action, (void *)addr, size);
		if (rc != 0 && action == SPDK_MEM_MAP_NOTIFY_REGISTER) {
			goto err_unregister;
		}
		addr += size;
	}

	pthread_mutex_unlock(&g_mem_reg_map->mutex);
	return 0;

err_unregister:
	fail_addr = addr;
	for (addr = mem_reg_map_next_region(0); addr != ADDR_INVALID && addr != fail_addr;
	     addr = mem_reg_map_next_region(addr)) {
		size = UINT64_MAX;
		spdk_mem_map_translate(g_mem_reg_map, addr, &size);
		map->ops.notify_cb(map->cb_ctx, map, SPDK_MEM_MAP_NOTIFY_UNREGISTER, (void *)addr, size);
		addr += size;
	}

	pthread_mutex_unlock(&g_mem_reg_map->mutex);
	return rc;
}

static void mem_map_free(struct spdk_mem_map *map)
{
	struct map_1gb4kb *map_1gb4kb;
	size_t i, j;

	for (i = 0; i < SPDK_COUNTOF(map->map_256tb.map); i++) {
		free(map->map_256tb.map[i].map_1gb2mb);
		map_1gb4kb = map->map_256tb.map[i].map_1gb4kb;
		if (map_1gb4kb == NULL) {
			continue;
		}
		for (j = 0; j < SPDK_COUNTOF(map_1gb4kb->map); j++) {
			free(map_1gb4kb->map[j]);
		}
		free(map_1gb4kb);
	}
	pthread_mutex_destroy(&map->mutex);
	free(map);
}

struct spdk_mem_map *spdk_mem_map_alloc(uint64_t default_translation, const struct spdk_mem_map_ops *ops, void *cb_ctx)
{
	struct spdk_mem_map *map;
	int rc;

	map = calloc(1, sizeof(*map));
	if (map == NULL) {
		return NULL;
	}

	if (pthread_mutex_init(&map->mutex, NULL)) {
		free(map);
		return NULL;
	}

	map->default_translation = default_translation;
	map->cb_ctx = cb_ctx;
	if (ops) {
		map->ops = *ops;
	}

	if (ops && ops->notify_cb) {
		pthread_mutex_lock(&g_spdk_mem_map_mutex);
		rc = mem_map_notify_walk(map, SPDK_MEM_MAP_NOTIFY_REGISTER);
		if (rc != 0) {
			pthread_mutex_unlock(&g_spdk_mem_map_mutex);
			DEBUG_PRINT("Initial mem_map notify failed");
			mem_map_free(map);
			return NULL;
		}
		TAILQ_INSERT_TAIL(&g_spdk_mem_maps, map, tailq);
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
	}

	return map;
}

void spdk_mem_map_free(struct spdk_mem_map **pmap)
{
	struct spdk_mem_map *map;

	if (!pmap) {
		return;
	}

	map = *pmap;

	if (!map) {
		return;
	}

	if (map->ops.notify_cb) {
		pthread_mutex_lock(&g_spdk_mem_map_mutex);
		mem_map_notify_walk(map, SPDK_MEM_MAP_NOTIFY_UNREGISTER);
		TAILQ_REMOVE(&g_spdk_mem_maps, map, tailq);
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
	}

	mem_map_free(map);
	*pmap = NULL;
}

static int mem_check_region_unregistered(struct spdk_mem_map *map, uint64_t vaddr, size_t len, void *ctx)
{
	uint64_t reg, curlen;

	while (len > 0) {
		curlen = len;
		reg = spdk_mem_map_translate(map, vaddr, &curlen);
		if (reg & REG_MAP_REGISTERED) {
			return -EBUSY;
		}

		vaddr += curlen;
		len -= curlen;
	}

	return 0;
}

static int mem_check_region_registered(struct spdk_mem_map *map, uint64_t vaddr, size_t len, void *ctx)
{
	uint64_t reg, curlen;

	while (len > 0) {
		curlen = len;
		reg = spdk_mem_map_translate(map, vaddr, &curlen);
		if (!(reg & REG_MAP_REGISTERED)) {
			return -EINVAL;
		}

		vaddr += curlen;
		len -= curlen;
	}

	return 0;
}

static int mem_register_page(struct spdk_mem_map *map, uint64_t vaddr, size_t len, void *ctx)
{
	int *page = ctx;

	return spdk_mem_map_set_translation(map,
					    vaddr,
					    len,
					    (*page)++ == 0 ? REG_MAP_REGISTERED | REG_MAP_NOTIFY_START :
							     REG_MAP_REGISTERED);
}

static int mem_clear_page(struct spdk_mem_map *map, uint64_t vaddr, size_t len, void *ctx)
{
	(void)ctx;

	return spdk_mem_map_set_translation(map, vaddr, len, 0);
}

int spdk_mem_register(void *vaddr, size_t len)
{
	struct spdk_mem_map *map;
	int rc, page = 0;

	if (g_mem_reg_map == NULL) {
		DEBUG_PRINT("%s before spdk_env_init() or after spdk_env_fini()", __func__);
		return -EINVAL;
	}

	if ((uintptr_t)vaddr & ~MASK_256TB) {
		DEBUG_PRINT("invalid usermode virtual address %p", vaddr);
		return -EINVAL;
	}

	if (((uintptr_t)vaddr & MASK_4KB) || (len & MASK_4KB)) {
		DEBUG_PRINT("invalid %s parameters, vaddr=%p len=%ju", __func__, vaddr, len);
		return -EINVAL;
	}

	if (len == 0) {
		return 0;
	}

	pthread_mutex_lock(&g_spdk_mem_map_mutex);
	rc = mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_check_region_unregistered, NULL);
	if (rc != 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return rc;
	}

	rc = mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_register_page, &page);
	if (rc != 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return rc;
	}

	TAILQ_FOREACH(map, &g_spdk_mem_maps, tailq)
	{
		rc = map->ops.notify_cb(map->cb_ctx, map, SPDK_MEM_MAP_NOTIFY_REGISTER, vaddr, len);
		if (rc != 0) {
			struct spdk_mem_map *unwind;

			DEBUG_PRINT("failed to register vaddr %p to map %p, rc = %d", vaddr, map, rc);

			/* Undo the maps already notified and drop the registration
			 * itself. Callers release the mapping on failure (see
			 * _dma_map_hugepages() in env.c), so leaving either behind
			 * would keep a translation for memory that no longer exists.
			 */
			TAILQ_FOREACH(unwind, &g_spdk_mem_maps, tailq)
			{
				if (unwind == map) {
					break;
				}
				unwind->ops.notify_cb(unwind->cb_ctx,
						      unwind,
						      SPDK_MEM_MAP_NOTIFY_UNREGISTER,
						      vaddr,
						      len);
			}
			mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_clear_page, NULL);

			pthread_mutex_unlock(&g_spdk_mem_map_mutex);
			return rc;
		}
	}

	pthread_mutex_unlock(&g_spdk_mem_map_mutex);
	return 0;
}

static int mem_unregister_page(struct spdk_mem_map *map, uint64_t vaddr, size_t len, void *ctx)
{
	struct iovec *region = ctx;
	uint64_t off, reg;
	int rc;

	if (len > VALUE_4KB && mem_map_is_4kb_mapping(map, vaddr)) {
		assert(len == VALUE_2MB);
		for (off = 0; off < len; off += VALUE_4KB) {
			rc = mem_unregister_page(map, vaddr + off, VALUE_4KB, ctx);
			if (rc != 0) {
				return rc;
			}
		}
		return spdk_mem_map_set_translation(map, vaddr, len, 0);
	}

	reg = spdk_mem_map_translate(map, vaddr, NULL);
	spdk_mem_map_set_translation(map, vaddr, len, 0);
	if (region->iov_len > 0 && (reg & REG_MAP_NOTIFY_START)) {
		struct spdk_mem_map *notify;

		/* A separate variable: the loop must not clobber the caller's map. */
		TAILQ_FOREACH_REVERSE(notify, &g_spdk_mem_maps, spdk_mem_map_head, tailq)
		{
			rc = notify->ops.notify_cb(notify->cb_ctx,
						   notify,
						   SPDK_MEM_MAP_NOTIFY_UNREGISTER,
						   region->iov_base,
						   region->iov_len);
			if (rc != 0) {
				DEBUG_PRINT("failed to unregister vaddr %p from map %p, rc = %d",
					    region->iov_base,
					    notify,
					    rc);
			}
		}

		region->iov_base = (void *)vaddr;
		region->iov_len = len;
	} else {
		region->iov_len += len;
	}

	return 0;
}

int spdk_mem_unregister(void *vaddr, size_t len)
{
	struct spdk_mem_map *map;
	struct iovec region;
	int rc;
	uint64_t reg, newreg;

	if (g_mem_reg_map == NULL) {
		DEBUG_PRINT("%s before spdk_env_init() or after spdk_env_fini()", __func__);
		return -EINVAL;
	}

	if ((uintptr_t)vaddr & ~MASK_256TB) {
		DEBUG_PRINT("invalid usermode virtual address %p", vaddr);
		return -EINVAL;
	}

	if (((uintptr_t)vaddr & MASK_4KB) || (len & MASK_4KB)) {
		DEBUG_PRINT("invalid %s parameters, vaddr=%p len=%ju", __func__, vaddr, len);
		return -EINVAL;
	}

	pthread_mutex_lock(&g_spdk_mem_map_mutex);

	reg = spdk_mem_map_translate(g_mem_reg_map, (uint64_t)vaddr, NULL);
	if ((reg & REG_MAP_REGISTERED) && (reg & REG_MAP_NOTIFY_START) == 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return -ERANGE;
	}

	rc = mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_check_region_registered, NULL);
	if (rc != 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return rc;
	}

	newreg = spdk_mem_map_translate(g_mem_reg_map, (uint64_t)vaddr + len, NULL);
	if ((newreg & REG_MAP_NOTIFY_START) == 0 && (newreg & REG_MAP_REGISTERED)) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return -ERANGE;
	}

	region.iov_base = vaddr;
	region.iov_len = 0;
	rc = mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_unregister_page, &region);
	if (rc != 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return rc;
	}

	if (region.iov_len > 0) {
		TAILQ_FOREACH_REVERSE(map, &g_spdk_mem_maps, spdk_mem_map_head, tailq)
		{
			rc = map->ops.notify_cb(map->cb_ctx,
						map,
						SPDK_MEM_MAP_NOTIFY_UNREGISTER,
						region.iov_base,
						region.iov_len);
			if (rc != 0) {
				DEBUG_PRINT("failed to unregister vaddr %p from map %p, rc = %d",
					    region.iov_base,
					    map,
					    rc);
				pthread_mutex_unlock(&g_spdk_mem_map_mutex);
				return rc;
			}
		}
	}

	pthread_mutex_unlock(&g_spdk_mem_map_mutex);
	return 0;
}

int spdk_mem_reserve(void *vaddr, size_t len)
{
	struct spdk_mem_map *map;
	int rc;

	if (g_mem_reg_map == NULL) {
		DEBUG_PRINT("%s before spdk_env_init() or after spdk_env_fini()", __func__);
		return -EINVAL;
	}

	if ((uintptr_t)vaddr & ~MASK_256TB) {
		DEBUG_PRINT("invalid usermode virtual address %p", vaddr);
		return -EINVAL;
	}

	if (((uintptr_t)vaddr & MASK_4KB) || (len & MASK_4KB)) {
		DEBUG_PRINT("invalid %s parameters, vaddr=%p len=%ju", __func__, vaddr, len);
		return -EINVAL;
	}

	if (len == 0) {
		return 0;
	}

	pthread_mutex_lock(&g_spdk_mem_map_mutex);

	rc = mem_map_walk_region(g_mem_reg_map, (uint64_t)vaddr, len, mem_check_region_unregistered, NULL);
	if (rc != 0) {
		pthread_mutex_unlock(&g_spdk_mem_map_mutex);
		return rc;
	}

	spdk_mem_map_set_translation(g_mem_reg_map, (uint64_t)vaddr, len, g_mem_reg_map->default_translation);

	TAILQ_FOREACH(map, &g_spdk_mem_maps, tailq)
	{
		spdk_mem_map_set_translation(map, (uint64_t)vaddr, len, map->default_translation);
	}

	pthread_mutex_unlock(&g_spdk_mem_map_mutex);
	return 0;
}

static struct map_1gb2mb *mem_map_get_map_1gb2mb(struct spdk_mem_map *map, uint64_t vfn_2mb, bool alloc)
{
	struct map_1gb2mb *map_1gb2mb;
	uint64_t idx_256tb = MAP_256TB_IDX(vfn_2mb);
	size_t i;

	if (spdk_unlikely(idx_256tb >= SPDK_COUNTOF(map->map_256tb.map))) {
		return NULL;
	}

	map_1gb2mb = map->map_256tb.map[idx_256tb].map_1gb2mb;
	if (!map_1gb2mb && alloc) {
		pthread_mutex_lock(&map->mutex);

		/* Recheck to make sure nobody else got the mutex first. */
		map_1gb2mb = map->map_256tb.map[idx_256tb].map_1gb2mb;
		if (!map_1gb2mb) {
			map_1gb2mb = malloc(sizeof(struct map_1gb2mb));
			if (map_1gb2mb) {
				for (i = 0; i < SPDK_COUNTOF(map_1gb2mb->translation_2mb); i++) {
					map_1gb2mb->translation_2mb[i] = map->default_translation;
				}
				map->map_256tb.map[idx_256tb].map_1gb2mb = map_1gb2mb;
			}
		}

		pthread_mutex_unlock(&map->mutex);

		if (!map_1gb2mb) {
			DEBUG_PRINT("allocation failed");
			return NULL;
		}
	}

	return map_1gb2mb;
}

static struct map_1gb4kb *mem_map_get_map_1gb4kb(struct spdk_mem_map *map, uint64_t vfn_4kb, bool alloc)
{
	struct map_1gb4kb *map_1gb4kb;
	uint64_t vfn_2mb, idx_256tb;

	vfn_2mb = FN_4KB_TO_2MB(vfn_4kb);
	idx_256tb = MAP_256TB_IDX(vfn_2mb);
	if (idx_256tb >= SPDK_COUNTOF(map->map_256tb.map)) {
		return NULL;
	}

	map_1gb4kb = map->map_256tb.map[idx_256tb].map_1gb4kb;
	if (map_1gb4kb == NULL && alloc) {
		pthread_mutex_lock(&map->mutex);
		/* Recheck to make sure nobody else got the mutex first. */
		map_1gb4kb = map->map_256tb.map[idx_256tb].map_1gb4kb;
		if (map_1gb4kb == NULL) {
			map_1gb4kb = calloc(1, sizeof(*map_1gb4kb));
			/* Store only on success, so a failed allocation does not
			 * write NULL back over the slot.
			 */
			if (map_1gb4kb) {
				map->map_256tb.map[idx_256tb].map_1gb4kb = map_1gb4kb;
			}
		}
		pthread_mutex_unlock(&map->mutex);

		if (!map_1gb4kb) {
			DEBUG_PRINT("allocation failed");
			return NULL;
		}
	}

	return map_1gb4kb;
}

static struct map_2mb4kb *mem_map_get_map_2mb4kb(struct spdk_mem_map *map, uint64_t vfn_4kb, bool alloc)
{
	struct map_2mb4kb *map_2mb4kb;
	struct map_1gb4kb *map_1gb4kb;
	uint64_t vfn_2mb, idx_1gb, translation;
	int page_size;
	size_t i;

	map_1gb4kb = mem_map_get_map_1gb4kb(map, vfn_4kb, alloc);
	if (map_1gb4kb == NULL) {
		return NULL;
	}

	vfn_2mb = FN_4KB_TO_2MB(vfn_4kb);
	idx_1gb = MAP_1GB_IDX(vfn_2mb);
	map_2mb4kb = map_1gb4kb->map[idx_1gb];
	if (map_2mb4kb == NULL && alloc) {
		pthread_mutex_lock(&map->mutex);
		/* Recheck to make sure nobody else got the mutex first. */
		map_2mb4kb = map_1gb4kb->map[idx_1gb];
		if (map_2mb4kb == NULL) {
			map_2mb4kb = malloc(sizeof(*map_2mb4kb));
			if (map_2mb4kb != NULL) {
				translation = mem_map_translate(map, vfn_4kb << SHIFT_4KB, &page_size);
				for (i = 0; i < SPDK_COUNTOF(map_2mb4kb->translation_4kb); i++) {
					map_2mb4kb->translation_4kb[i] = translation;
				}
				map_1gb4kb->map[idx_1gb] = map_2mb4kb;
			}
		}
		pthread_mutex_unlock(&map->mutex);
	}

	return map_2mb4kb;
}

static int mem_map_set_4kb_translation(struct spdk_mem_map *map, uint64_t vaddr, uint64_t translation)
{
	struct map_2mb4kb *map_2mb4kb;
	struct map_1gb2mb *map_1gb2mb;
	uint64_t vfn_4kb, vfn_2mb;
	uint64_t idx_2mb, idx_1gb;

	vfn_4kb = VFN_4KB(vaddr);
	map_2mb4kb = mem_map_get_map_2mb4kb(map, vfn_4kb, true);
	if (!map_2mb4kb) {
		DEBUG_PRINT("could not get %p map", (void *)vaddr);
		return -ENOMEM;
	}

	idx_2mb = MAP_2MB_IDX(vfn_4kb);
	map_2mb4kb->translation_4kb[idx_2mb] = translation;

	vfn_2mb = FN_4KB_TO_2MB(vfn_4kb);
	map_1gb2mb = mem_map_get_map_1gb2mb(map, vfn_2mb, false);
	if (map_1gb2mb != NULL) {
		idx_1gb = MAP_1GB_IDX(vfn_2mb);
		map_1gb2mb->translation_2mb[idx_1gb] = map->default_translation;
	}

	return 0;
}

static int mem_map_set_2mb_translation(struct spdk_mem_map *map, uint64_t vaddr, uint64_t translation)
{
	struct map_2mb4kb *map_2mb4kb;
	struct map_1gb2mb *map_1gb2mb;
	uint64_t i, vfn_2mb, idx_1gb;

	vfn_2mb = VFN_2MB(vaddr);
	map_1gb2mb = mem_map_get_map_1gb2mb(map, vfn_2mb, true);
	if (!map_1gb2mb) {
		DEBUG_PRINT("could not get %p map", (void *)vaddr);
		return -ENOMEM;
	}

	idx_1gb = MAP_1GB_IDX(vfn_2mb);
	map_1gb2mb->translation_2mb[idx_1gb] = translation;

	map_2mb4kb = mem_map_get_map_2mb4kb(map, FN_2MB_TO_4KB(vfn_2mb), false);
	if (map_2mb4kb != NULL) {
		for (i = 0; i < SPDK_COUNTOF(map_2mb4kb->translation_4kb); i++) {
			map_2mb4kb->translation_4kb[i] = translation;
		}
	}

	return 0;
}

static int mem_map_set_page_translation(struct spdk_mem_map *map, uint64_t vaddr, size_t page_size, void *translation)
{
	switch (page_size) {
	case VALUE_4KB:
		return mem_map_set_4kb_translation(map, vaddr, (uint64_t)translation);
	case VALUE_2MB:
		return mem_map_set_2mb_translation(map, vaddr, (uint64_t)translation);
	default:
		assert(0 && "should never happen");
		return -EINVAL;
	}
}

int spdk_mem_map_set_translation(struct spdk_mem_map *map, uint64_t vaddr, uint64_t size, uint64_t translation)
{
	if ((uintptr_t)vaddr & ~MASK_256TB) {
		DEBUG_PRINT("invalid usermode virtual address %" PRIu64, vaddr);
		return -EINVAL;
	}

	if (((uintptr_t)vaddr & MASK_4KB) || (size & MASK_4KB)) {
		DEBUG_PRINT("invalid %s parameters, vaddr=%" PRIu64 " len=%" PRIu64, __func__, vaddr, size);
		return -EINVAL;
	}

	return mem_map_walk_region(map, vaddr, size, mem_map_set_page_translation, (void *)translation);
}

int spdk_mem_map_clear_translation(struct spdk_mem_map *map, uint64_t vaddr, uint64_t size)
{
	return spdk_mem_map_set_translation(map, vaddr, size, map->default_translation);
}

inline uint64_t spdk_mem_map_translate(const struct spdk_mem_map *map, uint64_t vaddr, uint64_t *size)
{
	uint64_t cur_size;
	uint64_t prev_translation;
	uint64_t orig_translation;
	uint64_t curr_translation;
	int page_size;

	if (spdk_unlikely(vaddr & ~MASK_256TB)) {
		DEBUG_PRINT("invalid usermode virtual address %p", (void *)vaddr);
		return map->default_translation;
	}

	curr_translation = mem_map_translate(map, vaddr, &page_size);
	cur_size = page_size - (page_size == VALUE_4KB ? _4KB_OFFSET(vaddr) : _2MB_OFFSET(vaddr));
	if (size == NULL || map->ops.are_contiguous == NULL || curr_translation == map->default_translation) {
		if (size != NULL) {
			*size = spdk_min(*size, cur_size);
		}
		return curr_translation;
	}

	prev_translation = orig_translation = curr_translation;
	vaddr += cur_size;
	while (cur_size < *size) {
		curr_translation = mem_map_translate(map, vaddr, &page_size);
		if (!map->ops.are_contiguous(prev_translation, curr_translation)) {
			break;
		}

		cur_size += page_size;
		vaddr += page_size;
		prev_translation = curr_translation;
	}

	*size = spdk_min(*size, cur_size);
	return orig_translation;
}
