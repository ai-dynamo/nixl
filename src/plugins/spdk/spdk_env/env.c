/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "env_internal.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"
#include "spdk/memory.h"
#include "spdk/queue.h"
#include "spdk/util.h"

#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif

#define CACHE_LINE_SIZE 64

struct hugepage_chunk;

struct alloc_header {
	void *raw;
	struct hugepage_chunk *chunk;
	size_t size;
	size_t align;
	uint32_t flags;
};

/* align must be a power of two: the mask below is meaningless otherwise.
 * spdk_malloc()/spdk_realloc() enforce this on the caller-supplied value.
 */
static inline size_t align_up(size_t val, size_t align)
{
	return (val + align - 1) & ~(align - 1);
}

static inline bool is_power_of_two(size_t val)
{
	return val != 0 && (val & (val - 1)) == 0;
}

static void *_alloc(size_t size, size_t align, int numa_id, uint32_t flags)
{
	struct alloc_header *hdr;
	void *raw, *ret;

	if (align < CACHE_LINE_SIZE) {
		align = CACHE_LINE_SIZE;
	}

	raw = env_aligned_alloc(align, size + align);
	if (raw == NULL) {
		return NULL;
	}

	ret = (char *)raw + align;
	hdr = (struct alloc_header *)((char *)ret - sizeof(struct alloc_header));
	hdr->raw = raw;
	hdr->chunk = NULL;
	hdr->size = size;
	hdr->align = align;
	hdr->flags = flags;

	return ret;
}

static struct alloc_header *_get_header(void *buf)
{
	return (struct alloc_header *)((char *)buf - sizeof(struct alloc_header));
}

/* ---------- DMA hugepage allocator ---------- */

struct hugepage_chunk {
	void *base;
	size_t size;
	size_t offset;
	int alloc_count;
	TAILQ_ENTRY(hugepage_chunk) link;
};

static pthread_mutex_t g_dma_mutex = PTHREAD_MUTEX_INITIALIZER;
static TAILQ_HEAD(, hugepage_chunk) g_dma_chunks = TAILQ_HEAD_INITIALIZER(g_dma_chunks);
static struct hugepage_chunk *g_dma_current;

/*
 * Offset into chunk such that chunk->base + offset is aligned to align and
 * leaves room for the header below it. The alignment is computed on the
 * absolute address, not on the offset: the hugepage mmap in
 * _dma_map_hugepages() falls back to an anonymous mapping that is only
 * page-aligned, so aligning the offset alone would return a buffer that does
 * not satisfy an align larger than 4 KiB.
 */
static size_t chunk_aligned_start(const struct hugepage_chunk *chunk, size_t align)
{
	uintptr_t base = (uintptr_t)chunk->base;
	uintptr_t want = base + chunk->offset + sizeof(struct alloc_header);

	return (size_t)(align_up(want, align) - base);
}

static struct hugepage_chunk *_dma_map_hugepages(size_t size)
{
	struct hugepage_chunk *chunk;
	void *addr;

	size = align_up(size, VALUE_2MB);

	addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
	if (addr == MAP_FAILED) {
		addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (addr == MAP_FAILED) {
			return NULL;
		}
	}

	if (spdk_mem_register(addr, size) != 0) {
		munmap(addr, size);
		return NULL;
	}

	chunk = calloc(1, sizeof(*chunk));
	if (!chunk) {
		spdk_mem_unregister(addr, size);
		munmap(addr, size);
		return NULL;
	}

	chunk->base = addr;
	chunk->size = size;
	TAILQ_INSERT_TAIL(&g_dma_chunks, chunk, link);

	return chunk;
}

static void *_dma_alloc(size_t size, size_t align, uint32_t flags)
{
	struct hugepage_chunk *chunk;
	struct alloc_header *hdr;
	void *ret;
	size_t aligned_start, end;

	if (align < CACHE_LINE_SIZE) {
		align = CACHE_LINE_SIZE;
	}

	pthread_mutex_lock(&g_dma_mutex);

	chunk = g_dma_current;
	if (chunk) {
		aligned_start = chunk_aligned_start(chunk, align);
		end = aligned_start + size;
		if (end <= chunk->size) {
			goto allocate;
		}
	}

	{
		size_t need = sizeof(struct alloc_header) + align + size;
		size_t chunk_size = (need > VALUE_2MB) ? align_up(need, VALUE_2MB) : VALUE_2MB;

		chunk = _dma_map_hugepages(chunk_size);
		if (!chunk) {
			pthread_mutex_unlock(&g_dma_mutex);
			return NULL;
		}
		g_dma_current = chunk;

		aligned_start = chunk_aligned_start(chunk, align);
		end = aligned_start + size;
	}

allocate:
	ret = (char *)chunk->base + aligned_start;
	hdr = (struct alloc_header *)((char *)ret - sizeof(struct alloc_header));
	hdr->raw = NULL;
	hdr->chunk = chunk;
	hdr->size = size;
	hdr->align = align;
	hdr->flags = flags;

	chunk->offset = end;
	chunk->alloc_count++;

	pthread_mutex_unlock(&g_dma_mutex);
	return ret;
}

/*
 * Chunks are bump-allocated and never reused internally: a chunk is released
 * only once every allocation inside it has been freed. A single long-lived
 * buffer therefore pins its whole chunk (at least 2 MiB). This suits the
 * plugin, which allocates its DMA buffers per registration rather than per
 * transfer; a workload that churned mixed-lifetime buffers would need a
 * per-chunk free list.
 */
static void _dma_free(struct alloc_header *hdr)
{
	struct hugepage_chunk *chunk = hdr->chunk;

	pthread_mutex_lock(&g_dma_mutex);

	chunk->alloc_count--;
	if (chunk->alloc_count == 0) {
		if (g_dma_current == chunk) {
			g_dma_current = NULL;
		}
		TAILQ_REMOVE(&g_dma_chunks, chunk, link);
		pthread_mutex_unlock(&g_dma_mutex);

		spdk_mem_unregister(chunk->base, chunk->size);
		munmap(chunk->base, chunk->size);
		free(chunk);
		return;
	}

	pthread_mutex_unlock(&g_dma_mutex);
}

static void *_realloc(void *buf, size_t size, size_t align)
{
	struct alloc_header *old_hdr;
	void *new_buf;

	old_hdr = _get_header(buf);

	if (align == 0) {
		align = old_hdr->align;
	}

	if (old_hdr->chunk) {
		new_buf = _dma_alloc(size, align, old_hdr->flags);
	} else {
		new_buf = _alloc(size, align, SPDK_ENV_NUMA_ID_ANY, old_hdr->flags);
	}
	if (!new_buf) {
		return NULL;
	}

	memcpy(new_buf, buf, old_hdr->size < size ? old_hdr->size : size);

	if (old_hdr->chunk) {
		_dma_free(old_hdr);
	} else {
		free(old_hdr->raw);
	}

	return new_buf;
}

static void *_dma_realloc(void *buf, size_t size, size_t align)
{
	struct alloc_header *old_hdr;
	void *new_buf;

	old_hdr = _get_header(buf);

	if (align == 0) {
		align = old_hdr->align;
	}

	new_buf = _dma_alloc(size, align, old_hdr->flags);
	if (!new_buf) {
		return NULL;
	}

	memcpy(new_buf, buf, old_hdr->size < size ? old_hdr->size : size);

	if (old_hdr->chunk) {
		_dma_free(old_hdr);
	} else {
		free(old_hdr->raw);
	}

	return new_buf;
}

static void _free(void *buf)
{
	struct alloc_header *hdr = _get_header(buf);

	if (hdr->chunk) {
		_dma_free(hdr);
	} else {
		free(hdr->raw);
	}
}

void *spdk_malloc(size_t size, size_t align, uint64_t *unused, int numa_id, uint32_t flags)
{
	if (unused || flags == 0) {
		return NULL;
	}

	/* align == 0 means "no preference"; the allocators raise it to
	 * CACHE_LINE_SIZE. Anything else must be a power of two, or align_up()
	 * inside them computes a bogus offset.
	 */
	if (align != 0 && !is_power_of_two(align)) {
		ENV_ERRLOG("Alignment %zu is not a power of two", align);
		return NULL;
	}

	if (flags & SPDK_MALLOC_DMA) {
		return _dma_alloc(size, align, flags);
	}

	return _alloc(size, align, numa_id, flags);
}

void *spdk_zmalloc(size_t size, size_t align, uint64_t *unused, int numa_id, uint32_t flags)
{
	void *buf;

	buf = spdk_malloc(size, align, unused, numa_id, flags);
	if (buf) {
		memset(buf, 0, size);
	}

	return buf;
}

void *spdk_realloc(void *buf, size_t size, size_t align)
{
	/* align == 0 here means "keep the original alignment". */
	if (align != 0 && !is_power_of_two(align)) {
		ENV_ERRLOG("Alignment %zu is not a power of two", align);
		return NULL;
	}

	if (!buf) {
		return spdk_malloc(size, align, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_SHARE);
	}

	return _realloc(buf, size, align);
}

void spdk_free(void *buf)
{
	if (!buf) {
		return;
	}

	_free(buf);
}

void *spdk_dma_malloc(size_t size, size_t align, uint64_t *unused)
{
	return spdk_malloc(size, align, unused, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
}

void *spdk_dma_malloc_socket(size_t size, size_t align, uint64_t *unused, int numa_id)
{
	return spdk_malloc(size, align, unused, numa_id, SPDK_MALLOC_DMA);
}

void *spdk_dma_zmalloc(size_t size, size_t align, uint64_t *unused)
{
	return spdk_zmalloc(size, align, unused, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
}

void *spdk_dma_zmalloc_socket(size_t size, size_t align, uint64_t *unused, int numa_id)
{
	return spdk_zmalloc(size, align, unused, numa_id, SPDK_MALLOC_DMA);
}

void *spdk_dma_realloc(void *buf, size_t size, size_t align, uint64_t *unused)
{
	if (unused) {
		return NULL;
	}

	if (!buf) {
		return spdk_malloc(size, align, NULL, SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
	}

	return _dma_realloc(buf, size, align);
}

void spdk_dma_free(void *buf)
{
	spdk_free(buf);
}

/* ---------- memzone ---------- */

struct spdk_memzone_entry {
	char name[256];
	void *addr;
	size_t len;
	TAILQ_ENTRY(spdk_memzone_entry) link;
};

static TAILQ_HEAD(, spdk_memzone_entry) g_memzones = TAILQ_HEAD_INITIALIZER(g_memzones);
static pthread_mutex_t g_memzone_lock = PTHREAD_MUTEX_INITIALIZER;

void *spdk_memzone_reserve(const char *name, size_t len, int numa_id, unsigned flags)
{
	return spdk_memzone_reserve_aligned(name, len, numa_id, flags, 0);
}

void *spdk_memzone_reserve_aligned(const char *name, size_t len, int numa_id, unsigned flags, unsigned align)
{
	struct spdk_memzone_entry *entry;
	void *addr;

	pthread_mutex_lock(&g_memzone_lock);

	TAILQ_FOREACH(entry, &g_memzones, link)
	{
		if (strcmp(entry->name, name) == 0) {
			pthread_mutex_unlock(&g_memzone_lock);
			return NULL;
		}
	}

	addr = spdk_zmalloc(len, align, NULL, numa_id, SPDK_MALLOC_DMA | SPDK_MALLOC_SHARE);
	if (!addr) {
		pthread_mutex_unlock(&g_memzone_lock);
		return NULL;
	}

	entry = calloc(1, sizeof(*entry));
	if (!entry) {
		spdk_free(addr);
		pthread_mutex_unlock(&g_memzone_lock);
		return NULL;
	}

	snprintf(entry->name, sizeof(entry->name), "%s", name);
	entry->addr = addr;
	entry->len = len;
	TAILQ_INSERT_TAIL(&g_memzones, entry, link);

	pthread_mutex_unlock(&g_memzone_lock);
	return addr;
}

void *spdk_memzone_lookup(const char *name)
{
	struct spdk_memzone_entry *entry;

	pthread_mutex_lock(&g_memzone_lock);
	TAILQ_FOREACH(entry, &g_memzones, link)
	{
		if (strcmp(entry->name, name) == 0) {
			pthread_mutex_unlock(&g_memzone_lock);
			return entry->addr;
		}
	}
	pthread_mutex_unlock(&g_memzone_lock);
	return NULL;
}

int spdk_memzone_free(const char *name)
{
	struct spdk_memzone_entry *entry;

	pthread_mutex_lock(&g_memzone_lock);
	TAILQ_FOREACH(entry, &g_memzones, link)
	{
		if (strcmp(entry->name, name) == 0) {
			TAILQ_REMOVE(&g_memzones, entry, link);
			spdk_free(entry->addr);
			free(entry);
			pthread_mutex_unlock(&g_memzone_lock);
			return 0;
		}
	}
	pthread_mutex_unlock(&g_memzone_lock);
	return -1;
}

void spdk_memzone_dump(FILE *f)
{
	struct spdk_memzone_entry *entry;

	pthread_mutex_lock(&g_memzone_lock);
	TAILQ_FOREACH(entry, &g_memzones, link)
	{
		fprintf(f, "  zone \"%s\": addr=%p len=%zu\n", entry->name, entry->addr, entry->len);
	}
	pthread_mutex_unlock(&g_memzone_lock);
}

/* ---------- mempool ---------- */

struct spdk_mempool {
	char name[256];
	size_t ele_size;
	size_t count;
	void *base;
	size_t base_len;

	void **free_stack;
	size_t free_count;
	pthread_mutex_t lock;

	TAILQ_ENTRY(spdk_mempool) link;
};

static TAILQ_HEAD(, spdk_mempool) g_mempools = TAILQ_HEAD_INITIALIZER(g_mempools);
static pthread_mutex_t g_mempool_lock = PTHREAD_MUTEX_INITIALIZER;

struct spdk_mempool *spdk_mempool_create(const char *name, size_t count, size_t ele_size, size_t cache_size, int numa_id)
{
	return spdk_mempool_create_ctor(name, count, ele_size, cache_size, numa_id, NULL, NULL);
}

struct spdk_mempool *spdk_mempool_create_ctor(const char *name,
					      size_t count,
					      size_t ele_size,
					      size_t cache_size,
					      int numa_id,
					      spdk_mempool_obj_cb_t *obj_init,
					      void *obj_init_arg)
{
	struct spdk_mempool *mp;
	size_t aligned_ele_size;
	size_t i;

	mp = calloc(1, sizeof(*mp));
	if (!mp) {
		return NULL;
	}

	aligned_ele_size = align_up(ele_size, CACHE_LINE_SIZE);

	/* A wrapped base_len would under-allocate while the loop below still
	 * initialises count elements, writing past the end of the buffer.
	 */
	if (count != 0 && aligned_ele_size > SIZE_MAX / count) {
		ENV_ERRLOG("Mempool %s size overflow: %zu elements of %zu bytes",
			   name, count, aligned_ele_size);
		free(mp);
		return NULL;
	}

	snprintf(mp->name, sizeof(mp->name), "%s", name);
	mp->ele_size = ele_size;
	mp->count = count;
	pthread_mutex_init(&mp->lock, NULL);

	mp->base_len = aligned_ele_size * count;

	if (count > 0) {
		mp->base =
			spdk_zmalloc(mp->base_len, CACHE_LINE_SIZE, NULL, numa_id, SPDK_MALLOC_DMA | SPDK_MALLOC_SHARE);
		if (!mp->base) {
			pthread_mutex_destroy(&mp->lock);
			free(mp);
			return NULL;
		}

		mp->free_stack = calloc(count, sizeof(void *));
		if (!mp->free_stack) {
			spdk_free(mp->base);
			pthread_mutex_destroy(&mp->lock);
			free(mp);
			return NULL;
		}

		for (i = 0; i < count; i++) {
			void *obj = (char *)mp->base + i * aligned_ele_size;

			if (obj_init) {
				obj_init(mp, obj_init_arg, obj, (unsigned)i);
			}
			mp->free_stack[i] = obj;
		}
		mp->free_count = count;
	}

	pthread_mutex_lock(&g_mempool_lock);
	TAILQ_INSERT_TAIL(&g_mempools, mp, link);
	pthread_mutex_unlock(&g_mempool_lock);

	return mp;
}

char *spdk_mempool_get_name(struct spdk_mempool *mp)
{
	return mp->name;
}

void spdk_mempool_free(struct spdk_mempool *mp)
{
	if (!mp) {
		return;
	}

	pthread_mutex_lock(&g_mempool_lock);
	TAILQ_REMOVE(&g_mempools, mp, link);
	pthread_mutex_unlock(&g_mempool_lock);

	spdk_free(mp->base);
	free(mp->free_stack);
	pthread_mutex_destroy(&mp->lock);
	free(mp);
}

void *spdk_mempool_get(struct spdk_mempool *mp)
{
	void *ele = NULL;

	pthread_mutex_lock(&mp->lock);
	if (mp->free_count > 0) {
		ele = mp->free_stack[--mp->free_count];
	}
	pthread_mutex_unlock(&mp->lock);

	return ele;
}

int spdk_mempool_get_bulk(struct spdk_mempool *mp, void **ele_arr, size_t count)
{
	size_t i;

	pthread_mutex_lock(&mp->lock);
	if (mp->free_count < count) {
		pthread_mutex_unlock(&mp->lock);
		return -ENOMEM;
	}
	for (i = 0; i < count; i++) {
		ele_arr[i] = mp->free_stack[--mp->free_count];
	}
	pthread_mutex_unlock(&mp->lock);

	return 0;
}

void spdk_mempool_put(struct spdk_mempool *mp, void *ele)
{
	pthread_mutex_lock(&mp->lock);
	assert(mp->free_count < mp->count);
	mp->free_stack[mp->free_count++] = ele;
	pthread_mutex_unlock(&mp->lock);
}

void spdk_mempool_put_bulk(struct spdk_mempool *mp, void **ele_arr, size_t count)
{
	size_t i;

	pthread_mutex_lock(&mp->lock);
	for (i = 0; i < count; i++) {
		assert(mp->free_count < mp->count);
		mp->free_stack[mp->free_count++] = ele_arr[i];
	}
	pthread_mutex_unlock(&mp->lock);
}

size_t spdk_mempool_count(const struct spdk_mempool *pool)
{
	return pool->free_count;
}

uint32_t spdk_mempool_obj_iter(struct spdk_mempool *mp, spdk_mempool_obj_cb_t obj_cb, void *obj_cb_arg)
{
	size_t aligned_ele_size = align_up(mp->ele_size, CACHE_LINE_SIZE);
	uint32_t i;

	for (i = 0; i < mp->count; i++) {
		void *obj = (char *)mp->base + i * aligned_ele_size;
		obj_cb(mp, obj_cb_arg, obj, i);
	}

	return (uint32_t)mp->count;
}

uint32_t spdk_mempool_mem_iter(struct spdk_mempool *mp, spdk_mempool_mem_cb_t mem_cb, void *mem_cb_arg)
{
	if (mp->base && mp->base_len > 0) {
		mem_cb(mp, mem_cb_arg, mp->base, SPDK_VTOPHYS_ERROR, mp->base_len, 0);
		return 1;
	}

	return 0;
}

struct spdk_mempool *spdk_mempool_lookup(const char *name)
{
	struct spdk_mempool *mp;

	pthread_mutex_lock(&g_mempool_lock);
	TAILQ_FOREACH(mp, &g_mempools, link)
	{
		if (strcmp(mp->name, name) == 0) {
			pthread_mutex_unlock(&g_mempool_lock);
			return mp;
		}
	}
	pthread_mutex_unlock(&g_mempool_lock);
	return NULL;
}

bool spdk_process_is_primary(void)
{
	return true;
}

uint64_t spdk_get_ticks(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t spdk_get_ticks_hz(void)
{
	return 1000000000ULL;
}

void spdk_delay_us(unsigned int us)
{
	struct timespec ts;

	ts.tv_sec = us / 1000000;
	ts.tv_nsec = (us % 1000000) * 1000;
	nanosleep(&ts, NULL);
}

void spdk_pause(void)
{
#if defined(__x86_64__)
	__asm volatile("pause");
#elif defined(__aarch64__)
	__asm volatile("yield");
#endif
}

/* ---------- ring ----------
 *
 * Lock-free MPMC ring using DPDK's two-pointer scheme (see rte_ring_c11_pvt.h).
 * Each side has a (publication, reservation) pair:
 *
 *   tail       - producer publication: consumers read up to here.
 *   rsvd_tail  - producer reservation: next free slot for producers to claim.
 *   head       - consumer publication: producers see slots reusable up to here.
 *   rsvd_head  - consumer reservation: next slot for consumers to claim.
 *
 * Layout: producer-written fields (tail, rsvd_tail) and consumer-written fields
 * (head, rsvd_head) live on separate cache lines to eliminate producer/consumer false
 * sharing, matching DPDK's rte_ring_headtail layout.
 */
struct spdk_ring {
	uint32_t size;
	uint32_t mask;
	enum spdk_ring_type type;
	volatile uint32_t rsvd_tail __attribute__((aligned(CACHE_LINE_SIZE)));
	volatile uint32_t tail;
	volatile uint32_t rsvd_head __attribute__((aligned(CACHE_LINE_SIZE)));
	volatile uint32_t head;
	void *ring[];
};

struct spdk_ring *spdk_ring_create(enum spdk_ring_type type, size_t count, int numa_id)
{
	struct spdk_ring *ring;
	size_t alloc_size;
	size_t power;

	if (count == 0) {
		return NULL;
	}

	/*
	 * Round up to next power of 2. Bail out if the rounded value would
	 * not fit in ring->size/mask (uint32_t) or if the resulting alloc_size
	 * would overflow size_t.
	 */
	power = 1;
	while (power < count) {
		if (power > ((size_t)UINT32_MAX >> 1)) {
			return NULL;
		}
		power <<= 1;
	}

	if (power > (SIZE_MAX - sizeof(struct spdk_ring)) / sizeof(void *)) {
		return NULL;
	}
	alloc_size = sizeof(struct spdk_ring) + power * sizeof(void *);
	ring = spdk_zmalloc(alloc_size, CACHE_LINE_SIZE, NULL, numa_id, SPDK_MALLOC_DMA | SPDK_MALLOC_SHARE);
	if (!ring) {
		return NULL;
	}

	ring->size = (uint32_t)power;
	ring->mask = (uint32_t)(power - 1);
	ring->type = type;
	ring->rsvd_head = 0;
	ring->head = 0;
	ring->rsvd_tail = 0;
	ring->tail = 0;

	return ring;
}

void spdk_ring_free(struct spdk_ring *ring)
{
	spdk_free(ring);
}

size_t spdk_ring_count(struct spdk_ring *ring)
{
	uint32_t head = __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE);
	uint32_t tail = __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE);

	return tail - head;
}

size_t spdk_ring_enqueue(struct spdk_ring *ring, void **objs, size_t count, size_t *free_space)
{
	uint32_t head, tail, free_entries, i;

	if (ring->type == SPDK_RING_TYPE_SP_SC) {
		head = __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE);
		tail = ring->tail;

		free_entries = ring->size - (tail - head);
		if (count > free_entries) {
			count = free_entries;
		}

		for (i = 0; i < count; i++) {
			ring->ring[(tail + i) & ring->mask] = objs[i];
		}

		__atomic_store_n(&ring->tail, tail + (uint32_t)count, __ATOMIC_RELEASE);
	} else {
		/* MP enqueue: DPDK two-pointer scheme (see file header comment). */
		uint32_t tail_old, tail_new;

		do {
			tail_old = __atomic_load_n(&ring->rsvd_tail, __ATOMIC_ACQUIRE);
			head = __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE);

			free_entries = ring->size - (tail_old - head);
			if (count > free_entries) {
				count = free_entries;
			}
			if (count == 0) {
				if (free_space) {
					*free_space = 0;
				}
				return 0;
			}

			tail_new = tail_old + (uint32_t)count;
		} while (!__atomic_compare_exchange_n(&ring->rsvd_tail,
						      &tail_old,
						      tail_new,
						      false,
						      __ATOMIC_ACQ_REL,
						      __ATOMIC_ACQUIRE));

		for (i = 0; i < count; i++) {
			ring->ring[(tail_old + i) & ring->mask] = objs[i];
		}

		/* Wait for prior producers to publish, then publish ourselves. */
		while (__atomic_load_n(&ring->tail, __ATOMIC_RELAXED) != tail_old) {
			spdk_pause();
		}
		__atomic_store_n(&ring->tail, tail_new, __ATOMIC_RELEASE);
	}

	if (free_space) {
		*free_space = ring->size - (ring->tail - __atomic_load_n(&ring->head, __ATOMIC_ACQUIRE));
	}

	return count;
}

size_t spdk_ring_dequeue(struct spdk_ring *ring, void **objs, size_t count)
{
	uint32_t head, tail, avail, i;

	if (ring->type == SPDK_RING_TYPE_SP_SC || ring->type == SPDK_RING_TYPE_MP_SC) {
		tail = __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE);
		head = ring->head;

		avail = tail - head;
		if (count > avail) {
			count = avail;
		}

		for (i = 0; i < count; i++) {
			objs[i] = ring->ring[(head + i) & ring->mask];
		}

		__atomic_store_n(&ring->head, head + (uint32_t)count, __ATOMIC_RELEASE);
	} else {
		/* MC dequeue: DPDK two-pointer scheme (see file header comment). */
		uint32_t head_old, head_new;

		do {
			head_old = __atomic_load_n(&ring->rsvd_head, __ATOMIC_ACQUIRE);
			tail = __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE);

			avail = tail - head_old;
			if (count > avail) {
				count = avail;
			}
			if (count == 0) {
				return 0;
			}

			head_new = head_old + (uint32_t)count;
		} while (!__atomic_compare_exchange_n(&ring->rsvd_head,
						      &head_old,
						      head_new,
						      false,
						      __ATOMIC_ACQ_REL,
						      __ATOMIC_ACQUIRE));

		for (i = 0; i < count; i++) {
			objs[i] = ring->ring[(head_old + i) & ring->mask];
		}

		/* Wait for prior consumers to publish, then publish ourselves. */
		while (__atomic_load_n(&ring->head, __ATOMIC_RELAXED) != head_old) {
			spdk_pause();
		}
		__atomic_store_n(&ring->head, head_new, __ATOMIC_RELEASE);
	}

	return count;
}

int spdk_get_tid(void)
{
	return (int)syscall(SYS_gettid);
}
