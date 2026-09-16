/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "env_internal.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"
#include "spdk/cpuset.h"
#include "spdk/string.h"

#include <dirent.h>

#define MAX_CORES 128

#define THREAD_SIBLINGS_FILE "/sys/devices/system/cpu/cpu%d/topology/thread_siblings"

/*
 * Comma-separated 32-bit hex mask: 9 chars per 32 CPUs. 1024 bytes accommodates
 * over 3500 CPUs, far above any realistic system this code runs on.
 */
#define SMT_CPUSET_LINE_SIZE 1024

static uint32_t g_cores[MAX_CORES];
static uint32_t g_core_count;
static uint32_t g_main_core;

static __thread uint32_t t_core_id = SPDK_ENV_LCORE_ID_ANY;

struct worker_thread {
	pthread_t thread;
	thread_start_fn fn;
	void *arg;
	uint32_t core;
	bool running;
};

static struct worker_thread g_workers[MAX_CORES];

static int32_t g_numa_ids[MAX_CORES];
static uint32_t g_numa_count;

static int numa_id_cmp(const void *a, const void *b)
{
	int32_t x = *(const int32_t *)a;
	int32_t y = *(const int32_t *)b;

	return (x > y) - (x < y);
}

static void build_numa_list(void)
{
	uint32_t i, j;
	bool found;

	g_numa_count = 0;

	for (i = 0; i < g_core_count; i++) {
		int32_t nid = spdk_env_get_numa_id(g_cores[i]);

		found = false;
		for (j = 0; j < g_numa_count; j++) {
			if (g_numa_ids[j] == nid) {
				found = true;
				break;
			}
		}
		if (!found && g_numa_count < MAX_CORES) {
			g_numa_ids[g_numa_count] = nid;
			g_numa_count++;
		}
	}

	/* Ids are collected in core order, which is not id order. Sort so
	 * spdk_env_get_last_numa_id() returns the largest id and
	 * spdk_env_get_next_numa_id() walks them monotonically, matching the
	 * ordering threads_init() establishes for g_cores.
	 */
	qsort(g_numa_ids, g_numa_count, sizeof(g_numa_ids[0]), numa_id_cmp);
}

static int parse_core_mask(const char *mask)
{
	const char *p = mask;
	size_t len, i;

	if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
		p += 2;
	}

	len = strlen(p);
	if (len == 0) {
		return -EINVAL;
	}

	g_core_count = 0;
	/*
	 * Parse an arbitrary-width hex mask (not just 64 bits) so cores >= 64 are
	 * reachable. The rightmost hex digit holds cores 0-3; cores are emitted in
	 * ascending order and bounded by MAX_CORES.
	 */
	for (i = 0; i < len; i++) {
		char c = p[len - 1 - i];
		int nibble;

		if (c >= '0' && c <= '9') {
			nibble = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			nibble = c - 'a' + 10;
		} else if (c >= 'A' && c <= 'F') {
			nibble = c - 'A' + 10;
		} else {
			return -EINVAL;
		}

		for (int b = 0; b < 4; b++) {
			if (nibble & (1 << b)) {
				uint32_t core = (uint32_t)(i * 4 + b);

				if (core >= MAX_CORES) {
					ENV_ERRLOG("Core %u in mask exceeds the %d-core limit",
						   core, MAX_CORES);
					return -EINVAL;
				}
				if (g_core_count < MAX_CORES) {
					g_cores[g_core_count++] = core;
				}
			}
		}
	}

	if (g_core_count == 0) {
		return -EINVAL;
	}

	return 0;
}

static int parse_core_list(const char *list)
{
	const char *p = list;
	char *end;
	unsigned long val;

	if (*p == '[') {
		p++;
	}

	g_core_count = 0;

	while (*p != '\0' && *p != ']') {
		while (*p == ' ' || *p == ',') {
			p++;
		}
		if (*p == '\0' || *p == ']') {
			break;
		}

		val = strtoul(p, &end, 10);
		if (end == p) {
			return -EINVAL;
		}
		p = end;

		if (*p == '-') {
			unsigned long hi;

			p++;
			hi = strtoul(p, &end, 10);
			if (end == p || hi < val) {
				return -EINVAL;
			}
			p = end;

			if (hi >= MAX_CORES) {
				ENV_ERRLOG("Core %lu in list exceeds the %d-core limit",
					   hi, MAX_CORES);
				return -EINVAL;
			}

			while (val <= hi && g_core_count < MAX_CORES) {
				g_cores[g_core_count++] = (uint32_t)val;
				val++;
			}
		} else {
			/* Bound the id itself, not just the count: it later reaches
			 * CPU_SET(), which writes outside cpu_set_t for an index at
			 * or above CPU_SETSIZE.
			 */
			if (val >= MAX_CORES) {
				ENV_ERRLOG("Core %lu in list exceeds the %d-core limit",
					   val, MAX_CORES);
				return -EINVAL;
			}
			if (g_core_count < MAX_CORES) {
				g_cores[g_core_count++] = (uint32_t)val;
			}
		}
	}

	if (g_core_count == 0) {
		return -EINVAL;
	}

	return 0;
}

static int core_id_cmp(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a;
	uint32_t y = *(const uint32_t *)b;

	return (x > y) - (x < y);
}

int threads_init(const struct spdk_env_opts *opts)
{
	int rc;

	if (opts->lcore_map) {
		rc = parse_core_list(opts->lcore_map);
	} else if (opts->core_mask) {
		if (opts->core_mask[0] == '[') {
			rc = parse_core_list(opts->core_mask);
		} else {
			rc = parse_core_mask(opts->core_mask);
		}
	} else {
		return -EINVAL;
	}

	if (rc != 0) {
		ENV_ERRLOG("Failed to parse core mask/list");
		return rc;
	}

	/*
	 * spdk_env_get_first_core()/get_next_core() (and thus SPDK_ENV_FOREACH_CORE
	 * and the cpuset helpers) require g_cores to be ascending and de-duplicated;
	 * parse_core_list() preserves the user's input order and may contain
	 * duplicates.
	 */
	if (g_core_count > 1) {
		uint32_t r, w;

		qsort(g_cores, g_core_count, sizeof(g_cores[0]), core_id_cmp);
		w = 1;
		for (r = 1; r < g_core_count; r++) {
			if (g_cores[r] != g_cores[w - 1]) {
				g_cores[w++] = g_cores[r];
			}
		}
		g_core_count = w;
	}

	if (opts->main_core >= 0) {
		uint32_t i;

		/* The main core must be one the workers actually run on, or
		 * spdk_env_get_main_core() reports a core outside the core set.
		 */
		for (i = 0; i < g_core_count; i++) {
			if (g_cores[i] == (uint32_t)opts->main_core) {
				break;
			}
		}
		if (i == g_core_count) {
			ENV_ERRLOG("main_core %d is not in the core mask/list",
				   opts->main_core);
			return -EINVAL;
		}
		g_main_core = (uint32_t)opts->main_core;
	} else {
		g_main_core = g_cores[0];
	}

	t_core_id = g_main_core;

	memset(g_workers, 0, sizeof(g_workers));
	build_numa_list();

	return 0;
}

void threads_fini(void)
{
	/* Join first: spdk_env_thread_wait_all() walks g_workers up to
	 * g_core_count, so clearing the count first would strand any running
	 * worker and let it outlive the environment it runs against.
	 */
	spdk_env_thread_wait_all();

	g_core_count = 0;
	t_core_id = SPDK_ENV_LCORE_ID_ANY;
}

uint32_t spdk_env_get_core_count(void)
{
	return g_core_count;
}

uint32_t spdk_env_get_current_core(void)
{
	return t_core_id;
}

uint32_t spdk_env_get_main_core(void)
{
	return g_main_core;
}

uint32_t spdk_env_get_first_core(void)
{
	if (g_core_count == 0) {
		return UINT32_MAX;
	}

	return g_cores[0];
}

uint32_t spdk_env_get_last_core(void)
{
	uint32_t i;
	uint32_t last_core = UINT32_MAX;

	SPDK_ENV_FOREACH_CORE(i)
	{
		last_core = i;
	}

	assert(last_core != UINT32_MAX);

	return last_core;
}

uint32_t spdk_env_get_next_core(uint32_t prev_core)
{
	uint32_t i;

	for (i = 0; i < g_core_count; i++) {
		if (g_cores[i] > prev_core) {
			return g_cores[i];
		}
	}

	return UINT32_MAX;
}

int32_t spdk_env_get_numa_id(uint32_t core)
{
	char path[PATH_MAX];
	DIR *dir;
	struct dirent *ent;
	int32_t nid = SPDK_ENV_NUMA_ID_ANY;

	/*
	 * The NUMA node is exposed as a cpuN/nodeM symlink; use that rather than
	 * topology/physical_package_id, which is the socket/package id and differs
	 * from the NUMA node under sub-NUMA clustering (Intel SNC, AMD NPS > 1).
	 */
	snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", core);
	dir = opendir(path);
	if (dir == NULL) {
		return SPDK_ENV_NUMA_ID_ANY;
	}

	while ((ent = readdir(dir)) != NULL) {
		if (strncmp(ent->d_name, "node", 4) == 0 &&
		    ent->d_name[4] >= '0' && ent->d_name[4] <= '9') {
			nid = (int32_t)strtol(ent->d_name + 4, NULL, 10);
			break;
		}
	}

	closedir(dir);
	return nid;
}

int32_t spdk_env_get_first_numa_id(void)
{
	if (g_numa_count == 0) {
		return INT32_MAX;
	}

	return g_numa_ids[0];
}

int32_t spdk_env_get_last_numa_id(void)
{
	if (g_numa_count == 0) {
		return INT32_MAX;
	}

	return g_numa_ids[g_numa_count - 1];
}

int32_t spdk_env_get_next_numa_id(int32_t prev_numa_id)
{
	uint32_t i;

	for (i = 0; i < g_numa_count; i++) {
		if (g_numa_ids[i] == prev_numa_id && (i + 1) < g_numa_count) {
			return g_numa_ids[i + 1];
		}
	}

	return INT32_MAX;
}

void spdk_env_get_cpuset(struct spdk_cpuset *cpuset)
{
	uint32_t i;

	spdk_cpuset_zero(cpuset);
	SPDK_ENV_FOREACH_CORE(i)
	{
		spdk_cpuset_set_cpu(cpuset, i, true);
	}
}

/* Pure sysfs parsing of a core's SMT siblings. */
static bool env_core_get_smt_cpuset(struct spdk_cpuset *cpuset, uint32_t core)
{
	struct spdk_cpuset smt_siblings;
	char path[PATH_MAX];
	char line[SMT_CPUSET_LINE_SIZE];
	FILE *f;
	size_t len;
	bool valid = false;

	snprintf(path, sizeof(path), THREAD_SIBLINGS_FILE, core);
	f = fopen(path, "r");
	if (f == NULL) {
		ENV_ERRLOG("Could not fopen('%s'): %s", path, spdk_strerror(errno));
		return false;
	}
	if (fgets(line, sizeof(line), f) == NULL) {
		ENV_ERRLOG("Could not fgets() for '%s': %s", path, spdk_strerror(errno));
		goto ret;
	}

	len = strlen(line);
	if (len > 0 && line[len - 1] == '\n') {
		line[len - 1] = '\0';
	}
	if (spdk_cpuset_parse(&smt_siblings, line)) {
		ENV_ERRLOG("Could not parse '%s' from '%s'", line, path);
		goto ret;
	}

	valid = true;
	spdk_cpuset_or(cpuset, &smt_siblings);
ret:
	fclose(f);
	return valid;
}

bool spdk_env_core_get_smt_cpuset(struct spdk_cpuset *cpuset, uint32_t core)
{
	uint32_t i;

	spdk_cpuset_zero(cpuset);

	if (core != UINT32_MAX) {
		return env_core_get_smt_cpuset(cpuset, core);
	}

	SPDK_ENV_FOREACH_CORE(i)
	{
		if (!env_core_get_smt_cpuset(cpuset, i)) {
			return false;
		}
	}

	return true;
}

static void *worker_thread_entry(void *ctx)
{
	struct worker_thread *w = ctx;

	t_core_id = w->core;
	w->fn(w->arg);

	return NULL;
}

int spdk_env_thread_launch_pinned(uint32_t core, thread_start_fn fn, void *arg)
{
	struct worker_thread *w = NULL;
	cpu_set_t cpuset;
	pthread_attr_t attr;
	uint32_t i;
	int rc;

	for (i = 0; i < g_core_count; i++) {
		if (g_cores[i] == core) {
			w = &g_workers[i];
			break;
		}
	}

	if (w == NULL || w->running) {
		return -EINVAL;
	}

	w->fn = fn;
	w->arg = arg;
	w->core = core;

	CPU_ZERO(&cpuset);
	CPU_SET(core, &cpuset);

	pthread_attr_init(&attr);
	pthread_attr_setaffinity_np(&attr, sizeof(cpuset), &cpuset);

	rc = pthread_create(&w->thread, &attr, worker_thread_entry, w);
	pthread_attr_destroy(&attr);

	if (rc != 0) {
		return -rc;
	}

	w->running = true;
	return 0;
}

void spdk_env_thread_wait_all(void)
{
	uint32_t i;

	for (i = 0; i < g_core_count; i++) {
		if (g_workers[i].running) {
			pthread_join(g_workers[i].thread, NULL);
			g_workers[i].running = false;
		}
	}
}

void spdk_unaffinitize_thread(void)
{
	cpu_set_t mask;
	long num_cpus;
	int i;

	num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
	/* cpu_set_t holds CPU_SETSIZE bits; a larger machine would make CPU_SET()
	 * write past the end of mask.
	 */
	if (num_cpus > CPU_SETSIZE) {
		num_cpus = CPU_SETSIZE;
	}
	CPU_ZERO(&mask);
	for (i = 0; i < num_cpus; i++) {
		CPU_SET(i, &mask);
	}
	sched_setaffinity(0, sizeof(mask), &mask);
}

void *spdk_call_unaffinitized(void *cb(void *arg), void *arg)
{
	cpu_set_t orig;
	void *result;

	if (cb == NULL) {
		return NULL;
	}

	if (pthread_getaffinity_np(pthread_self(), sizeof(orig), &orig)) {
		return cb(arg);
	}

	spdk_unaffinitize_thread();
	result = cb(arg);

	pthread_setaffinity_np(pthread_self(), sizeof(orig), &orig);
	return result;
}
