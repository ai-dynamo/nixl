/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "env_internal.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"
#include "spdk/version.h"



static bool g_initialized;

static void env_copy_opts(struct spdk_env_opts *dst, const struct spdk_env_opts *src, size_t user_opts_size)
{
	spdk_env_opts_init(dst);
	memcpy(dst, src, offsetof(struct spdk_env_opts, opts_size));

#define SET_FIELD(field) \
	if (offsetof(struct spdk_env_opts, field) + sizeof(dst->field) <= user_opts_size) { \
		dst->field = src->field; \
	}

	SET_FIELD(enforce_numa);

#undef SET_FIELD
}

void spdk_env_opts_init(struct spdk_env_opts *opts)
{
	if (!opts) {
		return;
	}

	memset(opts, 0, sizeof(*opts));
	opts->name = "spdk";
	opts->core_mask = "0x1";
	opts->shm_id = -1;
	opts->mem_size = -1;
	opts->main_core = -1;
	opts->mem_channel = -1;
	opts->base_virtaddr = 0x200000000000;
	opts->opts_size = sizeof(*opts);
}

int spdk_env_init(const struct spdk_env_opts *opts_user)
{
	struct spdk_env_opts opts_local = {};
	struct spdk_env_opts *opts = &opts_local;
	size_t min_opts_size, user_opts_size;
	int rc;

	if (g_initialized) {
		if (opts_user != NULL) {
			ENV_ERRLOG("Invalid arguments to reinitialize SPDK env");
			return -EINVAL;
		}

		SPDK_PRINTF("Starting %s reinitialization...\n", SPDK_VERSION_STRING);
		pci_env_reinit();
		return 0;
	}

	if (opts_user == NULL) {
		ENV_ERRLOG("NULL arguments to initialize SPDK env");
		return -EINVAL;
	}

	min_opts_size = offsetof(struct spdk_env_opts, opts_size) + sizeof(opts->opts_size);
	user_opts_size = opts_user->opts_size;
	/* env_copy_opts() memcpy()s offsetof(opts_size) bytes out of the caller's
	 * structure, so a smaller one cannot be honoured by clamping: the copy
	 * would read past its end. Reject it instead.
	 */
	if (user_opts_size < min_opts_size) {
		ENV_ERRLOG("Invalid opts->opts_size %d too small", (int)opts_user->opts_size);
		return -EINVAL;
	}

	env_copy_opts(opts, opts_user, user_opts_size);

	if (opts->enforce_numa) {
		mem_enforce_numa();
	}

	SPDK_PRINTF("Starting %s initialization...\n", SPDK_VERSION_STRING);

	rc = threads_init(opts);
	if (rc < 0) {
		ENV_ERRLOG("threads_init() failed");
		return rc;
	}

	rc = pci_env_init();
	if (rc < 0) {
		ENV_ERRLOG("pci_env_init() failed");
		goto err_threads;
	}

	rc = mem_map_init();
	if (rc < 0) {
		ENV_ERRLOG("Failed to allocate mem_map");
		goto err_pci;
	}

	g_initialized = true;
	return 0;

	/* Unwind in reverse order. g_initialized stays false, so a later retry
	 * must not find half-configured global state.
	 */
err_pci:
	pci_env_fini();
err_threads:
	threads_fini();
	return rc;
}

void spdk_env_fini(void)
{
	/* Workers first: threads_fini() joins them, so they cannot touch the
	 * memory map or the PCI state after those are released.
	 */
	threads_fini();
	mem_map_fini();
	pci_env_fini();

	g_initialized = false;
}
