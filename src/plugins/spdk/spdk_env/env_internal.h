/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_INTERNAL_H
#define NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_INTERNAL_H

#include "spdk/stdinc.h"
#include "spdk/env.h"

/* x86-64 and ARM userspace virtual addresses use only the low 48 bits [0..47],
 * which is enough to cover 256 TB. env_mem_map.h builds its index macros on
 * these, so they live here and are defined once.
 */
#define SHIFT_256TB 48
#define MASK_256TB ((1ULL << SHIFT_256TB) - 1)

#define SHIFT_1GB 30
#define VALUE_1GB (1ULL << SHIFT_1GB)
#define MASK_1GB ((1ULL << SHIFT_1GB) - 1)

int threads_init(const struct spdk_env_opts *opts);
void threads_fini(void);

int pci_env_init(void);
void pci_env_reinit(void);
void pci_env_fini(void);
int mem_map_init(void);
void mem_map_fini(void);

void mem_enforce_numa(void);

#endif /* NIXL_SRC_PLUGINS_SPDK_SPDK_ENV_ENV_INTERNAL_H */
