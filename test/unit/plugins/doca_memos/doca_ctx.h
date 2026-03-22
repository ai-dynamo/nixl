/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * DOCA KV Mock - Context API
 * Mock implementation of DOCA context
 */

#ifndef NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_CTX_H
#define NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_CTX_H

#include <stddef.h>
#include <stdint.h>

#include <doca_compat.h>
#include <doca_error.h>
#include <doca_types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct doca_ctx;
struct doca_task;

/**
 * @brief Context states
 */
enum doca_ctx_states {
    DOCA_CTX_STATE_IDLE = 0,
    DOCA_CTX_STATE_STARTING = 1,
    DOCA_CTX_STATE_RUNNING = 2,
    DOCA_CTX_STATE_STOPPING = 3,
};

/**
 * @brief Finalizes all configurations, and starts the DOCA CTX.
 */
DOCA_STABLE
doca_error_t
doca_ctx_start(struct doca_ctx *ctx);

/**
 * @brief Stops the context allowing reconfiguration.
 */
DOCA_STABLE
doca_error_t
doca_ctx_stop(struct doca_ctx *ctx);

/**
 * @brief Get number of in flight tasks in a doca context
 */
DOCA_STABLE
doca_error_t
doca_ctx_get_num_inflight_tasks(const struct doca_ctx *ctx, size_t *num_inflight_tasks);

/**
 * @brief set user data to context
 */
DOCA_STABLE
doca_error_t
doca_ctx_set_user_data(struct doca_ctx *ctx, union doca_data user_data);

/**
 * @brief get user data from context
 */
DOCA_STABLE
doca_error_t
doca_ctx_get_user_data(const struct doca_ctx *ctx, union doca_data *user_data);

/**
 * @brief Get context state
 */
DOCA_STABLE
doca_error_t
doca_ctx_get_state(const struct doca_ctx *ctx, enum doca_ctx_states *state);

#ifdef __cplusplus
}
#endif

#endif /* NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_CTX_H */
