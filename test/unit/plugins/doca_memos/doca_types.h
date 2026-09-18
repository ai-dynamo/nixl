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
 * DOCA KV Mock - Common Types
 * Mock implementation of DOCA common types
 */

#ifndef NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_TYPES_H
#define NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_TYPES_H

#include <stdint.h>

#ifdef __linux__
#include <linux/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __linux__
/** 'fd' for blocking with epoll/select/poll, event type will be "read ready" */
typedef int doca_notification_handle_t;
#define doca_event_invalid_handle -1
#else
typedef void *doca_notification_handle_t;
#define doca_event_invalid_handle NULL
#endif

/**
 * @brief Convenience type for representing opaque data
 */
union doca_data {
    void *ptr;
    uint64_t u64;
};

/**
 * @brief Struct to represent a gather list
 */
struct doca_gather_list {
    void *addr;
    uint64_t len;
    struct doca_gather_list *next;
};

#ifdef __cplusplus
}
#endif

#endif /* NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_TYPES_H */
