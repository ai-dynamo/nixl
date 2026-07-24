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

#ifndef NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_MOCKS_H
#define NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_MOCKS_H

// This header is only consumed from C++ translation units. Including the C++
// standard library headers up-front keeps the mock state definitions below
// well-formed regardless of which DOCA header pulls them in.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <sys/uio.h>
#include <vector>

#include <doca_compat.h>
#include <doca_ctx.h>
#include <doca_error.h>
#include <doca_kvdev.h>
#include <doca_kvdev_io.h>
#include <doca_nvme_kernel_kvdev.h>
#include <doca_nvme_kernel_kvdev_io.h>
#include <doca_pe.h>
#include <doca_types.h>

// Concrete definitions for opaque mock types. These structs are forward
// declared in the public DOCA headers above; defining them here gives the
// mocks something to allocate, while still keeping them opaque to the plugin.
struct doca_kvdev {};

struct doca_kvdev_io {};

struct doca_pe {};

struct doca_ctx {};

struct doca_nvme_kernel_kvdev {};

struct doca_nvme_kernel_kvdev_io {};

// All typed KV tasks are unified into a single mock task carrying the per-op
// state the implementation needs to observe (key, value buffer, user data,
// completion status, store/retrieve options). The typed structs in the public
// headers are forward declarations only; the mocks derive them from doca_task
// so the *_as_task / *_from_task helpers are simple pointer casts.
struct doca_task {
    enum Type { STORE, RETRIEVE, EXIST } type;

    std::string key;
    void *buffer = nullptr;
    size_t buffer_size = 0;
    uint32_t value_len = 0;
    union doca_data user_data{};
    doca_error_t status = DOCA_SUCCESS;
    uint8_t do_not_overwrite = 0;
    uint8_t must_exist = 0;
    uint32_t result_value_len = 0;
};

struct doca_kvdev_io_task_store : doca_task {};

struct doca_kvdev_io_task_retrieve : doca_task {};

struct doca_kvdev_io_task_exist : doca_task {};

// Mock control singleton. Tests set fields here to control how the mock
// implementation responds to plugin API calls.
class DocaMockControl {
public:
    static DocaMockControl &
    instance() {
        static DocaMockControl inst;
        return inst;
    }

    // Serialises every singleton access. Recursive so callbacks fired from
    // doca_pe_progress() can re-enter mocks (e.g. doca_task_free) without
    // self-deadlock. Tests should hold this lock around any field they read
    // or mutate when a threaded progress engine is alive.
    static std::unique_lock<std::recursive_mutex>
    lock() {
        return std::unique_lock<std::recursive_mutex>(instance().mutex_);
    }

    // Configuration for mock behavior - all DOCA API return values
    doca_error_t nvme_kvdev_create_result = DOCA_SUCCESS;
    doca_error_t kvdev_start_result = DOCA_SUCCESS;
    doca_error_t kvdev_stop_result = DOCA_SUCCESS;
    doca_error_t pe_create_result = DOCA_SUCCESS;
    doca_error_t pe_destroy_result = DOCA_SUCCESS;
    doca_error_t kv_io_create_result = DOCA_SUCCESS;
    doca_error_t kv_io_destroy_result = DOCA_SUCCESS;
    doca_error_t ctx_start_result = DOCA_SUCCESS;
    doca_error_t ctx_stop_result = DOCA_SUCCESS;
    doca_error_t pe_connect_ctx_result = DOCA_SUCCESS;
    doca_error_t kv_task_alloc_result = DOCA_SUCCESS;
    doca_error_t task_submit_result = DOCA_SUCCESS;
    doca_error_t pe_get_notification_handle_result = DOCA_SUCCESS;
    doca_error_t pe_request_notification_result = DOCA_SUCCESS;
    doca_error_t get_max_value_len_result = DOCA_SUCCESS;
    uint32_t max_value_len = 1u << 20; // 1 MiB default

    // Advanced error injection - fail after N successful calls.
    // task_alloc_fail_after_n: -1 disables; 0 fails first call; N fails the
    // (N+1)th call onwards. Applies uniformly to store / retrieve / exist.
    int task_alloc_fail_after_n = -1;
    int task_alloc_call_count = 0;
    doca_error_t task_alloc_fail_error = DOCA_ERROR_FULL;
    int task_submit_fail_after_n = -1;
    int task_submit_call_count = 0;
    doca_error_t task_submit_fail_error = DOCA_ERROR_FULL;

    // Force specific task to fail with specific error
    bool force_task_error = false;
    doca_error_t forced_task_error_code = DOCA_ERROR_NOT_FOUND;

    // Callbacks installed by doca_kvdev_io_set_task_{completion,error}_cb.
    // Signatures match the public typedef.
    using TaskCallback = std::function<void(struct doca_task *, union doca_data, union doca_data)>;
    TaskCallback task_completion_cb = nullptr;
    TaskCallback task_error_cb = nullptr;

    // Simulated KV storage
    std::map<std::string, std::vector<uint8_t>> kv_store;

    // Submitted tasks tracking
    std::vector<struct doca_task *> submitted_tasks;
    std::vector<union doca_data> submitted_task_user_data;

    // Progress tracking
    int pe_progress_return = 0;
    bool auto_complete_tasks = false;

    // Notification handle
    doca_notification_handle_t notification_handle = 100;

    // Reset all state. Acquires the mock lock so callers don't have to;
    // since the lock is recursive this is safe to call from a context that
    // already holds it.
    void
    reset() {
        std::unique_lock<std::recursive_mutex> guard(mutex_);
        nvme_kvdev_create_result = DOCA_SUCCESS;
        kvdev_start_result = DOCA_SUCCESS;
        kvdev_stop_result = DOCA_SUCCESS;
        pe_create_result = DOCA_SUCCESS;
        pe_destroy_result = DOCA_SUCCESS;
        kv_io_create_result = DOCA_SUCCESS;
        kv_io_destroy_result = DOCA_SUCCESS;
        ctx_start_result = DOCA_SUCCESS;
        ctx_stop_result = DOCA_SUCCESS;
        pe_connect_ctx_result = DOCA_SUCCESS;
        kv_task_alloc_result = DOCA_SUCCESS;
        task_submit_result = DOCA_SUCCESS;
        pe_get_notification_handle_result = DOCA_SUCCESS;
        pe_request_notification_result = DOCA_SUCCESS;
        get_max_value_len_result = DOCA_SUCCESS;
        max_value_len = 1u << 20;

        task_alloc_fail_after_n = -1;
        task_alloc_call_count = 0;
        task_alloc_fail_error = DOCA_ERROR_FULL;
        task_submit_fail_after_n = -1;
        task_submit_call_count = 0;
        task_submit_fail_error = DOCA_ERROR_FULL;
        force_task_error = false;
        forced_task_error_code = DOCA_ERROR_NOT_FOUND;

        task_completion_cb = nullptr;
        task_error_cb = nullptr;

        kv_store.clear();
        submitted_tasks.clear();
        submitted_task_user_data.clear();

        pe_progress_return = 0;
        auto_complete_tasks = false;
        notification_handle = 100;
    }

private:
    DocaMockControl() = default;

    std::recursive_mutex mutex_;
};

#endif // NIXL_TEST_UNIT_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_MOCKS_H
