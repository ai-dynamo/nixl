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

#include "doca_memos_mocks.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace {

template<typename TypedTask>
doca_error_t
alloc_typed_task(union doca_data user_data, doca_task::Type type, TypedTask **out) {
    auto &mock = DocaMockControl::instance();

    if (mock.task_alloc_fail_after_n >= 0) {
        if (mock.task_alloc_call_count >= mock.task_alloc_fail_after_n) {
            mock.task_alloc_call_count++;
            *out = nullptr;
            return mock.task_alloc_fail_error;
        }
        mock.task_alloc_call_count++;
    }

    if (mock.kv_task_alloc_result != DOCA_SUCCESS) {
        *out = nullptr;
        return mock.kv_task_alloc_result;
    }

    TypedTask *task = new TypedTask{};
    task->type = type;
    task->user_data = user_data;
    *out = task;
    return DOCA_SUCCESS;
}

} // namespace

extern "C" {

/*********************************************************************************************************************
 * doca_nvme_kernel_kvdev API
 *********************************************************************************************************************/

doca_error_t
doca_nvme_kernel_kvdev_cap_get_max_path_len(uint32_t *max_path_len) {
    if (!max_path_len) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *max_path_len = 4096;
    return DOCA_SUCCESS;
}

doca_error_t
doca_nvme_kernel_kvdev_create(struct doca_nvme_kernel_kvdev **kkvdev) {
    if (!kkvdev) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();
    if (mock.nvme_kvdev_create_result == DOCA_SUCCESS) {
        *kkvdev = new doca_nvme_kernel_kvdev();
    } else {
        *kkvdev = nullptr;
    }
    return mock.nvme_kvdev_create_result;
}

doca_error_t
doca_nvme_kernel_kvdev_destroy(struct doca_nvme_kernel_kvdev *kkvdev) {
    delete kkvdev;
    return DOCA_SUCCESS;
}

struct doca_kvdev *
doca_nvme_kernel_kvdev_as_kvdev(struct doca_nvme_kernel_kvdev *kkvdev) {
    return reinterpret_cast<struct doca_kvdev *>(kkvdev);
}

doca_error_t
doca_nvme_kernel_kvdev_set_path(struct doca_nvme_kernel_kvdev *kkvdev, const char *path) {
    if (!kkvdev || !path) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    return DOCA_SUCCESS;
}

doca_error_t
doca_nvme_kernel_kvdev_get_path(const struct doca_nvme_kernel_kvdev *kkvdev,
                                char *path,
                                uint32_t path_len) {
    (void)kkvdev;
    (void)path;
    (void)path_len;
    return DOCA_ERROR_NOT_FOUND;
}

/*********************************************************************************************************************
 * doca_kvdev API
 *********************************************************************************************************************/

doca_error_t
doca_kvdev_set_nguid(struct doca_kvdev *kvdev, const uint8_t *nguid) {
    if (!kvdev || !nguid) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_get_nguid(const struct doca_kvdev *kvdev, uint8_t *nguid, uint16_t nguid_len) {
    if (!kvdev || !nguid || nguid_len < DOCA_KVDEV_NGUID_LEN) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    std::memset(nguid, 0, DOCA_KVDEV_NGUID_LEN);
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_start(struct doca_kvdev *kvdev) {
    auto lk = DocaMockControl::lock();
    (void)kvdev;
    return DocaMockControl::instance().kvdev_start_result;
}

doca_error_t
doca_kvdev_stop(struct doca_kvdev *kvdev) {
    auto lk = DocaMockControl::lock();
    (void)kvdev;
    return DocaMockControl::instance().kvdev_stop_result;
}

doca_error_t
doca_kvdev_is_started(const struct doca_kvdev *kvdev, uint8_t *started) {
    if (!kvdev || !started) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    // The plugin only calls is_started during teardown and only acts on the
    // "started" branch, so reporting started=1 keeps the teardown path
    // exercised in tests.
    *started = 1;
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_get_size(const struct doca_kvdev *kvdev, uint64_t *size) {
    if (!kvdev || !size) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *size = static_cast<uint64_t>(1) << 30; // 1 GiB
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_get_max_key_len(const struct doca_kvdev *kvdev, uint16_t *max_key_len) {
    if (!kvdev || !max_key_len) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *max_key_len = 16;
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_get_max_value_len(const struct doca_kvdev *kvdev, uint32_t *max_value_len) {
    if (!kvdev || !max_value_len) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *max_value_len = 1u << 20; // 1 MiB
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_get_max_tasks(const struct doca_kvdev *kvdev, uint32_t *max_tasks) {
    if (!kvdev || !max_tasks) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *max_tasks = 65536;
    return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * doca_nvme_kernel_kvdev_io API
 *********************************************************************************************************************/

doca_error_t
doca_nvme_kernel_kvdev_io_create(struct doca_nvme_kernel_kvdev *kkvdev,
                                 struct doca_nvme_kernel_kvdev_io **kio) {
    if (!kkvdev || !kio) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();
    if (mock.kv_io_create_result == DOCA_SUCCESS) {
        *kio = new doca_nvme_kernel_kvdev_io();
    } else {
        *kio = nullptr;
    }
    return mock.kv_io_create_result;
}

doca_error_t
doca_nvme_kernel_kvdev_io_destroy(struct doca_nvme_kernel_kvdev_io *kio) {
    auto lk = DocaMockControl::lock();
    delete kio;
    return DocaMockControl::instance().kv_io_destroy_result;
}

struct doca_kvdev_io *
doca_nvme_kernel_kvdev_io_as_kvdev_io(struct doca_nvme_kernel_kvdev_io *kio) {
    return reinterpret_cast<struct doca_kvdev_io *>(kio);
}

/*********************************************************************************************************************
 * doca_kvdev_io API
 *********************************************************************************************************************/

struct doca_ctx *
doca_kvdev_io_as_ctx(struct doca_kvdev_io *io) {
    return reinterpret_cast<struct doca_ctx *>(io);
}

doca_error_t
doca_kvdev_io_get_num_tasks(const struct doca_kvdev_io *io, uint32_t *num_tasks) {
    if (!io || !num_tasks) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *num_tasks = 8192;
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_io_set_num_tasks(struct doca_kvdev_io *io, uint32_t num_tasks) {
    if (!io || num_tasks == 0) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_io_set_task_completion_cb(struct doca_kvdev_io *io,
                                     doca_kvdev_io_task_completion_cb_t cb) {
    if (!io || !cb) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    DocaMockControl::instance().task_completion_cb = cb;
    return DOCA_SUCCESS;
}

doca_error_t
doca_kvdev_io_set_task_error_cb(struct doca_kvdev_io *io, doca_kvdev_io_task_completion_cb_t cb) {
    if (!io || !cb) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    DocaMockControl::instance().task_error_cb = cb;
    return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * Store task API
 *********************************************************************************************************************/

doca_error_t
doca_kvdev_io_task_store_alloc_init(struct doca_kvdev_io *io,
                                    union doca_data task_user_data,
                                    struct doca_kvdev_io_task_store **task) {
    if (!io || !task) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    return alloc_typed_task(task_user_data, doca_task::STORE, task);
}

void
doca_kvdev_io_task_store_set_key_value_conf(struct doca_kvdev_io_task_store *task,
                                            const void *key,
                                            uint16_t key_len,
                                            struct iovec *value_iovecs,
                                            uint32_t iovcnt,
                                            uint32_t value_len) {
    if (!task) {
        return;
    }
    task->key.assign(static_cast<const char *>(key), key_len);
    task->buffer = (iovcnt > 0) ? value_iovecs[0].iov_base : nullptr;
    task->buffer_size = (iovcnt > 0) ? value_iovecs[0].iov_len : 0;
    task->value_len = value_len;
}

void
doca_kvdev_io_task_store_set_do_not_overwrite(struct doca_kvdev_io_task_store *task,
                                              uint8_t do_not_overwrite) {
    if (task) {
        task->do_not_overwrite = do_not_overwrite;
    }
}

uint8_t
doca_kvdev_io_task_store_get_do_not_overwrite(const struct doca_kvdev_io_task_store *task) {
    return task ? task->do_not_overwrite : 0;
}

void
doca_kvdev_io_task_store_set_must_exist(struct doca_kvdev_io_task_store *task, uint8_t must_exist) {
    if (task) {
        task->must_exist = must_exist;
    }
}

uint8_t
doca_kvdev_io_task_store_get_must_exist(const struct doca_kvdev_io_task_store *task) {
    return task ? task->must_exist : 0;
}

struct doca_task *
doca_kvdev_io_task_store_as_task(struct doca_kvdev_io_task_store *task) {
    return static_cast<struct doca_task *>(task);
}

struct doca_kvdev_io_task_store *
doca_kvdev_io_task_store_from_task(struct doca_task *task) {
    return static_cast<struct doca_kvdev_io_task_store *>(task);
}

const void *
doca_kvdev_io_task_store_get_key(const struct doca_kvdev_io_task_store *task) {
    return task ? task->key.data() : nullptr;
}

uint16_t
doca_kvdev_io_task_store_get_key_len(const struct doca_kvdev_io_task_store *task) {
    return task ? static_cast<uint16_t>(task->key.size()) : 0;
}

const struct iovec *
doca_kvdev_io_task_store_get_value_iovecs(const struct doca_kvdev_io_task_store *task) {
    (void)task;
    return nullptr;
}

uint32_t
doca_kvdev_io_task_store_get_value_iovcnt(const struct doca_kvdev_io_task_store *task) {
    return task && task->buffer ? 1 : 0;
}

uint32_t
doca_kvdev_io_task_store_get_value_len(const struct doca_kvdev_io_task_store *task) {
    return task ? task->value_len : 0;
}

/*********************************************************************************************************************
 * Retrieve task API
 *********************************************************************************************************************/

doca_error_t
doca_kvdev_io_task_retrieve_alloc_init(struct doca_kvdev_io *io,
                                       union doca_data task_user_data,
                                       struct doca_kvdev_io_task_retrieve **task) {
    if (!io || !task) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    return alloc_typed_task(task_user_data, doca_task::RETRIEVE, task);
}

void
doca_kvdev_io_task_retrieve_set_key_value_conf(struct doca_kvdev_io_task_retrieve *task,
                                               const void *key,
                                               uint16_t key_len,
                                               struct iovec *value_iovecs,
                                               uint32_t iovcnt,
                                               uint32_t value_len) {
    if (!task) {
        return;
    }
    task->key.assign(static_cast<const char *>(key), key_len);
    task->buffer = (iovcnt > 0) ? value_iovecs[0].iov_base : nullptr;
    task->buffer_size = (iovcnt > 0) ? value_iovecs[0].iov_len : 0;
    task->value_len = value_len;
}

struct doca_task *
doca_kvdev_io_task_retrieve_as_task(struct doca_kvdev_io_task_retrieve *task) {
    return static_cast<struct doca_task *>(task);
}

struct doca_kvdev_io_task_retrieve *
doca_kvdev_io_task_retrieve_from_task(struct doca_task *task) {
    return static_cast<struct doca_kvdev_io_task_retrieve *>(task);
}

const void *
doca_kvdev_io_task_retrieve_get_key(const struct doca_kvdev_io_task_retrieve *task) {
    return task ? task->key.data() : nullptr;
}

uint16_t
doca_kvdev_io_task_retrieve_get_key_len(const struct doca_kvdev_io_task_retrieve *task) {
    return task ? static_cast<uint16_t>(task->key.size()) : 0;
}

const struct iovec *
doca_kvdev_io_task_retrieve_get_value_iovecs(const struct doca_kvdev_io_task_retrieve *task) {
    (void)task;
    return nullptr;
}

uint32_t
doca_kvdev_io_task_retrieve_get_value_iovcnt(const struct doca_kvdev_io_task_retrieve *task) {
    return task && task->buffer ? 1 : 0;
}

uint32_t
doca_kvdev_io_task_retrieve_get_value_len(const struct doca_kvdev_io_task_retrieve *task) {
    return task ? task->value_len : 0;
}

uint32_t
doca_kvdev_io_task_retrieve_get_result_value_len(const struct doca_kvdev_io_task_retrieve *task) {
    return task ? task->result_value_len : 0;
}

/*********************************************************************************************************************
 * Exist task API
 *********************************************************************************************************************/

doca_error_t
doca_kvdev_io_task_exist_alloc_init(struct doca_kvdev_io *io,
                                    union doca_data task_user_data,
                                    struct doca_kvdev_io_task_exist **task) {
    if (!io || !task) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    return alloc_typed_task(task_user_data, doca_task::EXIST, task);
}

void
doca_kvdev_io_task_exist_set_key_conf(struct doca_kvdev_io_task_exist *task,
                                      const void *key,
                                      uint16_t key_len) {
    if (!task) {
        return;
    }
    task->key.assign(static_cast<const char *>(key), key_len);
}

struct doca_task *
doca_kvdev_io_task_exist_as_task(struct doca_kvdev_io_task_exist *task) {
    return static_cast<struct doca_task *>(task);
}

struct doca_kvdev_io_task_exist *
doca_kvdev_io_task_exist_from_task(struct doca_task *task) {
    return static_cast<struct doca_kvdev_io_task_exist *>(task);
}

const void *
doca_kvdev_io_task_exist_get_key(const struct doca_kvdev_io_task_exist *task) {
    return task ? task->key.data() : nullptr;
}

uint16_t
doca_kvdev_io_task_exist_get_key_len(const struct doca_kvdev_io_task_exist *task) {
    return task ? static_cast<uint16_t>(task->key.size()) : 0;
}

/*********************************************************************************************************************
 * doca_pe API
 *********************************************************************************************************************/

doca_error_t
doca_pe_create(struct doca_pe **pe) {
    if (!pe) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();
    if (mock.pe_create_result == DOCA_SUCCESS) {
        *pe = new doca_pe();
    } else {
        *pe = nullptr;
    }
    return mock.pe_create_result;
}

doca_error_t
doca_pe_destroy(struct doca_pe *pe) {
    auto lk = DocaMockControl::lock();
    delete pe;
    return DocaMockControl::instance().pe_destroy_result;
}

doca_error_t
doca_pe_connect_ctx(struct doca_pe *pe, struct doca_ctx *ctx) {
    (void)pe;
    (void)ctx;
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().pe_connect_ctx_result;
}

int
doca_pe_progress(struct doca_pe *pe) {
    (void)pe;
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();

    if (!mock.auto_complete_tasks || mock.submitted_tasks.empty()) {
        return mock.pe_progress_return;
    }

    // Drain in submission order (FIFO). Capped per call to mimic the real PE
    // and to keep a runaway callback (one that re-submits forever) from
    // hanging the test process.
    constexpr size_t max_completions_per_progress = 64;
    size_t drained = 0;
    while (!mock.submitted_tasks.empty() && drained < max_completions_per_progress) {
        struct doca_task *task = mock.submitted_tasks.front();
        mock.submitted_tasks.erase(mock.submitted_tasks.begin());
        mock.submitted_task_user_data.erase(mock.submitted_task_user_data.begin());

        union doca_data ctx_user_data{};
        ctx_user_data.ptr = nullptr;

        switch (task->type) {
        case doca_task::STORE:
            if (mock.force_task_error) {
                task->status = mock.forced_task_error_code;
                if (mock.task_error_cb) {
                    mock.task_error_cb(task, task->user_data, ctx_user_data);
                }
            } else if (mock.task_completion_cb) {
                std::vector<uint8_t> value(task->buffer_size);
                std::memcpy(value.data(), task->buffer, task->buffer_size);
                mock.kv_store[task->key] = std::move(value);
                mock.task_completion_cb(task, task->user_data, ctx_user_data);
            }
            break;
        case doca_task::RETRIEVE:
            if (mock.force_task_error) {
                task->status = mock.forced_task_error_code;
                if (mock.task_error_cb) {
                    mock.task_error_cb(task, task->user_data, ctx_user_data);
                }
            } else if (mock.task_completion_cb) {
                auto it = mock.kv_store.find(task->key);
                if (it != mock.kv_store.end()) {
                    size_t copy_size = std::min(task->buffer_size, it->second.size());
                    if (copy_size > 0) {
                        std::memcpy(task->buffer, it->second.data(), copy_size);
                    }
                    task->result_value_len = static_cast<uint32_t>(it->second.size());
                    mock.task_completion_cb(task, task->user_data, ctx_user_data);
                } else if (mock.task_error_cb) {
                    task->status = DOCA_ERROR_NOT_FOUND;
                    mock.task_error_cb(task, task->user_data, ctx_user_data);
                }
            }
            break;
        case doca_task::EXIST:
            // Real DOCA reports EXIST results through the error callback:
            // ALREADY_EXIST when found, NOT_FOUND when absent.
            if (mock.force_task_error) {
                task->status = mock.forced_task_error_code;
                if (mock.task_error_cb) {
                    mock.task_error_cb(task, task->user_data, ctx_user_data);
                }
            } else if (mock.task_error_cb) {
                auto it = mock.kv_store.find(task->key);
                task->status =
                    (it != mock.kv_store.end()) ? DOCA_ERROR_ALREADY_EXIST : DOCA_ERROR_NOT_FOUND;
                mock.task_error_cb(task, task->user_data, ctx_user_data);
            }
            break;
        }
        drained++;
    }

    return drained > 0 ? 1 : mock.pe_progress_return;
}

doca_error_t
doca_pe_get_notification_handle(struct doca_pe *pe, doca_notification_handle_t *handle) {
    (void)pe;
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();
    if (mock.pe_get_notification_handle_result == DOCA_SUCCESS) {
        *handle = mock.notification_handle;
    }
    return mock.pe_get_notification_handle_result;
}

doca_error_t
doca_pe_request_notification(struct doca_pe *pe) {
    (void)pe;
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().pe_request_notification_result;
}

void
doca_pe_clear_notification(struct doca_pe *pe, doca_notification_handle_t handle) {
    (void)pe;
    (void)handle;
}

doca_error_t
doca_task_submit(struct doca_task *task) {
    auto lk = DocaMockControl::lock();
    auto &mock = DocaMockControl::instance();

    if (mock.task_submit_fail_after_n >= 0) {
        if (mock.task_submit_call_count >= mock.task_submit_fail_after_n) {
            mock.task_submit_call_count++;
            return mock.task_submit_fail_error;
        }
        mock.task_submit_call_count++;
    }

    if (mock.task_submit_result == DOCA_SUCCESS) {
        mock.submitted_tasks.push_back(task);
        mock.submitted_task_user_data.push_back(task->user_data);
    }
    return mock.task_submit_result;
}

void
doca_task_free(struct doca_task *task) {
    auto lk = DocaMockControl::lock();
    delete task;
}

doca_error_t
doca_task_get_status(const struct doca_task *task) {
    auto lk = DocaMockControl::lock();
    return task ? task->status : DOCA_SUCCESS;
}

doca_error_t
doca_task_get_user_data(const struct doca_task *task, union doca_data *user_data) {
    if (!task || !user_data) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *user_data = task->user_data;
    return DOCA_SUCCESS;
}

doca_error_t
doca_task_set_user_data(struct doca_task *task, union doca_data user_data) {
    if (!task) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    task->user_data = user_data;
    return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * doca_ctx API
 *********************************************************************************************************************/

doca_error_t
doca_ctx_start(struct doca_ctx *ctx) {
    (void)ctx;
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().ctx_start_result;
}

doca_error_t
doca_ctx_stop(struct doca_ctx *ctx) {
    (void)ctx;
    auto lk = DocaMockControl::lock();
    return DocaMockControl::instance().ctx_stop_result;
}

doca_error_t
doca_ctx_get_num_inflight_tasks(const struct doca_ctx *ctx, size_t *num_inflight_tasks) {
    if (!ctx || !num_inflight_tasks) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    auto lk = DocaMockControl::lock();
    *num_inflight_tasks = DocaMockControl::instance().submitted_tasks.size();
    return DOCA_SUCCESS;
}

doca_error_t
doca_ctx_set_user_data(struct doca_ctx *ctx, union doca_data user_data) {
    (void)ctx;
    (void)user_data;
    return DOCA_SUCCESS;
}

doca_error_t
doca_ctx_get_user_data(const struct doca_ctx *ctx, union doca_data *user_data) {
    if (!ctx || !user_data) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    user_data->ptr = nullptr;
    return DOCA_SUCCESS;
}

doca_error_t
doca_ctx_get_state(const struct doca_ctx *ctx, enum doca_ctx_states *state) {
    if (!ctx || !state) {
        return DOCA_ERROR_INVALID_VALUE;
    }
    *state = DOCA_CTX_STATE_RUNNING;
    return DOCA_SUCCESS;
}

/*********************************************************************************************************************
 * doca_error API
 *********************************************************************************************************************/

const char *
doca_error_get_descr(doca_error_t error) {
    switch (error) {
    case DOCA_SUCCESS:
        return "Success";
    case DOCA_ERROR_UNKNOWN:
        return "Unknown error";
    case DOCA_ERROR_INVALID_VALUE:
        return "Invalid value";
    case DOCA_ERROR_NO_MEMORY:
        return "Out of memory";
    case DOCA_ERROR_NOT_FOUND:
        return "Not found";
    case DOCA_ERROR_IO_FAILED:
        return "I/O failed";
    case DOCA_ERROR_BAD_STATE:
        return "Bad state";
    case DOCA_ERROR_OPERATING_SYSTEM:
        return "Operating system error";
    case DOCA_ERROR_ALREADY_EXIST:
        return "Already exists";
    case DOCA_ERROR_FULL:
        return "Full";
    case DOCA_ERROR_IN_PROGRESS:
        return "In progress";
    case DOCA_ERROR_TOO_BIG:
        return "Too big";
    case DOCA_ERROR_DRIVER:
        return "Driver error";
    default:
        return "Unrecognized error";
    }
}

const char *
doca_error_get_name(doca_error_t error) {
    return doca_error_get_descr(error);
}

} // extern "C"
