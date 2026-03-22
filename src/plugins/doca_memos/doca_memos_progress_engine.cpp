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

#include "doca_memos_progress_engine.h"
#include "doca_memos_backend.h" // For nixlDocaMemosBackendReqH
#include "common/nixl_log.h"

#include <algorithm>
#include <thread>
#include <chrono>

// DOCA includes
#include <doca_pe.h>
#include <doca_kv_dev.h>
#include <doca_kv_io.h>
#include <doca_kv_nvme_kernel_dev.h>
#include <doca_kv_nvme_kernel_io.h>
#include <doca_ctx.h>
#include <doca_error.h>

namespace {

// doca_pe_progress() reaps at most one completion per call, so reaping once per
// checkXfer() costs a full poll-loop round trip per completed task. Drain a
// bounded burst instead; the bound keeps a saturating device from starving the
// caller (and, in the no-thread engine, from holding lock_ indefinitely).
constexpr int kProgressBurst = 64;

constexpr auto kDestructorDrainBudget = std::chrono::seconds(10);

} // anonymous namespace

void
nixlDocaMemosProgressEngine::handleSubmissionFailure(nixlDocaMemosBackendReqH *req_h,
                                                     nixl_status_t status) {
    if (req_h->overallStatus_ == NIXL_IN_PROG) {
        req_h->overallStatus_ = status;
    }

    // Tasks beyond this point were never submitted and will never generate
    // callbacks. Adjust totalTasks_ so the in-flight tasks plus this failure
    // can reach the completion threshold.
    req_h->totalTasks_ = req_h->submittedTasks_ + 1;

    if (++req_h->completedTasks_ >= req_h->totalTasks_) {
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
    }
}

void
nixlDocaMemosProgressEngine::dropUnsubmittedTasks(nixlDocaMemosBackendReqH *req_h) {
    req_h->totalTasks_ = req_h->submittedTasks_;
    if (req_h->completedTasks_ >= req_h->totalTasks_) {
        req_h->allTasksCompleted_.store(true, std::memory_order_release);
    }
}

void
nixlDocaMemosProgressEngine::drainInflightTasks() {
    if (!ctx_ || !pe_) {
        return;
    }

    // doca_pe_progress() returning 0 only means nothing completed on that call,
    // not that nothing is in flight.
    auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;
    size_t inflight = 0;
    while (doca_ctx_get_num_inflight_tasks(ctx_, &inflight) == DOCA_SUCCESS && inflight > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            NIXL_WARN << inflight << " DOCA task(s) still in flight after "
                      << kDestructorDrainBudget.count() << "s; leaking their requests";
            return;
        }
        doca_pe_progress(pe_);
    }
}

void
nixlDocaMemosProgressEngine::taskCompletionCallback(
    struct doca_task *task,
    union doca_data task_user_data,
    union doca_data ctx_user_data,
    const struct doca_kv_io_task_retrieve *retrieve_task) {
    (void)ctx_user_data;

    auto *task_ctx = static_cast<docaMemosTaskContext *>(task_user_data.ptr);
    if (task_ctx && task_ctx->reqH) {
        auto *req_h = task_ctx->reqH;
        size_t task_index = task_ctx->taskIndex;
        // Relaxed load: missing a recent cancel is harmless because
        // processCancellations reconciles totalTasks_ afterwards. Anything
        // that changes how cancellations are reaped must preserve that.
        bool is_cancelled = req_h->cancelled_.load(std::memory_order_relaxed);

        if (!is_cancelled) {
            if (req_h->isExistQuery_) {
                req_h->taskResult_.store(static_cast<int>(DOCA_SUCCESS), std::memory_order_release);
                NIXL_DEBUG << "EXIST query completed - key exists (task " << task_index << ")";
            } else if (task_ctx->isRetrieve) {
                uint32_t result_len = doca_kv_io_task_retrieve_get_result_value_len(retrieve_task);
                if (result_len != task_ctx->expectedValueLen) {
                    // The caller cannot tell which bytes of a partial read are valid.
                    NIXL_ERROR << "Task " << task_index << " retrieved " << result_len
                               << " bytes, expected " << task_ctx->expectedValueLen;
                    if (req_h->overallStatus_ == NIXL_IN_PROG) {
                        req_h->overallStatus_ = NIXL_ERR_BACKEND;
                    }
                } else {
                    NIXL_DEBUG << "Task " << task_index << " completed successfully (" << result_len
                               << " bytes)";
                }
            } else {
                NIXL_DEBUG << "Task " << task_index << " completed successfully";
            }
        }

        if (++req_h->completedTasks_ >= req_h->totalTasks_) {
            if (!is_cancelled && req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_SUCCESS;
            }
            req_h->allTasksCompleted_.store(true, std::memory_order_release);
        }
    }
    doca_task_free(task);
}

void
nixlDocaMemosProgressEngine::storeTaskCompletionCallback(struct doca_kv_io_task_store *task,
                                                         union doca_data task_user_data,
                                                         union doca_data ctx_user_data) {
    taskCompletionCallback(doca_kv_io_task_store_as_task(task), task_user_data, ctx_user_data);
}

void
nixlDocaMemosProgressEngine::storeTaskErrorCallback(struct doca_kv_io_task_store *task,
                                                    union doca_data task_user_data,
                                                    union doca_data ctx_user_data) {
    taskErrorCallback(doca_kv_io_task_store_as_task(task), task_user_data, ctx_user_data);
}

void
nixlDocaMemosProgressEngine::retrieveTaskCompletionCallback(struct doca_kv_io_task_retrieve *task,
                                                            union doca_data task_user_data,
                                                            union doca_data ctx_user_data) {
    taskCompletionCallback(
        doca_kv_io_task_retrieve_as_task(task), task_user_data, ctx_user_data, task);
}

void
nixlDocaMemosProgressEngine::retrieveTaskErrorCallback(struct doca_kv_io_task_retrieve *task,
                                                       union doca_data task_user_data,
                                                       union doca_data ctx_user_data) {
    taskErrorCallback(doca_kv_io_task_retrieve_as_task(task), task_user_data, ctx_user_data);
}

void
nixlDocaMemosProgressEngine::existTaskCompletionCallback(struct doca_kv_io_task_exist *task,
                                                         union doca_data task_user_data,
                                                         union doca_data ctx_user_data) {
    taskCompletionCallback(doca_kv_io_task_exist_as_task(task), task_user_data, ctx_user_data);
}

void
nixlDocaMemosProgressEngine::existTaskErrorCallback(struct doca_kv_io_task_exist *task,
                                                    union doca_data task_user_data,
                                                    union doca_data ctx_user_data) {
    taskErrorCallback(doca_kv_io_task_exist_as_task(task), task_user_data, ctx_user_data);
}

void
nixlDocaMemosProgressEngine::taskErrorCallback(struct doca_task *task,
                                               union doca_data task_user_data,
                                               union doca_data ctx_user_data) {
    (void)ctx_user_data;

    auto *task_ctx = static_cast<docaMemosTaskContext *>(task_user_data.ptr);
    if (task_ctx && task_ctx->reqH) {
        auto *req_h = task_ctx->reqH;
        size_t task_index = task_ctx->taskIndex;
        // See taskCompletionCallback for why relaxed is sufficient here.
        bool is_cancelled = req_h->cancelled_.load(std::memory_order_relaxed);

        if (!is_cancelled) {
            doca_error_t task_err = doca_task_get_status(task);

            if (req_h->isExistQuery_) {
                // The DOCA EXIST task always completes via the error callback:
                // ALREADY_EXIST = key found, NOT_FOUND = key absent. Any other
                // status is a real failure.
                if (task_err == DOCA_ERROR_ALREADY_EXIST) {
                    req_h->taskResult_.store(static_cast<int>(DOCA_SUCCESS),
                                             std::memory_order_release);
                    NIXL_DEBUG << "EXIST query completed - key exists (task " << task_index << ")";
                } else if (task_err == DOCA_ERROR_NOT_FOUND) {
                    req_h->taskResult_.store(static_cast<int>(DOCA_ERROR_NOT_FOUND),
                                             std::memory_order_release);
                    NIXL_DEBUG << "EXIST query completed - key does not exist (task " << task_index
                               << ")";
                } else {
                    NIXL_ERROR << "EXIST task " << task_index
                               << " failed: " << doca_error_get_descr(task_err) << " ("
                               << static_cast<int>(task_err) << ")";
                    if (req_h->overallStatus_ == NIXL_IN_PROG) {
                        req_h->overallStatus_ = NIXL_ERR_BACKEND;
                    }
                }
            } else {
                NIXL_ERROR << "Task " << task_index
                           << " failed (non-EXIST operation): " << doca_error_get_descr(task_err)
                           << " (" << static_cast<int>(task_err) << ")";

                if (req_h->overallStatus_ == NIXL_IN_PROG) {
                    req_h->overallStatus_ = NIXL_ERR_BACKEND;
                }
            }
        }

        if (++req_h->completedTasks_ >= req_h->totalTasks_) {
            if (!is_cancelled && req_h->overallStatus_ == NIXL_IN_PROG) {
                req_h->overallStatus_ = NIXL_SUCCESS;
            }
            req_h->allTasksCompleted_.store(true, std::memory_order_release);
        }
    }
    doca_task_free(task);
}

nixlDocaMemosProgressEngine::nixlDocaMemosProgressEngine(struct doca_kv_nvme_kernel_dev *nvme_kvdev,
                                                         uint32_t num_tasks,
                                                         uint32_t max_value_len)
    : maxValueLen_(max_value_len) {
    doca_error_t result;

    result = doca_pe_create(&pe_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA progress engine: " << doca_error_get_descr(result);
        initErr_ = true;
        return;
    }

    // The I/O context is independent of any device now: it is created bare, started,
    // and only then does a device bind to it.
    result = doca_kv_nvme_kernel_io_create(&kkvIo_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA NVMe kernel KV IO context: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    kvIo_ = doca_kv_nvme_kernel_io_as_kv_io(kkvIo_);
    if (!kvIo_) {
        NIXL_ERROR << "Failed to convert NVMe kernel KV IO context to generic KV IO context";
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_set_num_tasks(kvIo_, num_tasks);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set DOCA KV IO task count: " << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_store_register_completion_cb(kvIo_, storeTaskCompletionCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV store completion callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_store_register_error_cb(kvIo_, storeTaskErrorCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV store error callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_retrieve_register_completion_cb(kvIo_, retrieveTaskCompletionCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV retrieve completion callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_retrieve_register_error_cb(kvIo_, retrieveTaskErrorCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV retrieve error callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_exist_register_completion_cb(kvIo_, existTaskCompletionCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV exist completion callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kv_io_task_exist_register_error_cb(kvIo_, existTaskErrorCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to register DOCA KV exist error callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    ctx_ = doca_kv_io_as_ctx(kvIo_);
    if (!ctx_) {
        NIXL_ERROR << "Failed to convert KV IO context to DOCA context";
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_pe_connect_ctx(pe_, ctx_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to connect context to progress engine: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_ctx_start(ctx_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to start DOCA context: " << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    // Bind only once the context runs, and record the device so cleanup can unbind it.
    // This runs before the subclass starts its progress thread, which matters because
    // bind and unbind both have to happen on the thread that drives the context.
    result = doca_kv_io_bind_dev(kvIo_, doca_kv_nvme_kernel_dev_as_kv_dev(nvme_kvdev));
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to bind KV device to IO context: " << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }
    boundDev_ = doca_kv_nvme_kernel_dev_as_kv_dev(nvme_kvdev);

    NIXL_DEBUG << "Progress engine base initialization complete";
}

void
nixlDocaMemosProgressEngine::cleanupDocaResources() {
    doca_error_t result;

    // Every device must be unbound before the context will stop, and unbind fails
    // with IN_USE while any task for it is still outstanding, so callers must have
    // drained first.
    if (boundDev_ && kvIo_) {
        result = doca_kv_io_unbind_dev(kvIo_, boundDev_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to unbind KV device from IO context: "
                      << doca_error_get_descr(result);
        }
        boundDev_ = nullptr;
    }

    if (ctx_) {
        result = doca_ctx_stop(ctx_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to stop DOCA context: " << doca_error_get_descr(result);
        }
        ctx_ = nullptr;
    }

    if (kkvIo_) {
        result = doca_kv_nvme_kernel_io_destroy(kkvIo_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to destroy DOCA NVMe kernel KV IO context: "
                      << doca_error_get_descr(result);
        }
        kkvIo_ = nullptr;
        kvIo_ = nullptr;
    }

    if (pe_) {
        result = doca_pe_destroy(pe_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to destroy DOCA progress engine: " << doca_error_get_descr(result);
        }
        pe_ = nullptr;
    }
}

nixl_status_t
nixlDocaMemosProgressEngine::checkXfer(nixlDocaMemosBackendReqH *req_h) const {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (req_h->allTasksCompleted()) {
        // Completion callbacks skip status bookkeeping once a request is
        // cancelled, so a released request that drained reports the abort.
        if (req_h->cancelled_.load(std::memory_order_acquire) &&
            req_h->getOverallStatus() == NIXL_IN_PROG) {
            return NIXL_ERR_CANCELED;
        }
        return req_h->getOverallStatus();
    }

    return NIXL_IN_PROG;
}

bool
nixlDocaMemosProgressEngine::trySubmitRequest(nixlDocaMemosBackendReqH *req_h,
                                              const nixl_xfer_op_t &operation,
                                              const nixl_meta_dlist_t &local,
                                              const nixl_meta_dlist_t &remote,
                                              bool &tasks_queued) const {
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        NIXL_DEBUG << "Pending queue not empty - queuing new request without submission";
        req_h->nextDescriptorIndex_ = 0;
        return false;
    }

    const size_t desc_count = static_cast<size_t>(local.descCount());
    for (size_t i = req_h->nextDescriptorIndex_; i < desc_count; i++) {
        const docaMemosKey &object_key = req_h->objectKeys_[i];

        auto *task_ctx = &req_h->taskContexts_[i];
        task_ctx->reqH = req_h;
        task_ctx->taskIndex = i;
        task_ctx->isRetrieve = (operation == NIXL_READ);
        task_ctx->expectedValueLen = req_h->valueIovecs_[i].iov_len;
        union doca_data task_user_data = {.ptr = task_ctx};

        struct doca_task *doca_task = nullptr;
        doca_error_t result;

        const size_t iov_len = req_h->valueIovecs_[i].iov_len;
        if (iov_len > UINT32_MAX) {
            NIXL_ERROR << "Buffer length " << iov_len << " exceeds DOCA KV max (UINT32_MAX)";
            handleSubmissionFailure(req_h, NIXL_ERR_INVALID_PARAM);
            return true;
        }
        const uint32_t value_len = static_cast<uint32_t>(iov_len);
        if (maxValueLen_ > 0 && value_len > maxValueLen_) {
            NIXL_ERROR << "Buffer length " << value_len << " exceeds device max value length "
                       << maxValueLen_;
            handleSubmissionFailure(req_h, NIXL_ERR_INVALID_PARAM);
            return true;
        }

        if (operation == NIXL_WRITE) {
            struct doca_kv_io_task_store *store_task = nullptr;
            result =
                doca_kv_io_task_store_alloc_init(kvIo_, boundDev_, task_user_data, &store_task);
            if (result != DOCA_SUCCESS) {
                if (result == DOCA_ERROR_FULL || result == DOCA_ERROR_NO_MEMORY) {
                    req_h->nextDescriptorIndex_ = i;
                    NIXL_DEBUG << "Task pool exhausted at descriptor " << i
                               << ", queueing for retry";
                    return false;
                }
                NIXL_ERROR << "Failed to allocate DOCA KV store task: "
                           << doca_error_get_descr(result);
                handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
                return true;
            }
            doca_kv_io_task_store_set_key_value_conf(store_task,
                                                     object_key.key,
                                                     object_key.keyLen,
                                                     &req_h->valueIovecs_[i],
                                                     1,
                                                     value_len);
            doca_task = doca_kv_io_task_store_as_task(store_task);
        } else { // NIXL_READ
            struct doca_kv_io_task_retrieve *retrieve_task = nullptr;
            result = doca_kv_io_task_retrieve_alloc_init(
                kvIo_, boundDev_, task_user_data, &retrieve_task);
            if (result != DOCA_SUCCESS) {
                if (result == DOCA_ERROR_FULL || result == DOCA_ERROR_NO_MEMORY) {
                    req_h->nextDescriptorIndex_ = i;
                    NIXL_DEBUG << "Task pool exhausted at descriptor " << i
                               << ", queueing for retry";
                    return false;
                }
                NIXL_ERROR << "Failed to allocate DOCA KV retrieve task: "
                           << doca_error_get_descr(result);
                handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
                return true;
            }
            doca_kv_io_task_retrieve_set_key_value_conf(retrieve_task,
                                                        object_key.key,
                                                        object_key.keyLen,
                                                        &req_h->valueIovecs_[i],
                                                        1,
                                                        value_len);
            doca_task = doca_kv_io_task_retrieve_as_task(retrieve_task);
        }

        // Queue without flushing so all tasks accepted during this scheduling
        // pass can be published to hardware with one doorbell.
        result = doca_task_submit_ex(doca_task, DOCA_TASK_SUBMIT_FLAG_NONE);
        if (result != DOCA_SUCCESS) {
            if (result == DOCA_ERROR_FULL) {
                doca_task_free(doca_task);
                req_h->nextDescriptorIndex_ = i;
                NIXL_DEBUG << "Task queue full at descriptor " << i << ", queueing for retry";
                return false;
            }
            NIXL_ERROR << "Failed to submit task: " << doca_error_get_descr(result);
            doca_task_free(doca_task);
            handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
            return true;
        }
        tasks_queued = true;
        req_h->submittedTasks_++;
        req_h->nextDescriptorIndex_ = i + 1;
    }

    return true;
}

void
nixlDocaMemosProgressEngine::flushSubmittedTasks(bool tasks_queued) const {
    if (tasks_queued) {
        doca_ctx_flush_tasks(ctx_);
    }
}

nixl_status_t
nixlNoThreadProgressEngine::postXfer(nixlDocaMemosBackendReqH *req_h,
                                     const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote) const {
    NIXL_LOCK_GUARD(lock_);

    req_h->totalTasks_ = static_cast<size_t>(local.descCount());
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextDescriptorIndex_ = 0;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);
    req_h->cancelled_.store(false, std::memory_order_relaxed);
    req_h->overallStatus_ = NIXL_IN_PROG;

    NIXL_DEBUG << "Posting transfer with " << req_h->totalTasks_ << " tasks";

    bool tasks_queued = false;
    bool fully_submitted = trySubmitRequest(req_h, operation, local, remote, tasks_queued);
    flushSubmittedTasks(tasks_queued);

    if (!fully_submitted) {
        req_h->storedOperation_ = operation;
        req_h->storedLocal_ = std::make_unique<nixl_meta_dlist_t>(local);
        req_h->storedRemote_ = std::make_unique<nixl_meta_dlist_t>(remote);
        req_h->isPending_ = true;
        pendingRequests_.push_back(req_h);

        NIXL_DEBUG << "Request partially submitted (" << req_h->submittedTasks_ << "/"
                   << req_h->totalTasks_ << "), queued for retry";
    }

    // Only reachable when trySubmitRequest hit a synchronous failure and
    // called handleSubmissionFailure, which caps totalTasks_ to the in-flight
    // count plus one. In-flight callbacks cannot race here because lock_ is
    // held and doca_pe_progress runs only under the same lock.
    if (req_h->allTasksCompleted()) {
        return req_h->getOverallStatus();
    }

    NIXL_DEBUG << "Transfer posted (" << req_h->submittedTasks_ << " tasks submitted)";
    return NIXL_IN_PROG;
}

void
nixlNoThreadProgressEngine::tryResumePendingRequests() const {
    NIXL_LOCK_GUARD(lock_);
    bool tasks_queued = false;

    while (!pendingRequests_.empty()) {
        auto *req_h = pendingRequests_.front();

        bool completed;
        if (req_h->isExistQuery_) {
            completed = trySubmitExistTask(req_h, tasks_queued);
        } else {
            completed = trySubmitRequest(req_h,
                                         req_h->storedOperation_,
                                         *req_h->storedLocal_,
                                         *req_h->storedRemote_,
                                         tasks_queued);
        }

        if (!completed) {
            break;
        }

        pendingRequests_.pop_front();
        req_h->isPending_ = false;
    }

    flushSubmittedTasks(tasks_queued);
}

nixl_status_t
nixlNoThreadProgressEngine::releaseRequest(nixlDocaMemosBackendReqH *req_h) const {
    NIXL_LOCK_GUARD(lock_);

    if (req_h->isPending_) {
        auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
        if (it != pendingRequests_.end()) {
            pendingRequests_.erase(it);
        }
        req_h->isPending_ = false;
        dropUnsubmittedTasks(req_h);
    }

    if (req_h->allTasksCompleted()) {
        delete req_h;
        return NIXL_SUCCESS;
    }

    req_h->cancelled_.store(true, std::memory_order_release);
    return NIXL_ERR_REPOST_ACTIVE;
}

int
nixlNoThreadProgressEngine::progress() const {
    NIXL_LOCK_GUARD(lock_);
    if (!pe_) {
        return 0;
    }
    int ret = 0;
    while (ret < kProgressBurst && doca_pe_progress(pe_) != 0) {
        ret++;
    }

    return ret;
}

void
nixlDocaMemosProgressEngine::prepareQueryHandles(
    const nixl_reg_dlist_t &descs,
    std::vector<nixlDocaMemosBackendReqH> &query_handles,
    bool convert_key_to_128bit,
    size_t max_key_len) const {
    size_t count = static_cast<size_t>(descs.descCount());
    // reserve() is load-bearing: callers stash &query_handles[i] in DOCA's
    // task_user_data, so the vector must not reallocate during emplace_back.
    query_handles.reserve(count);

    for (size_t i = 0; i < count; i++) {
        query_handles.emplace_back(1);
        auto &req_h = query_handles.back();
        req_h.isExistQuery_ = true;
        req_h.taskContexts_.resize(1);
        req_h.objectKeys_.resize(1);

        const auto &desc = descs[i];
        if (!nixlDocaMemosEngine::resolveMemosKey(desc.devId,
                                                  desc.metaInfo,
                                                  req_h.objectKeys_[0],
                                                  convert_key_to_128bit,
                                                  max_key_len)) {
            NIXL_ERROR << "Failed to resolve key for descriptor " << i;
            handleSubmissionFailure(&req_h, NIXL_ERR_INVALID_PARAM);
        }
    }
}

void
nixlDocaMemosProgressEngine::waitForQueryCompletion(
    const std::vector<nixlDocaMemosBackendReqH> &query_handles,
    const std::function<void()> &poll) const {
    while (true) {
        if (poll) {
            poll();
        }

        bool all_completed = true;
        for (const auto &req_h : query_handles) {
            if (!req_h.allTasksCompleted()) {
                all_completed = false;
                break;
            }
        }
        if (all_completed) {
            break;
        }

        std::this_thread::yield();
    }
}

size_t
nixlDocaMemosProgressEngine::collectQueryResults(
    const std::vector<nixlDocaMemosBackendReqH> &query_handles,
    std::vector<nixl_query_resp_t> &resp) const {
    size_t successful_queries = 0;

    for (const auto &req_h : query_handles) {
        if (req_h.getOverallStatus() == NIXL_SUCCESS) {
            doca_error_t task_result =
                static_cast<doca_error_t>(req_h.taskResult_.load(std::memory_order_acquire));
            if (task_result == DOCA_SUCCESS) {
                resp.emplace_back(nixl_query_resp_t{nixl_b_params_t{}});
                successful_queries++;
                NIXL_DEBUG << "Key exists";
            } else if (task_result == DOCA_ERROR_NOT_FOUND) {
                resp.emplace_back(std::nullopt);
                successful_queries++;
                NIXL_DEBUG << "Key does not exist";
            } else {
                resp.emplace_back(std::nullopt);
                NIXL_DEBUG << "Unexpected task result: " << task_result;
            }
        } else {
            resp.emplace_back(std::nullopt);
            NIXL_DEBUG << "Query failed with error";
        }
    }

    return successful_queries;
}

nixlNoThreadProgressEngine::nixlNoThreadProgressEngine(struct doca_kv_nvme_kernel_dev *nvme_kvdev,
                                                       uint32_t num_tasks,
                                                       uint32_t max_value_len,
                                                       nixl_thread_sync_t sync_mode)
    : nixlDocaMemosProgressEngine(nvme_kvdev, num_tasks, max_value_len),
      lock_(sync_mode) {
    if (initErr_) {
        return;
    }

    NIXL_DEBUG << "Created no-thread progress engine";
}

nixlNoThreadProgressEngine::~nixlNoThreadProgressEngine() {
    drainInflightTasks();
    cleanupDocaResources();
}

nixl_status_t
nixlNoThreadProgressEngine::checkXfer(nixlDocaMemosBackendReqH *req_h) const {
    if (!req_h) {
        return NIXL_ERR_INVALID_PARAM;
    }

    progress();
    tryResumePendingRequests();
    return nixlDocaMemosProgressEngine::checkXfer(req_h);
}

nixl_status_t
nixlNoThreadProgressEngine::queryMem(const nixl_reg_dlist_t &descs,
                                     std::vector<nixl_query_resp_t> &resp,
                                     bool convert_key_to_128bit,
                                     size_t max_key_len) const {
    resp.reserve(descs.descCount());

    std::vector<nixlDocaMemosBackendReqH> query_handles;
    prepareQueryHandles(descs, query_handles, convert_key_to_128bit, max_key_len);

    // See nixlThreadedProgressEngine::queryMem for the lifetime contract:
    // &query_handles[i] is published into pendingRequests_ and into DOCA's
    // task_user_data, so query_handles must not reallocate (prepareQueryHandles
    // reserves) and waitForQueryCompletion must drain every callback before
    // we return.
    {
        NIXL_LOCK_GUARD(lock_);
        bool tasks_queued = false;
        for (auto &req_h : query_handles) {
            if (req_h.allTasksCompleted()) {
                continue;
            }
            if (!trySubmitExistTask(&req_h, tasks_queued)) {
                req_h.isPending_ = true;
                pendingRequests_.push_back(&req_h);
            }
        }
        flushSubmittedTasks(tasks_queued);
    }

    waitForQueryCompletion(query_handles, [this] {
        progress();
        tryResumePendingRequests();
    });

    size_t successful_queries = collectQueryResults(query_handles, resp);
    size_t total = query_handles.size();
    if (successful_queries < total) {
        NIXL_ERROR << (total - successful_queries) << " of " << total << " query task(s) failed";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlThreadedProgressEngine::postXfer(nixlDocaMemosBackendReqH *req_h,
                                     const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote) const {
    {
        const std::lock_guard<std::mutex> g(queueMutex_);
        // The progress thread still has to reconcile an abort from an earlier
        // releaseRequest(), and would apply it to this post.
        if (req_h->abortQueued_) {
            return NIXL_ERR_REPOST_ACTIVE;
        }

        req_h->totalTasks_ = static_cast<size_t>(local.descCount());
        req_h->submittedTasks_ = 0;
        req_h->completedTasks_ = 0;
        req_h->nextDescriptorIndex_ = 0;
        req_h->allTasksCompleted_.store(false, std::memory_order_release);
        req_h->cancelled_.store(false, std::memory_order_relaxed);
        req_h->overallStatus_ = NIXL_IN_PROG;

        NIXL_DEBUG << "Posting transfer with " << req_h->totalTasks_ << " tasks (queued)";

        producerVec_->push_back({req_h,
                                 operation,
                                 std::make_unique<nixl_meta_dlist_t>(local),
                                 std::make_unique<nixl_meta_dlist_t>(remote)});
        hasNewWork_.store(true, std::memory_order_release);
    }
    wakeup_.notify_one();

    return NIXL_IN_PROG;
}

nixl_status_t
nixlThreadedProgressEngine::releaseRequest(nixlDocaMemosBackendReqH *req_h) const {
    nixl_status_t status;
    {
        const std::lock_guard<std::mutex> g(queueMutex_);

        auto it = std::find_if(producerVec_->begin(),
                               producerVec_->end(),
                               [req_h](const pendingEntry &e) { return e.reqH == req_h; });
        if (it != producerVec_->end()) {
            producerVec_->erase(it);
            delete req_h;
            return NIXL_SUCCESS;
        }

        // pendingRequests_ belongs to the progress thread, and the thread can
        // still touch a request just after its last task completes, so even a
        // completed request is freed on the thread.
        if (req_h->abortQueued_) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        if (req_h->allTasksCompleted()) {
            cancelledRequests_.push_back({req_h, true});
            status = NIXL_SUCCESS;
        } else if (req_h->cancelled_.load(std::memory_order_acquire)) {
            return NIXL_ERR_REPOST_ACTIVE;
        } else {
            req_h->abortQueued_ = true;
            cancelledRequests_.push_back({req_h, false});
            status = NIXL_ERR_REPOST_ACTIVE;
        }
        hasNewWork_.store(true, std::memory_order_release);
    }
    wakeup_.notify_one();
    return status;
}

nixlThreadedProgressEngine::nixlThreadedProgressEngine(struct doca_kv_nvme_kernel_dev *nvme_kvdev,
                                                       uint32_t num_tasks,
                                                       uint32_t max_value_len,
                                                       std::chrono::microseconds thread_delay)
    : nixlDocaMemosProgressEngine(nvme_kvdev, num_tasks, max_value_len),
      threadDelay_(thread_delay) {
    if (initErr_) {
        return;
    }

    NIXL_INFO << "Starting threaded progress engine with delay " << threadDelay_.count() << "us";

    // std::thread's constructor throws std::system_error on pthread_create
    // failure; nothing else to check.
    progressThread_ = std::thread(&nixlThreadedProgressEngine::progressThreadFunc, this);
}

nixlThreadedProgressEngine::~nixlThreadedProgressEngine() {
    NIXL_INFO << "Stopping threaded progress engine";

    threadStop_.store(true, std::memory_order_release);
    wakeup_.notify_all();

    if (progressThread_.joinable()) {
        progressThread_.join();
    }

    drainInflightTasks();

    // Release entries have completed and were handed over by the caller. Abort
    // entries are still owned by the caller.
    for (auto &entry : cancelledRequests_) {
        if (entry.release) {
            delete entry.reqH;
        }
    }
    cancelledRequests_.clear();

    // Producer queues may still hold handles the progress thread never popped.
    // No DOCA submit ever happened for these, so deleting them is safe.
    for (auto &entry : vecA_) {
        delete entry.reqH;
    }
    vecA_.clear();
    for (auto &entry : vecB_) {
        delete entry.reqH;
    }
    vecB_.clear();

    cleanupDocaResources();
}

bool
nixlDocaMemosProgressEngine::trySubmitExistTask(nixlDocaMemosBackendReqH *req_h,
                                                bool &tasks_queued) const {
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        return false;
    }

    const docaMemosKey &key = req_h->objectKeys_[0];

    auto *task_ctx = &req_h->taskContexts_[0];
    task_ctx->reqH = req_h;
    task_ctx->taskIndex = 0;
    union doca_data task_user_data = {.ptr = task_ctx};

    struct doca_kv_io_task_exist *exist_task = nullptr;
    doca_error_t result =
        doca_kv_io_task_exist_alloc_init(kvIo_, boundDev_, task_user_data, &exist_task);
    if (result != DOCA_SUCCESS) {
        if (result == DOCA_ERROR_FULL || result == DOCA_ERROR_NO_MEMORY) {
            return false;
        }
        NIXL_ERROR << "Failed to allocate DOCA KV exist task: " << doca_error_get_descr(result);
        handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
        return true;
    }

    doca_kv_io_task_exist_set_key_conf(exist_task, key.key, key.keyLen);
    struct doca_task *doca_task = doca_kv_io_task_exist_as_task(exist_task);

    result = doca_task_submit_ex(doca_task, DOCA_TASK_SUBMIT_FLAG_NONE);
    if (result != DOCA_SUCCESS) {
        if (result == DOCA_ERROR_FULL) {
            doca_task_free(doca_task);
            return false;
        }
        NIXL_ERROR << "Failed to submit EXIST task: " << doca_error_get_descr(result);
        doca_task_free(doca_task);
        handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
        return true;
    }
    tasks_queued = true;
    req_h->submittedTasks_++;

    return true;
}

void
nixlThreadedProgressEngine::processCancellations(std::vector<cancelEntry> &cancels) const {
    for (auto &entry : cancels) {
        auto *req_h = entry.reqH;
        if (entry.release) {
            delete req_h;
            continue;
        }

        req_h->cancelled_.store(true, std::memory_order_release);
        if (req_h->isPending_) {
            auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
            if (it != pendingRequests_.end()) {
                pendingRequests_.erase(it);
            }
            req_h->isPending_ = false;
            dropUnsubmittedTasks(req_h);
        }
    }

    // The drain loop applies cancelled_ to a request it has not submitted yet,
    // so nothing re-applies these aborts after this point.
    const std::lock_guard<std::mutex> g(queueMutex_);
    for (auto &entry : cancels) {
        if (!entry.release) {
            entry.reqH->abortQueued_ = false;
        }
    }
}

void
nixlThreadedProgressEngine::progressThreadFunc() {
    NIXL_INFO << "Progress thread running";

    while (!threadStop_.load(std::memory_order_acquire)) {
        bool made_progress = false;
        bool tasks_queued = false;

        if (pe_) {
            for (int n = 0; n < kProgressBurst && doca_pe_progress(pe_) != 0; n++) {
                made_progress = true;
            }
        }

        while (!pendingRequests_.empty()) {
            auto *req_h = pendingRequests_.front();
            bool completed;
            if (req_h->isExistQuery_) {
                completed = trySubmitExistTask(req_h, tasks_queued);
            } else {
                completed = trySubmitRequest(req_h,
                                             req_h->storedOperation_,
                                             *req_h->storedLocal_,
                                             *req_h->storedRemote_,
                                             tasks_queued);
            }
            if (!completed) {
                break;
            }
            pendingRequests_.pop_front();
            req_h->isPending_ = false;
            made_progress = true;
        }

        if (hasNewWork_.load(std::memory_order_acquire)) {
            std::vector<pendingEntry> *drain_vec;
            std::vector<cancelEntry> cancels;
            {
                const std::lock_guard<std::mutex> g(queueMutex_);
                drain_vec = producerVec_;
                producerVec_ = (producerVec_ == &vecA_) ? &vecB_ : &vecA_;
                cancels.swap(cancelledRequests_);
                hasNewWork_.store(false, std::memory_order_release);
            }

            if (!cancels.empty()) {
                processCancellations(cancels);
                made_progress = true;
            }

            for (auto &entry : *drain_vec) {
                // An abort that lands after this vector was swapped out cannot
                // be found in the producer queue. processCancellations() has
                // already set cancelled_, so never submit the request.
                if (entry.reqH->cancelled_.load(std::memory_order_acquire)) {
                    dropUnsubmittedTasks(entry.reqH);
                    made_progress = true;
                    continue;
                }

                bool fully_submitted;
                if (entry.reqH->isExistQuery_) {
                    fully_submitted = trySubmitExistTask(entry.reqH, tasks_queued);
                } else {
                    fully_submitted = trySubmitRequest(
                        entry.reqH, entry.operation, *entry.local, *entry.remote, tasks_queued);
                }

                if (!fully_submitted) {
                    if (!entry.reqH->isExistQuery_) {
                        entry.reqH->storedOperation_ = entry.operation;
                        entry.reqH->storedLocal_ = std::move(entry.local);
                        entry.reqH->storedRemote_ = std::move(entry.remote);
                    }
                    entry.reqH->isPending_ = true;
                    pendingRequests_.push_back(entry.reqH);
                }
                made_progress = true;
            }
            drain_vec->clear();
        }

        flushSubmittedTasks(tasks_queued);
        made_progress = made_progress || tasks_queued;

        if (made_progress) {
            if (threadDelay_.count() > 0) {
                std::this_thread::sleep_for(threadDelay_);
            }
            continue;
        }

        // delay_us = 0 means "busy-spin" (the user opted in, see the warning
        // in createProgressEngine). Skip the condvar entirely so we keep
        // polling doca_pe_progress at full rate.
        if (threadDelay_.count() == 0) {
            continue;
        }

        // Idle path: wait on the condvar to amortise the cost of polling
        // doca_pe_progress. Producers notify after setting hasNewWork_; the
        // predicate is also checked on spurious wakeups.
        std::unique_lock<std::mutex> lk(wakeupMutex_);
        wakeup_.wait_for(lk, threadDelay_, [this] {
            return threadStop_.load(std::memory_order_acquire) ||
                hasNewWork_.load(std::memory_order_acquire);
        });
    }

    NIXL_INFO << "Progress thread exiting";
}

nixl_status_t
nixlThreadedProgressEngine::queryMem(const nixl_reg_dlist_t &descs,
                                     std::vector<nixl_query_resp_t> &resp,
                                     bool convert_key_to_128bit,
                                     size_t max_key_len) const {
    size_t count = static_cast<size_t>(descs.descCount());
    resp.reserve(count);

    std::vector<nixlDocaMemosBackendReqH> query_handles;
    prepareQueryHandles(descs, query_handles, convert_key_to_128bit, max_key_len);

    // Lifetime contract: we hand &query_handles[i] to the progress thread via
    // producerVec_. Two invariants must hold for this to stay safe:
    //   1. prepareQueryHandles() reserved capacity, so query_handles never
    //      reallocates and the addresses we publish stay valid.
    //   2. waitForQueryCompletion() below blocks until every callback has
    //      fired, so query_handles outlives all in-flight DOCA tasks. Adding
    //      a timeout to the wait would silently introduce a use-after-free.
    size_t enqueued = 0;
    {
        const std::lock_guard<std::mutex> g(queueMutex_);
        for (auto &req_h : query_handles) {
            if (!req_h.allTasksCompleted()) {
                // operation/local/remote are unused for exist queries; the
                // progress thread dispatches on req_h->isExistQuery_.
                producerVec_->push_back({&req_h, {}, nullptr, nullptr});
                enqueued++;
            }
        }
        if (enqueued > 0) {
            hasNewWork_.store(true, std::memory_order_release);
        }
    }
    if (enqueued > 0) {
        wakeup_.notify_one();
        // Nothing to wait for when every handle already completed (e.g. an
        // empty batch or unresolvable keys); fall through to collect
        // the results that are already in place, matching the no-thread engine.
        waitForQueryCompletion(query_handles, nullptr);
    }

    size_t successful_queries = collectQueryResults(query_handles, resp);
    if (successful_queries < count) {
        NIXL_ERROR << (count - successful_queries) << " of " << count << " query task(s) failed";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
