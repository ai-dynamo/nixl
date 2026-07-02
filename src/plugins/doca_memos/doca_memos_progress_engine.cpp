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
#include <doca_kvdev.h>
#include <doca_kvdev_io.h>
#include <doca_nvme_kernel_kvdev.h>
#include <doca_nvme_kernel_kvdev_io.h>
#include <doca_ctx.h>
#include <doca_error.h>

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
nixlDocaMemosProgressEngine::taskCompletionCallback(struct doca_task *task,
                                                    union doca_data task_user_data,
                                                    union doca_data ctx_user_data) {
    (void)ctx_user_data;

    auto *task_ctx = static_cast<docaMemosTaskContext *>(task_user_data.ptr);
    if (task_ctx && task_ctx->reqH) {
        auto *req_h = task_ctx->reqH;
        int task_index = task_ctx->taskIndex;
        // Relaxed load: missing a recent cancel is harmless because
        // processCancellations reconciles totalTasks_ afterwards. Anything
        // that changes how cancellations are reaped must preserve that.
        bool is_cancelled = req_h->cancelled_.load(std::memory_order_relaxed);

        if (!is_cancelled) {
            if (req_h->isExistQuery_) {
                req_h->taskResult_.store(static_cast<int>(DOCA_SUCCESS), std::memory_order_release);
                NIXL_DEBUG << "EXIST query completed - key exists (task " << task_index << ")";
            } else if (task_ctx->isRetrieve) {
                const struct doca_kvdev_io_task_retrieve *retrieve_task =
                    doca_kvdev_io_task_retrieve_from_task(task);
                uint32_t result_len =
                    doca_kvdev_io_task_retrieve_get_result_value_len(retrieve_task);
                if (result_len != task_ctx->expectedValueLen) {
                    NIXL_WARN << "Task " << task_index << " retrieved " << result_len
                              << " bytes, expected " << task_ctx->expectedValueLen;
                }
                NIXL_DEBUG << "Task " << task_index << " completed successfully (" << result_len
                           << " bytes)";
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
nixlDocaMemosProgressEngine::taskErrorCallback(struct doca_task *task,
                                               union doca_data task_user_data,
                                               union doca_data ctx_user_data) {
    (void)ctx_user_data;

    auto *task_ctx = static_cast<docaMemosTaskContext *>(task_user_data.ptr);
    if (task_ctx && task_ctx->reqH) {
        auto *req_h = task_ctx->reqH;
        int task_index = task_ctx->taskIndex;
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
            } else if (task_err == DOCA_ERROR_NOT_FOUND && req_h->ignoreNotFound_) {
                NIXL_DEBUG << "Task " << task_index
                           << " got key-not-found on retrieve, ignoring per configuration";
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

nixlDocaMemosProgressEngine::nixlDocaMemosProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
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

    result = doca_nvme_kernel_kvdev_io_create(nvme_kvdev, &kkvIo_);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA NVMe kernel KV IO context: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    kvIo_ = doca_nvme_kernel_kvdev_io_as_kvdev_io(kkvIo_);
    if (!kvIo_) {
        NIXL_ERROR << "Failed to convert NVMe kernel KV IO context to generic KV IO context";
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kvdev_io_set_num_tasks(kvIo_, num_tasks);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set DOCA KV IO task count: " << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kvdev_io_set_task_completion_cb(kvIo_, taskCompletionCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set DOCA KV IO completion callback: "
                   << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    result = doca_kvdev_io_set_task_error_cb(kvIo_, taskErrorCallback);
    if (result != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set DOCA KV IO error callback: " << doca_error_get_descr(result);
        cleanupDocaResources();
        initErr_ = true;
        return;
    }

    ctx_ = doca_kvdev_io_as_ctx(kvIo_);
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

    NIXL_DEBUG << "Progress engine base initialization complete";
}

void
nixlDocaMemosProgressEngine::cleanupDocaResources() {
    doca_error_t result;
    if (ctx_) {
        result = doca_ctx_stop(ctx_);
        if (result != DOCA_SUCCESS) {
            NIXL_WARN << "Failed to stop DOCA context: " << doca_error_get_descr(result);
        }
        ctx_ = nullptr;
    }

    if (kkvIo_) {
        result = doca_nvme_kernel_kvdev_io_destroy(kkvIo_);
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
        return req_h->getOverallStatus();
    }

    return NIXL_IN_PROG;
}

bool
nixlDocaMemosProgressEngine::trySubmitRequest(nixlDocaMemosBackendReqH *req_h,
                                              const nixl_xfer_op_t &operation,
                                              const nixl_meta_dlist_t &local,
                                              const nixl_meta_dlist_t &remote) const {
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        NIXL_DEBUG << "Pending queue not empty - queuing new request without submission";
        req_h->nextDescriptorIndex_ = 0;
        return false;
    }

    for (int i = req_h->nextDescriptorIndex_; i < local.descCount(); i++) {
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
            struct doca_kvdev_io_task_store *store_task = nullptr;
            result = doca_kvdev_io_task_store_alloc_init(kvIo_, task_user_data, &store_task);
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
            doca_kvdev_io_task_store_set_key_value_conf(store_task,
                                                        object_key.key,
                                                        object_key.keyLen,
                                                        &req_h->valueIovecs_[i],
                                                        1,
                                                        value_len);
            doca_task = doca_kvdev_io_task_store_as_task(store_task);
        } else { // NIXL_READ
            struct doca_kvdev_io_task_retrieve *retrieve_task = nullptr;
            result = doca_kvdev_io_task_retrieve_alloc_init(kvIo_, task_user_data, &retrieve_task);
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
            doca_kvdev_io_task_retrieve_set_key_value_conf(retrieve_task,
                                                           object_key.key,
                                                           object_key.keyLen,
                                                           &req_h->valueIovecs_[i],
                                                           1,
                                                           value_len);
            doca_task = doca_kvdev_io_task_retrieve_as_task(retrieve_task);
        }

        // Bump submittedTasks_ only after a successful submit, so the count
        // stays authoritative even if doca_task_submit invokes the callback
        // synchronously on failure.
        result = doca_task_submit(doca_task);
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
        req_h->submittedTasks_++;
        req_h->nextDescriptorIndex_ = i + 1;
    }

    return true;
}

nixl_status_t
nixlNoThreadProgressEngine::postXfer(nixlDocaMemosBackendReqH *req_h,
                                     const nixl_xfer_op_t &operation,
                                     const nixl_meta_dlist_t &local,
                                     const nixl_meta_dlist_t &remote) const {
    const std::lock_guard<std::mutex> guard(lock_);

    req_h->totalTasks_ = local.descCount();
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextDescriptorIndex_ = 0;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);
    req_h->overallStatus_ = NIXL_IN_PROG;

    NIXL_DEBUG << "Posting transfer with " << req_h->totalTasks_ << " tasks";

    bool fully_submitted = trySubmitRequest(req_h, operation, local, remote);

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
    const std::lock_guard<std::mutex> guard(lock_);

    while (!pendingRequests_.empty()) {
        auto *req_h = pendingRequests_.front();

        bool completed;
        if (req_h->isExistQuery_) {
            completed = trySubmitExistTask(req_h);
        } else {
            completed = trySubmitRequest(
                req_h, req_h->storedOperation_, *req_h->storedLocal_, *req_h->storedRemote_);
        }

        if (!completed) {
            break;
        }

        pendingRequests_.pop_front();
        req_h->isPending_ = false;
    }
}

void
nixlNoThreadProgressEngine::cancelRequest(nixlDocaMemosBackendReqH *req_h) const {
    const std::lock_guard<std::mutex> guard(lock_);

    if (req_h->isPending_) {
        auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
        if (it != pendingRequests_.end()) {
            pendingRequests_.erase(it);
        }
        req_h->isPending_ = false;
        req_h->totalTasks_ = req_h->submittedTasks_;
    }

    if (req_h->completedTasks_ >= req_h->totalTasks_) {
        delete req_h;
        return;
    }

    req_h->cancelled_.store(true, std::memory_order_release);
    cancelledRequests_.push_back(req_h);
}

int
nixlNoThreadProgressEngine::progress() const {
    const std::lock_guard<std::mutex> guard(lock_);
    if (!pe_) {
        return 0;
    }
    int ret = doca_pe_progress(pe_);

    for (auto it = cancelledRequests_.begin(); it != cancelledRequests_.end();) {
        if ((*it)->allTasksCompleted_.load(std::memory_order_acquire)) {
            delete *it;
            it = cancelledRequests_.erase(it);
        } else {
            ++it;
        }
    }

    return ret;
}

void
nixlDocaMemosProgressEngine::prepareQueryHandles(
    const nixl_reg_dlist_t &descs,
    std::vector<nixlDocaMemosBackendReqH> &query_handles) const {
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
        if (!nixlDocaMemosEngine::resolveMemosKey(
                desc.devId, desc.metaInfo, req_h.objectKeys_[0])) {
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

nixlNoThreadProgressEngine::nixlNoThreadProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
                                                       uint32_t num_tasks,
                                                       uint32_t max_value_len)
    : nixlDocaMemosProgressEngine(nvme_kvdev, num_tasks, max_value_len) {
    if (initErr_) {
        return;
    }

    NIXL_DEBUG << "Created no-thread progress engine";
}

namespace {

constexpr auto kDestructorDrainBudget = std::chrono::seconds(10);

} // anonymous namespace

nixlNoThreadProgressEngine::~nixlNoThreadProgressEngine() {
    // Drain in-flight tasks so DOCA doesn't see them still owned by ctx_
    // when cleanupDocaResources stops the context.
    auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;
    while (pe_ && std::chrono::steady_clock::now() < deadline) {
        int did = doca_pe_progress(pe_);
        for (auto it = cancelledRequests_.begin(); it != cancelledRequests_.end();) {
            if ((*it)->allTasksCompleted_.load(std::memory_order_acquire)) {
                delete *it;
                it = cancelledRequests_.erase(it);
            } else {
                ++it;
            }
        }
        if (did == 0 && cancelledRequests_.empty()) {
            break;
        }
    }
    if (!cancelledRequests_.empty()) {
        NIXL_WARN << "no-thread engine destructor: " << cancelledRequests_.size()
                  << " cancelled request(s) did not drain within " << kDestructorDrainBudget.count()
                  << "s";
    }
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
                                     std::vector<nixl_query_resp_t> &resp) const {
    resp.reserve(descs.descCount());

    std::vector<nixlDocaMemosBackendReqH> query_handles;
    prepareQueryHandles(descs, query_handles);

    // See nixlThreadedProgressEngine::queryMem for the lifetime contract:
    // &query_handles[i] is published into pendingRequests_ and into DOCA's
    // task_user_data, so query_handles must not reallocate (prepareQueryHandles
    // reserves) and waitForQueryCompletion must drain every callback before
    // we return.
    {
        const std::lock_guard<std::mutex> guard(lock_);
        for (auto &req_h : query_handles) {
            if (req_h.allTasksCompleted()) {
                continue;
            }
            if (!trySubmitExistTask(&req_h)) {
                req_h.isPending_ = true;
                pendingRequests_.push_back(&req_h);
            }
        }
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
    req_h->totalTasks_ = local.descCount();
    req_h->submittedTasks_ = 0;
    req_h->completedTasks_ = 0;
    req_h->nextDescriptorIndex_ = 0;
    req_h->allTasksCompleted_.store(false, std::memory_order_release);
    req_h->overallStatus_ = NIXL_IN_PROG;

    NIXL_DEBUG << "Posting transfer with " << req_h->totalTasks_ << " tasks (queued)";

    {
        const std::lock_guard<std::mutex> g(queueMutex_);
        producerVec_->push_back({req_h,
                                 operation,
                                 std::make_unique<nixl_meta_dlist_t>(local),
                                 std::make_unique<nixl_meta_dlist_t>(remote)});
        hasNewWork_.store(true, std::memory_order_release);
    }
    wakeup_.notify_one();

    return NIXL_IN_PROG;
}

void
nixlThreadedProgressEngine::cancelRequest(nixlDocaMemosBackendReqH *req_h) const {
    // If the request is still in the producer queue it can be deleted
    // immediately. Otherwise mark it cancelled here (not in
    // processCancellations) so any callbacks firing in the window before the
    // progress thread observes the cancel see cancelled=true and skip their
    // status bookkeeping.
    {
        const std::lock_guard<std::mutex> g(queueMutex_);

        auto it = std::find_if(producerVec_->begin(),
                               producerVec_->end(),
                               [req_h](const pendingEntry &e) { return e.reqH == req_h; });
        if (it != producerVec_->end()) {
            producerVec_->erase(it);
            delete req_h;
            return;
        }

        req_h->cancelled_.store(true, std::memory_order_release);
        cancelledRequests_.push_back(req_h);
        hasNewWork_.store(true, std::memory_order_release);
    }
    wakeup_.notify_one();
}

nixlThreadedProgressEngine::nixlThreadedProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
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

    auto deadline = std::chrono::steady_clock::now() + kDestructorDrainBudget;
    while (pe_ && std::chrono::steady_clock::now() < deadline) {
        if (doca_pe_progress(pe_) == 0) {
            break;
        }
    }
    if (pe_ && std::chrono::steady_clock::now() >= deadline) {
        NIXL_WARN << "threaded engine destructor: doca_pe drain timed out after "
                  << kDestructorDrainBudget.count() << "s";
    }

    for (auto *req_h : pendingDeletes_) {
        delete req_h;
    }
    pendingDeletes_.clear();

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

    for (auto *req_h : cancelledRequests_) {
        delete req_h;
    }
    cancelledRequests_.clear();

    // Anything still in pendingRequests_ has tasks that were submitted to DOCA
    // but did not complete within the drain budget; deleting them now would
    // race with any callback DOCA might still fire. Leak with a warning.
    if (!pendingRequests_.empty()) {
        NIXL_WARN << "threaded engine destructor: " << pendingRequests_.size()
                  << " request(s) with in-flight tasks did not drain; leaking to avoid UAF";
    }

    cleanupDocaResources();
}

bool
nixlDocaMemosProgressEngine::trySubmitExistTask(nixlDocaMemosBackendReqH *req_h) const {
    if (!pendingRequests_.empty() && !req_h->isPending_) {
        return false;
    }

    const docaMemosKey &key = req_h->objectKeys_[0];

    auto *task_ctx = &req_h->taskContexts_[0];
    task_ctx->reqH = req_h;
    task_ctx->taskIndex = 0;
    union doca_data task_user_data = {.ptr = task_ctx};

    struct doca_kvdev_io_task_exist *exist_task = nullptr;
    doca_error_t result = doca_kvdev_io_task_exist_alloc_init(kvIo_, task_user_data, &exist_task);
    if (result != DOCA_SUCCESS) {
        if (result == DOCA_ERROR_FULL || result == DOCA_ERROR_NO_MEMORY) {
            return false;
        }
        NIXL_ERROR << "Failed to allocate DOCA KV exist task: " << doca_error_get_descr(result);
        handleSubmissionFailure(req_h, NIXL_ERR_BACKEND);
        return true;
    }

    doca_kvdev_io_task_exist_set_key_conf(exist_task, key.key, key.keyLen);
    struct doca_task *doca_task = doca_kvdev_io_task_exist_as_task(exist_task);

    result = doca_task_submit(doca_task);
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
    req_h->submittedTasks_++;

    return true;
}

void
nixlThreadedProgressEngine::processCancellations(
    std::vector<nixlDocaMemosBackendReqH *> &cancels) const {
    // cancelled flag is already set by cancelRequest; here we reconcile queue
    // state and schedule deferred deletion once in-flight callbacks drain.
    for (auto *req_h : cancels) {
        if (req_h->isPending_) {
            auto it = std::find(pendingRequests_.begin(), pendingRequests_.end(), req_h);
            if (it != pendingRequests_.end()) {
                pendingRequests_.erase(it);
            }
            req_h->totalTasks_ = req_h->submittedTasks_;
        }
        req_h->isPending_ = false;
        if (req_h->completedTasks_ >= req_h->totalTasks_) {
            delete req_h;
        } else {
            pendingDeletes_.push_back(req_h);
        }
    }
    cancels.clear();
}

void
nixlThreadedProgressEngine::progressThreadFunc() {
    NIXL_INFO << "Progress thread running";

    while (!threadStop_.load(std::memory_order_acquire)) {
        bool made_progress = false;

        if (pe_) {
            int n = doca_pe_progress(pe_);
            if (n > 0) {
                made_progress = true;
            }
        }

        for (auto it = pendingDeletes_.begin(); it != pendingDeletes_.end();) {
            if ((*it)->allTasksCompleted_.load(std::memory_order_acquire)) {
                delete *it;
                it = pendingDeletes_.erase(it);
                made_progress = true;
            } else {
                ++it;
            }
        }

        while (!pendingRequests_.empty()) {
            auto *req_h = pendingRequests_.front();
            bool completed;
            if (req_h->isExistQuery_) {
                completed = trySubmitExistTask(req_h);
            } else {
                completed = trySubmitRequest(
                    req_h, req_h->storedOperation_, *req_h->storedLocal_, *req_h->storedRemote_);
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
            std::vector<nixlDocaMemosBackendReqH *> cancels;
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
                // A cancel that lands after this vector was swapped out cannot
                // be found in the producer queue, so cancelRequest() only sets
                // the flag and defers reclamation. Honor it here so a cancelled
                // request is never submitted to DOCA. Collapsing totalTasks_ to
                // submittedTasks_ lets the deferred processCancellations() free
                // it (nothing was submitted, so it completes immediately).
                if (entry.reqH->cancelled_.load(std::memory_order_acquire)) {
                    entry.reqH->totalTasks_ = entry.reqH->submittedTasks_;
                    made_progress = true;
                    continue;
                }

                bool fully_submitted;
                if (entry.reqH->isExistQuery_) {
                    fully_submitted = trySubmitExistTask(entry.reqH);
                } else {
                    fully_submitted =
                        trySubmitRequest(entry.reqH, entry.operation, *entry.local, *entry.remote);
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
                                     std::vector<nixl_query_resp_t> &resp) const {
    size_t count = static_cast<size_t>(descs.descCount());
    resp.reserve(count);

    std::vector<nixlDocaMemosBackendReqH> query_handles;
    prepareQueryHandles(descs, query_handles);

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
        // empty batch or the assume_success fast-path); fall through to collect
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
