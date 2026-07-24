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

#ifndef NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_PROGRESS_ENGINE_H
#define NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_PROGRESS_ENGINE_H

#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include "nixl_types.h"
#include "backend/backend_aux.h" // For nixl_meta_dlist_t

// Forward declaration suffices for the static callback declarations below
// (function declarations only need an incomplete type for value parameters).
// The full definition is pulled in by the .cpp via the real DOCA headers (or
// by the mock headers in unit tests).
union doca_data;

// Forward declarations for DOCA types
struct doca_pe;
struct doca_kvdev;
struct doca_kvdev_io;
struct doca_nvme_kernel_kvdev;
struct doca_nvme_kernel_kvdev_io;
struct doca_ctx;
struct doca_task;
// Forward declaration for request handle
class nixlDocaMemosBackendReqH;

/**
 * @brief Base class for DOCA KV progress engine management.
 *
 * Owns the DOCA progress-engine, KV-IO context, and the shared submission
 * helpers (trySubmitRequest, trySubmitExistTask, collectQueryResults).
 *
 * pendingRequests_ is the only piece of mutable state that lives in the base
 * because both subclasses share the helpers that consult / mutate it.
 * Synchronisation around it is the subclass's responsibility:
 *   - nixlNoThreadProgressEngine guards every access with its own lock_.
 *   - nixlThreadedProgressEngine touches it only from the progress thread.
 *
 * Per-subclass state (cancellation queues, mutexes, double-buffers, etc.)
 * lives in the subclasses, not here.
 */
class nixlDocaMemosProgressEngine {
public:
    virtual ~nixlDocaMemosProgressEngine() = default;

    bool
    hasInitError() const {
        return initErr_;
    }

    virtual nixl_status_t
    checkXfer(nixlDocaMemosBackendReqH *req_h) const;
    virtual nixl_status_t
    postXfer(nixlDocaMemosBackendReqH *req_h,
             const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote) const = 0;
    virtual void
    cancelRequest(nixlDocaMemosBackendReqH *req_h) const = 0;
    virtual nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const = 0;

protected:
    nixlDocaMemosProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
                                uint32_t num_tasks,
                                uint32_t max_value_len);
    void
    cleanupDocaResources();
    bool
    trySubmitRequest(nixlDocaMemosBackendReqH *req_h,
                     const nixl_xfer_op_t &operation,
                     const nixl_meta_dlist_t &local,
                     const nixl_meta_dlist_t &remote) const;
    bool
    trySubmitExistTask(nixlDocaMemosBackendReqH *req_h) const;
    void
    prepareQueryHandles(const nixl_reg_dlist_t &descs,
                        std::vector<nixlDocaMemosBackendReqH> &query_handles) const;
    void
    waitForQueryCompletion(const std::vector<nixlDocaMemosBackendReqH> &query_handles,
                           const std::function<void()> &poll) const;
    size_t
    collectQueryResults(const std::vector<nixlDocaMemosBackendReqH> &query_handles,
                        std::vector<nixl_query_resp_t> &resp) const;

    // DOCA task callbacks and submission-failure helper. They live as static
    // members so they can touch nixlDocaMemosBackendReqH's private bookkeeping
    // through the friendship granted to nixlDocaMemosProgressEngine.
    static void
    taskCompletionCallback(struct doca_task *task,
                           union doca_data task_user_data,
                           union doca_data ctx_user_data);
    static void
    taskErrorCallback(struct doca_task *task,
                      union doca_data task_user_data,
                      union doca_data ctx_user_data);
    static void
    handleSubmissionFailure(nixlDocaMemosBackendReqH *req_h, nixl_status_t status);

    struct doca_pe *pe_ = nullptr;
    struct doca_nvme_kernel_kvdev_io *kkvIo_ = nullptr;
    struct doca_kvdev_io *kvIo_ = nullptr;
    struct doca_ctx *ctx_ = nullptr;

    // Synchronisation owned by the subclass; see class doc.
    mutable std::deque<nixlDocaMemosBackendReqH *> pendingRequests_;
    bool initErr_ = false;
    uint32_t maxValueLen_ = 0;
};

/**
 * @brief Synchronous progress engine. The caller's thread drives the DOCA PE
 * via checkXfer()/queryMem(). All shared state is serialised by lock_.
 */
class nixlNoThreadProgressEngine : public nixlDocaMemosProgressEngine {
public:
    nixlNoThreadProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
                               uint32_t num_tasks,
                               uint32_t max_value_len);
    ~nixlNoThreadProgressEngine() override;

    nixl_status_t
    postXfer(nixlDocaMemosBackendReqH *req_h,
             const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote) const override;
    void
    cancelRequest(nixlDocaMemosBackendReqH *req_h) const override;
    nixl_status_t
    checkXfer(nixlDocaMemosBackendReqH *req_h) const override;
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

private:
    int
    progress() const;
    void
    tryResumePendingRequests() const;

    mutable std::vector<nixlDocaMemosBackendReqH *> cancelledRequests_;
    mutable std::mutex lock_;
};

/**
 * @brief Threaded progress engine — lock-free hot path.
 *
 * The progress thread is the sole caller of all DOCA APIs (doca_pe_progress,
 * doca_kvdev_io_task_*_alloc_init, doca_task_submit, etc.).
 *
 * Producers (postXfer, cancelRequest) append to *producerVec_ and to
 * cancelledRequests_ under queueMutex_ and set hasNewWork_. The progress thread,
 * when it observes hasNewWork_, briefly takes queueMutex_ to swap producerVec_
 * between vecA_ and vecB_ and to swap-out cancelledRequests_, then drains the
 * inactive vector outside the lock. This keeps producers' critical sections
 * to a single push + pointer swap, regardless of how much work the consumer
 * has queued up.
 */
class nixlThreadedProgressEngine : public nixlDocaMemosProgressEngine {
public:
    nixlThreadedProgressEngine(struct doca_nvme_kernel_kvdev *nvme_kvdev,
                               uint32_t num_tasks,
                               uint32_t max_value_len,
                               std::chrono::microseconds thread_delay);
    ~nixlThreadedProgressEngine() override;

    nixl_status_t
    postXfer(nixlDocaMemosBackendReqH *req_h,
             const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote) const override;
    void
    cancelRequest(nixlDocaMemosBackendReqH *req_h) const override;
    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

private:
    struct pendingEntry {
        nixlDocaMemosBackendReqH *reqH;
        nixl_xfer_op_t operation;
        std::unique_ptr<nixl_meta_dlist_t> local;
        std::unique_ptr<nixl_meta_dlist_t> remote;
    };

    void
    progressThreadFunc();
    void
    processCancellations(std::vector<nixlDocaMemosBackendReqH *> &cancels) const;

    std::chrono::microseconds threadDelay_;
    std::atomic<bool> threadStop_{false};
    std::thread progressThread_;

    mutable std::mutex queueMutex_;
    mutable std::atomic<bool> hasNewWork_{false};
    mutable std::vector<pendingEntry> vecA_;
    mutable std::vector<pendingEntry> vecB_;
    mutable std::vector<pendingEntry> *producerVec_ = &vecA_;
    mutable std::vector<nixlDocaMemosBackendReqH *> cancelledRequests_;
    mutable std::vector<nixlDocaMemosBackendReqH *> pendingDeletes_;

    mutable std::mutex wakeupMutex_;
    mutable std::condition_variable wakeup_;
};

#endif // NIXL_SRC_PLUGINS_DOCA_MEMOS_DOCA_MEMOS_PROGRESS_ENGINE_H
