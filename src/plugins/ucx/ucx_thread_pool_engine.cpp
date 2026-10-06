/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_thread_pool_engine.h"
#include "ucx_backend_req.h"

#include "common/backend.h"
#include "common/blocking_queue.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <latch>
#include <ostream>

#include "absl/container/inlined_vector.h"

#include "ucx_utils.h"

namespace {

constexpr size_t inline_worker_tasks = 16;

class nixlUcxDedicatedWorker;
struct nixlUcxBackendSharedState;

/**
 * @brief Task executed by a single dedicated worker thread.
 */
class nixlUcxThreadTask : public nixl::blockingQueue<nixlUcxThreadTask>::node {
public:
    virtual void
    run(nixlUcxDedicatedWorker &worker) noexcept = 0;

protected:
    ~nixlUcxThreadTask() = default;
};

/**
 * @brief Task executed by multiple dedicated threads, with shared completion tracking.
 */
template<typename taskFunctionType> class nixlUcxThreadPoolTask final {
public:
    nixlUcxThreadPoolTask(size_t num_tasks, taskFunctionType &task_function)
        : taskFunction_(task_function),
          status_(NIXL_SUCCESS),
          completed_(num_tasks) {
        tasks_.reserve(num_tasks);
        for (size_t i = 0; i < num_tasks; ++i) {
            tasks_.emplace_back(*this);
        }
    }

    [[nodiscard]] nixlUcxThreadTask *
    getTask(size_t idx) {
        return &tasks_[idx];
    }

    void
    run(nixlUcxDedicatedWorker &worker) noexcept {
        try {
            const nixl_status_t ret = taskFunction_(worker);
            if (ret != NIXL_SUCCESS) {
                status_.store(ret);
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Threadpool task failed: " << e.what();
            status_.store(NIXL_ERR_BACKEND);
        }
        completed_.count_down();
    }

    [[nodiscard]] nixl_status_t
    waitAll() {
        completed_.wait();
        return status_.load();
    }

private:
    struct subtask final : nixlUcxThreadTask {
        explicit subtask(nixlUcxThreadPoolTask &task) noexcept : task(task) {}

        void
        run(nixlUcxDedicatedWorker &worker) noexcept override {
            task.run(worker);
        }

        nixlUcxThreadPoolTask &task;
    };

    taskFunctionType &taskFunction_;
    std::atomic<nixl_status_t> status_;
    std::latch completed_;
    absl::InlinedVector<subtask, inline_worker_tasks> tasks_;
};

/*
 * This class represents a chunk of a composite request.
 * It is used to encapsulate a batch of requests (subset of the larger batch)
 * performed by a dedicated worker thread of threadpool. It holds a shared state
 * with the main request to track its completion status and control the lifetime.
 */
class nixlUcxChunkBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxChunkBackendReqH() : nixlUcxBackendReqH(nullptr) {}

    void
    startXfer(const std::shared_ptr<nixlUcxBackendSharedState> &shared_state,
              nixlUcxWorker *worker) {
        NIXL_ASSERT(sharedState_.get() == nullptr);
        sharedState_ = shared_state;
        setWorker(worker);
    }

    void
    complete(nixl_status_t status);

    [[nodiscard]] nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendReqH &chunk) {
        std::string id = chunk.getWorker() ? std::to_string(chunk.getWorkerId()) : "none";
        os << "chunk " << &chunk << "{worker_id: " << id << ", state: " << chunk.sharedState_.get()
           << "}";
        return os;
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
};

/*
 * This class represents a shared state between a main request and all of its
 * chunks. It is used to track the completion status of the request and the
 * number of pending requests, and to control the lifetime of the chunks.
 */
struct nixlUcxBackendSharedState {
    std::atomic<nixl_status_t> status;
    std::atomic<size_t> pendingReqs;
    std::vector<nixlUcxChunkBackendReqH> chunks;

    nixlUcxBackendSharedState() : status(NIXL_SUCCESS), pendingReqs(0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxBackendSharedState &state) {
        return os << "state " << &state << "{status: " << state.status.load()
                  << ", pending=" << state.pendingReqs.load() << "}";
    }
};

void
nixlUcxChunkBackendReqH::complete(const nixl_status_t status) {
    NIXL_ASSERT(sharedState_.get() != nullptr);
    if (status != NIXL_SUCCESS) {
        nixlUcxBackendReqH::release();
        sharedState_->status.store(status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << status << ", " << *sharedState_;
    setWorker(nullptr);
    sharedState_.reset();
}

nixl_status_t
nixlUcxChunkBackendReqH::status() {
    // First check if entire request was cancelled or failed
    const nixl_status_t status = sharedState_->status.load();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return nixlUcxBackendReqH::status();
}

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxCompositeBackendReqH(nixlUcxWorker *worker, size_t num_chunks)
        : nixlUcxBackendReqH(worker),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()) {
        sharedState_->chunks.resize(num_chunks);
    }

    [[nodiscard]] size_t
    getNumChunks() const noexcept {
        return sharedState_ ? sharedState_->chunks.size() : 0;
    }

    void
    startXfer() {
        NIXL_ASSERT(sharedState_->pendingReqs.load() == 0);
        sharedState_->status.store(NIXL_SUCCESS);
        sharedState_->pendingReqs.store(getNumChunks());
    }

    [[nodiscard]] nixlUcxChunkBackendReqH *
    startChunk(size_t idx, nixlUcxWorker *worker) {
        nixlUcxChunkBackendReqH *chunk = &sharedState_->chunks[idx];
        chunk->startXfer(sharedState_, worker);
        NIXL_TRACE << "dedicated " << *nixlUcxThread::tlsThread() << " starting " << *chunk;
        return chunk;
    }

    [[nodiscard]] bool
    isComposite() const noexcept override {
        return true;
    }

    void
    release() override {
        NIXL_TRACE << *this << " releasing";
        nixlUcxBackendReqH::release();
        if (sharedState_) {
            // Set failed status to stop progress chunks
            sharedState_->status.store(NIXL_ERR_NOT_FOUND);
            // Reset shared state - it will be effectively released when the last chunk
            // resets the shared state pointer
            sharedState_.reset();
        }
    }

    [[nodiscard]] nixl_status_t
    status() override {
        getWorker()->progressLoop();

        if (sharedState_->pendingReqs.load()) {
            return NIXL_IN_PROG;
        }

        const nixl_status_t status = nixlUcxBackendReqH::status();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        return sharedState_->status.load();
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxCompositeBackendReqH &handle) {
        os << "composite handle " << &handle << "{chunks: " << handle.getNumChunks();
        if (handle.sharedState_) {
            os << ", " << *handle.sharedState_;
        } else {
            os << ", state: nullptr";
        }
        return os << "}}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
};

} // namespace

class nixlUcxDedicatedThread final : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, nixlUcxDedicatedWorker &worker);

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return static_cast<nixlUcxDedicatedThread *>(tlsThread());
    }

    /**
     * @brief Queue a task for execution on this thread without taking ownership
     * @param task Task the caller must keep alive until execution completes
     */
    void
    post(nixlUcxThreadTask *task) {
        queue_.push(task);
    }

    void
    addRequest(nixlUcxChunkBackendReqH *handle) {
        NIXL_TRACE << "dedicated " << *this << " sent " << *handle;
        requests_.push_back(handle);
    }

protected:
    void
    run(std::stop_token token) override {
        NIXL_DEBUG << "dedicated " << *this << " running";

        while (!token.stop_requested()) {
            nixlUcxThreadTask *task;
            if (!requests_.empty()) {
                task = queue_.tryPop();
            } else {
                NIXL_TRACE << "dedicated " << *this << " waiting for requests";
                task = queue_.pop(token);
            }

            if (task != nullptr) {
                task->run(worker_);
            }

            for (auto it = requests_.begin(); it != requests_.end();) {
                nixl_status_t status = (*it)->status();
                if (status != NIXL_IN_PROG) {
                    NIXL_TRACE << "dedicated " << *this << " completing " << *(*it)
                               << " with status: " << status;
                    (*it)->complete(status);
                    it = requests_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (!requests_.empty()) {
            NIXL_WARN << "dedicated " << *this << " dropping " << requests_.size()
                      << " requests on exit";
            for (auto *req : requests_) {
                NIXL_INFO << "dropping " << *req;
                req->complete(NIXL_ERR_BACKEND);
            }
            requests_.clear();
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    nixlUcxDedicatedWorker &worker_;
    nixl::blockingQueue<nixlUcxThreadTask> queue_;
    std::vector<nixlUcxChunkBackendReqH *> requests_;
    std::jthread thread_;
};

namespace {

/**
 * @brief UCX worker that owns its dedicated thread.
 */
class nixlUcxDedicatedWorker : public nixlUcxWorker {
public:
    nixlUcxDedicatedWorker(const nixlUcxContext &context,
                           ucp_err_handling_mode_t err_handling_mode,
                           size_t id,
                           nixlUcxEngine *engine)
        : nixlUcxWorker(context, err_handling_mode, id),
          thread_(std::in_place, engine, *this) {}

    void
    stopThread() {
        thread_.reset();
    }

    [[nodiscard]] nixlUcxDedicatedThread &
    getThread() noexcept {
        return *thread_;
    }

private:
    std::optional<nixlUcxDedicatedThread> thread_;
};

} // namespace

nixlUcxDedicatedThread::nixlUcxDedicatedThread(nixlUcxEngine *engine,
                                               nixlUcxDedicatedWorker &worker)
    : nixlUcxThread(engine, {&worker}),
      worker_(worker),
      thread_(startThread()) {}

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params,
                                                 size_t num_threads)
    : nixlUcxThreadEngine(init_params, num_threads),
      splitBatchSize_(
          std::max<size_t>(num_threads,
                           nixl::getBackendParamDefaulted(init_params.customParams,
                                                          "split_batch_size",
                                                          isSglEnabled() ? 4096u : 1024u))) {

    for (size_t i = 0; i < num_threads; ++i) {
        addWorker<nixlUcxDedicatedWorker>(this);
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    for (const auto &worker : getDedicatedWorkers()) {
        static_cast<nixlUcxDedicatedWorker *>(worker.get())->stopThread();
    }
}

template<typename callbackType>
nixl_status_t
nixlUcxThreadPoolEngine::execute(callbackType &&callback) const {
    const auto workers = getDedicatedWorkers();
    nixlUcxThreadPoolTask task(workers.size(), callback);

    for (size_t i = 0; i < workers.size(); ++i) {
        auto &worker = *static_cast<nixlUcxDedicatedWorker *>(workers[i].get());
        worker.getThread().post(task.getTask(i));
    }

    return task.waitAll();
}

nixl_status_t
nixlUcxThreadPoolEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    size_t batch_size = local.descCount();
    if (batch_size < splitBatchSize_) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    const size_t num_chunks = getDedicatedWorkers().size();
    const auto comp_handle =
        new nixlUcxCompositeBackendReqH(getSharedWorker(getSharedWorkerId()).get(), num_chunks);
    NIXL_TRACE << "created " << *comp_handle;
    handle = comp_handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxThreadPoolEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH *handle,
                                       size_t start_idx,
                                       size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    if (!int_handle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    const auto comp_handle = static_cast<nixlUcxCompositeBackendReqH *>(int_handle);
    comp_handle->startXfer();
    const size_t batch_size = local.descCount();
    const size_t num_chunks = comp_handle->getNumChunks();
    NIXL_TRACE << "sending " << *comp_handle;

    const nixl_status_t status = execute([&](nixlUcxDedicatedWorker &worker) {
        const size_t i = worker.getId() - getSharedWorkersSize();
        nixlUcxChunkBackendReqH *chunk_handle = comp_handle->startChunk(i, &worker);

        const size_t chunk_start = i * batch_size / num_chunks;
        const size_t chunk_end = (i + 1) * batch_size / num_chunks;
        try {
            const nixl_status_t ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunk_handle, chunk_start, chunk_end);
            if (ret != NIXL_SUCCESS) {
                chunk_handle->complete(ret);
            } else {
                worker.getThread().addRequest(chunk_handle);
            }
            return ret;
        }
        catch (const std::exception &) {
            chunk_handle->complete(NIXL_ERR_BACKEND);
            throw;
        }
    });

    NIXL_TRACE << "sent " << *comp_handle << " with status: " << status;
    return status;
}
