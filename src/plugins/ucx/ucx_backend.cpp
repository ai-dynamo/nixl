/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"
#include "ucx/gpu_xfer_req_h.h"

#include <optional>
#include <limits>
#include <future>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include <asio.hpp>

#ifdef HAVE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#endif

namespace {
    void moveNotifList(notif_list_t &src, notif_list_t &tgt)
    {
        if (src.size() > 0) {
            std::move(src.begin(), src.end(), std::back_inserter(tgt));
            src.clear();
        }
    }
}

/****************************************
 * CUDA related code
 *****************************************/

class nixlUcxCudaCtx {
public:
#ifdef HAVE_CUDA
    CUcontext pthrCudaCtx;
    int myDevId;

    nixlUcxCudaCtx() {
        pthrCudaCtx = NULL;
        myDevId = -1;
    }
#endif
    void cudaResetCtxPtr();
    int cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated);
    int cudaSetCtx();
};

class nixlUcxCudaDevicePrimaryCtx {
#ifndef HAVE_CUDA
public:
    bool push() { return false; }
    void pop() {};
#else
    static constexpr int defaultCudaDeviceOrdinal = 0;
    int m_ordinal{defaultCudaDeviceOrdinal};
    CUdevice m_device{CU_DEVICE_INVALID};
    CUcontext m_context{nullptr};
public:

    bool push() {
        CUcontext context;

        const auto res = cuCtxGetCurrent(&context);
        if (res != CUDA_SUCCESS || context != nullptr) {
            return false;
        }

        if (m_context == nullptr) {
            CUresult res = cuDeviceGet(&m_device, m_ordinal);
            if (res != CUDA_SUCCESS) {
                return false;
            }

            res = cuDevicePrimaryCtxRetain(&m_context, m_device);
            if (res != CUDA_SUCCESS) {
                m_context = nullptr;
                return false;
            }
        }

        return cuCtxPushCurrent(m_context) == CUDA_SUCCESS;
    }

    void pop() {
        cuCtxPopCurrent(nullptr);
    }

    ~nixlUcxCudaDevicePrimaryCtx() {
        if (m_context != nullptr) {
            cuDevicePrimaryCtxRelease(m_device);
        }
    }
#endif
};

class nixlUcxCudaCtxGuard {
    nixlUcxCudaDevicePrimaryCtxPtr m_primary;
public:
    nixlUcxCudaCtxGuard(nixl_mem_t nixl_mem,
                        nixlUcxCudaDevicePrimaryCtxPtr primary) {
        if (nixl_mem == VRAM_SEG && primary && primary->push()) {
            m_primary = primary;
        }
    }
    ~nixlUcxCudaCtxGuard() {
        if (m_primary) {
            m_primary->pop();
        }
    }
};

#ifdef HAVE_CUDA

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

// This routine finds the CUDA context matching for the given input address.
int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    int ret;

    was_updated = false;

    /* TODO: proper error codes and log outputs through this method */
    if (expected_dev == -1) {
        return -1;
    }

    ret = cudaQueryAddr(address, is_dev, dev, ctx);
    if (ret) {
        return ret;
    }

    if (!is_dev) {
        return 0;
    }

    if (dev != expected_dev) {
        // User provided address that does not match dev_id
        return -1;
    }

    pthrCudaCtx = ctx;
    was_updated = true;
    myDevId = expected_dev;

    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx()
{
    CUresult result;
    if (NULL == pthrCudaCtx) {
        return 0;
    }

    result = cuCtxSetCurrent(pthrCudaCtx);

    return (CUDA_SUCCESS == result);
}

#else

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    was_updated = false;
    return 0;
}

int nixlUcxCudaCtx::cudaSetCtx() {
    return 0;
}

#endif


void nixlUcxEngine::vramInitCtx()
{
    cudaCtx = std::make_unique<nixlUcxCudaCtx>();
}

int
nixlUcxEngine::vramUpdateCtx(void *address, uint64_t dev_id, bool &restart_reqd) {
    int ret;
    bool was_updated;

    restart_reqd = false;

    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    ret = cudaCtx->cudaUpdateCtxPtr(address, dev_id, was_updated);
    if (ret) {
        return ret;
    }

    restart_reqd = was_updated;

    return 0;
}

int nixlUcxEngine::vramApplyCtx()
{
    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    return cudaCtx->cudaSetCtx();
}

void nixlUcxEngine::vramFiniCtx()
{
    cudaCtx.reset();
}

/****************************************
 * UCX request management
*****************************************/


class nixlUcxIntReq {
public:
    std::unique_ptr<std::string> amBuffer;

    bool
    is_complete() const {
        return completed_;
    }

    void
    completed() {
        completed_ = true;
    }

    void
    setConnection(ucx_connection_ptr_t conn) {
        conn_ = conn;
    }

    nixl_status_t
    checkConnection(size_t ep_id) const {
        NIXL_ASSERT(conn_) << "Connection is not set";
        return conn_->getEp(ep_id)->checkTxState();
    }

private:
    bool completed_ = false;
    ucx_connection_ptr_t conn_;
};

static void
nixlUcxReqSetConnection(nixlUcxReq req, ucx_connection_ptr_t conn) {
    nixlUcxIntReq *req_int = reinterpret_cast<nixlUcxIntReq *>(req);
    req_int->setConnection(conn);
}

static void _internalRequestInit(void *request)
{
    /* Initialize request in-place (aka "placement new")*/
    new(request) nixlUcxIntReq;
}

static void _internalRequestFini(void *request)
{
    /* Finalize request */
    nixlUcxIntReq *req = (nixlUcxIntReq*)request;
    req->~nixlUcxIntReq();
}


static void _internalRequestReset(nixlUcxIntReq *req) {
    _internalRequestFini((void *)req);
    _internalRequestInit((void *)req);
}

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendH : public nixlBackendReqH {
private:
    std::vector<nixlUcxIntReq *> requests_;
    nixlUcxWorker *worker;
    size_t worker_id;

    // Notification to be sent after completion of all requests
    struct Notif {
	    std::string agent;
	    nixl_blob_t payload;
	    Notif(const std::string& remote_agent, const nixl_blob_t& msg)
		    : agent(remote_agent), payload(msg) {}
    };
    std::optional<Notif> notif;

public:
    auto& notification() {
        return notif;
    }

    nixlUcxBackendH(nixlUcxWorker *worker, size_t worker_id)
        : worker(worker),
          worker_id(worker_id) {}

    void
    reserve(size_t size) {
        requests_.reserve(size);
    }

    void
    append(nixlUcxIntReq *req) {
        requests_.push_back(req);
    }

    virtual bool
    isComposite() const {
        return false;
    }

    virtual nixl_status_t
    release() {
        // TODO: Error log: uncompleted requests found! Cancelling ...
        for (nixlUcxIntReq *req : requests_) {
            if (!req->is_complete()) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker->reqCancel((nixlUcxReq)req);
            }
            _internalRequestReset(req);
            worker->reqRelease((nixlUcxReq)req);
        }
        requests_.clear();
        return NIXL_SUCCESS;
    }

    virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            return NIXL_SUCCESS;
        }

        /* Maximum progress */
        while (worker->progress())
            ;

        /* Go over all request updating their status */
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (nixlUcxIntReq *req : requests_) {
            nixl_status_t ret;
            if (!req->is_complete()) {
                ret = ucx_status_to_nixl(ucp_request_check_status((nixlUcxReq)req));
                switch (ret) {
                case NIXL_SUCCESS:
                    /* Mark as completed */
                    req->completed();
                    break;
                case NIXL_IN_PROG:
                    out_ret = NIXL_IN_PROG;
                    break;
                default:
                    // Any other ret value is ERR and will be returned
                    nixl_status_t conn_status = req->checkConnection(worker_id);
                    return (conn_status == NIXL_SUCCESS) ? ret : conn_status;
                }
            }
        }

        size_t incomplete_reqs = 0;
        for (nixlUcxIntReq *req : requests_) {
            if (req->is_complete()) {
                _internalRequestReset(req);
                worker->reqRelease((nixlUcxReq)req);
            } else {
                requests_[incomplete_reqs++] = req;
            }
        }
        requests_.resize(incomplete_reqs);
        return out_ret;
    }

    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(this->worker == nullptr || worker == nullptr);
        this->worker = worker;
        this->worker_id = worker_id;
    }

    nixlUcxWorker *
    getWorker() const {
        return worker;
    }

    size_t getWorkerId() const {
        return worker_id;
    }
};

/****************************************
 * Progress thread management
*****************************************/

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, std::function<void()> init, size_t num_workers)
        : engine_(engine),
          init_(std::move(init)) {
        workers_.reserve(num_workers);
    }

    virtual ~nixlUcxThread() {
        if (threadActive_) {
            join();
        }
    }

    void
    start() {
        NIXL_ASSERT(!threadActive_);
        threadActive_ = std::make_unique<std::promise<void>>();
        auto active = threadActive_->get_future();
        thread_ = std::make_unique<std::thread>(std::ref(*this));
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT(threadActive_);
        threadActive_.reset();
        thread_->join();
    }

    virtual void
    addWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(workers_.size() < workers_.capacity());
        workers_.push_back(worker);
        workerIds_.push_back(worker_id);
    }

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return workers_;
    }

    size_t
    getWorkerId(size_t idx = 0) const {
        return workerIds_[idx];
    }

    void
    operator()() {
        tlsThread() = this;
        init_();
        threadActive_->set_value();
        run();
    }

    static nixlUcxThread *&
    tlsThread() {
        static thread_local nixlUcxThread *tls = nullptr;
        return tls;
    }

    static bool
    isProgressThread(const nixlUcxEngine *engine) noexcept {
        nixlUcxThread *thread = tlsThread();
        return thread && thread->engine_ == engine;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workerIds_, ",") << "]}";
    }

protected:
    virtual void
    run() = 0;

private:
    const nixlUcxEngine *engine_;
    std::function<void()> init_;
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<std::promise<void>> threadActive_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine,
                        std::function<void()> init,
                        size_t num_workers,
                        nixlTime::us_t delay)
        : nixlUcxThread(engine, std::move(init), num_workers) {
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        int delay_us = std::min((int)delay, std::numeric_limits<int>::max());
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(controlPipe_[1], &signal, sizeof(signal));
        if (ret < 0) NIXL_PERROR << "write to progress thread control pipe failed";
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker, size_t worker_id) override {
        pollFds_[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker, worker_id);
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "shared " << *this << " running";
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthr_stop = false;
        while (!pthr_stop) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout) continue;
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    while (worker->progress())
                        ;
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0)
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";

            if (!ret) {
                timeout = true;
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                int ret = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (ret < 0) NIXL_PERROR << "read() on control pipe failed";

                pthr_stop = true;
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    int controlPipe_[2];
    std::vector<pollfd> pollFds_;
};

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl_ucx_mt_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t num_workers = getWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(
        this, [this]() { nixlUcxEngine::vramApplyCtx(); }, num_workers, init_params.pthrDelay);
    for (size_t i = 0; i < num_workers; i++) {
        thread_->addWorker(getWorkers()[i].get(), i);
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    thread_->join();
}

int
nixlUcxThreadEngine::vramApplyCtx() {
    thread_->join();
    thread_->start();
    return nixlUcxEngine::vramApplyCtx();
}

void
nixlUcxThreadEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        /* Append to the private list to allow batching */
        const std::lock_guard<std::mutex> lock(notifMtx_);
        notifPthr_.push_back(std::make_pair(std::move(remote_name), std::move(msg)));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    getNotifsImpl(notif_list);
    const std::lock_guard<std::mutex> lock(notifMtx_);
    moveNotifList(notifPthr_, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Threadpool engine
 ****************************************/

struct nixlUcxBackendSharedState;

/*
 * This class represents a chunk of a composite request.
 * It is used to encapsulate a batch of requests (subset of the larger batch)
 * performed by a dedicated worker thread of threadpool. It holds a shared state
 * with the main request to track its completion status and control the lifetime.
 */
class nixlUcxChunkBackendH : public nixlUcxBackendH {
public:
    nixlUcxChunkBackendH() : nixlUcxBackendH(nullptr, UINT64_MAX) {}

    void
    startXfer(const std::shared_ptr<nixlUcxBackendSharedState> &shared_state,
              nixlUcxWorker *worker,
              size_t worker_id) {
        NIXL_ASSERT(sharedState_.get() == nullptr);
        sharedState_ = shared_state;
        setWorker(worker, worker_id);
    }

    void
    complete(nixl_status_t status);

    nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendH &chunk) {
        return os << "chunk " << &chunk << "{worker_id: " << chunk.getWorkerId()
                  << ", state: " << chunk.sharedState_.get() << "}";
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
    std::vector<nixlUcxChunkBackendH> chunks;

    nixlUcxBackendSharedState() : status(NIXL_SUCCESS), pendingReqs(0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxBackendSharedState &state) {
        return os << "state " << &state << "{status: " << state.status.load()
                  << ", pending=" << state.pendingReqs.load() << "}";
    }
};

void
nixlUcxChunkBackendH::complete(nixl_status_t status) {
    NIXL_ASSERT(sharedState_.get() != nullptr);
    if (status != NIXL_SUCCESS) {
        nixlUcxBackendH::release();
        sharedState_->status.store(status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << status << ", " << *sharedState_;
    setWorker(nullptr, UINT64_MAX);
    sharedState_.reset();
}

nixl_status_t
nixlUcxChunkBackendH::status() {
    // First check if entire request was cancelled or failed
    nixl_status_t status = sharedState_->status.load();
    if (status == NIXL_SUCCESS) {
        status = nixlUcxBackendH::status();
    }
    return status;
}

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendH : public nixlUcxBackendH {
public:
    nixlUcxCompositeBackendH(nixlUcxWorker *worker,
                             size_t worker_id,
                             size_t chunk_size,
                             size_t num_chunks)
        : nixlUcxBackendH(worker, worker_id),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.resize(num_chunks);
    }

    size_t
    getChunkSize() const {
        return chunkSize_;
    }

    size_t
    getNumChunks() const {
        return sharedState_ ? sharedState_->chunks.size() : 0;
    }

    void
    startXfer() {
        NIXL_ASSERT(sharedState_->pendingReqs.load() == 0);
        sharedState_->status.store(NIXL_SUCCESS);
        sharedState_->pendingReqs.store(getNumChunks());
    }

    nixlUcxChunkBackendH *
    startChunk(size_t idx, nixlUcxWorker *worker, size_t worker_id) {
        nixlUcxChunkBackendH *chunk = &sharedState_->chunks[idx];
        chunk->startXfer(sharedState_, worker, worker_id);
        return chunk;
    }

    bool
    isComposite() const override {
        return true;
    }

    nixl_status_t
    release() override {
        NIXL_TRACE << *this << " releasing";
        nixl_status_t status = nixlUcxBackendH::release();
        if (sharedState_) {
            // Set failed status to stop progress chunks
            sharedState_->status.store(NIXL_ERR_NOT_FOUND);
            // Reset shared state - it will be effectively released when the last chunk
            // resets the shared state pointer
            sharedState_.reset();
        }

        return status;
    }

    nixl_status_t
    status() override {
        while (getWorker()->progress())
            ;

        if (sharedState_->pendingReqs.load()) {
            return NIXL_IN_PROG;
        }

        nixl_status_t status = nixlUcxBackendH::status();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        return sharedState_->status.load();
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxCompositeBackendH &handle) {
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
    size_t chunkSize_;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, std::function<void()> init, asio::io_context &io)
        : nixlUcxThread(engine, std::move(init), 1),
          io_(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendH *handle) {
        requests_.push_back(handle);
    }

protected:
    void
    run() override {
        auto guard = asio::make_work_guard(io_);
        NIXL_DEBUG << "dedicated " << *this << " running";

        while (!io_.stopped()) {
            if (!requests_.empty()) {
                io_.poll_one();
            } else {
                NIXL_TRACE << "dedicated " << *this << " waiting for requests";
                io_.run_one();
            }

            if (requests_.empty()) {
                continue;
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
            for (auto it = requests_.begin(); it != requests_.end();) {
                NIXL_INFO << "dropping " << *(*it);
                (*it)->complete(NIXL_ERR_BACKEND);
            }
            requests_.clear();
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    asio::io_context &io_;
    std::vector<nixlUcxChunkBackendH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    numSharedWorkers_ = getWorkers().size() - num_threads;
    NIXL_ASSERT(numSharedWorkers_ > 0);

    splitBatchSize_ = nixl_b_params_get(init_params.customParams, "split_batch_size", 1024);

    auto init = [this]() { nixlUcxEngine::vramApplyCtx(); };

    if (init_params.enableProgTh) {
        sharedThread_ = std::make_unique<nixlUcxSharedThread>(
            this, init, numSharedWorkers_, init_params.pthrDelay);
        for (size_t i = 0; i < numSharedWorkers_; i++) {
            sharedThread_->addWorker(getWorkers()[i].get(), i);
        }
        sharedThread_->start();
    }

    if (num_threads > 0) {
        io_.reset(new asio::io_context());
        dedicatedThreads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            size_t worker_id = numSharedWorkers_ + i;
            dedicatedThreads_.emplace_back(
                std::make_unique<nixlUcxDedicatedThread>(this, init, *io_));
            dedicatedThreads_.back()->addWorker(getWorker(worker_id).get(), worker_id);
            dedicatedThreads_.back()->start();
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    if (sharedThread_) {
        sharedThread_->join();
    }

    if (io_) {
        io_->stop();
        for (auto &thread : dedicatedThreads_) {
            thread->join();
        }
    }
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

    size_t chunk_size = std::max(batch_size / dedicatedThreads_.size(), splitBatchSize_);
    size_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    size_t worker_id = getWorkerId();
    auto comp_handle =
        new nixlUcxCompositeBackendH(getWorker(worker_id).get(), worker_id, chunk_size, num_chunks);
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
    nixlUcxBackendH *int_handle = static_cast<nixlUcxBackendH *>(handle);
    if (!int_handle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    nixlUcxCompositeBackendH *comp_handle = static_cast<nixlUcxCompositeBackendH *>(int_handle);
    comp_handle->startXfer();
    size_t chunk_size = comp_handle->getChunkSize();
    NIXL_TRACE << "sending " << *comp_handle;

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{comp_handle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < comp_handle->getNumChunks(); i++) {
        io_->post([&, i]() {
            auto thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxChunkBackendH *chunk_handle =
                comp_handle->startChunk(i, thread->getWorkers()[0], thread->getWorkerId());
            NIXL_TRACE << "dedicated " << *thread << " starting " << *chunk_handle;

            size_t start_idx = i * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, (size_t)local.descCount());
            nixl_status_t ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunk_handle, start_idx, end_idx);
            if (ret != NIXL_SUCCESS) {
                status.store(ret);
                chunk_handle->complete(ret);
            } else {
                NIXL_TRACE << "dedicated " << *thread << " sent " << *chunk_handle;
                thread->addRequest(chunk_handle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    NIXL_TRACE << "sent " << *comp_handle << " with status: " << status.load();
    return status.load();
}

int
nixlUcxThreadPoolEngine::vramApplyCtx() {
    if (sharedThread_) {
        sharedThread_->join();
        sharedThread_->start();
    }
    if (io_) {
        io_->stop();
        for (auto &thread : dedicatedThreads_) {
            thread->join();
        }
        io_->restart();
        for (auto &thread : dedicatedThreads_) {
            thread->start();
        }
    }
    return nixlUcxEngine::vramApplyCtx();
}

void
nixlUcxThreadPoolEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        std::lock_guard<std::mutex> lock(notifMutex_);
        notifThread_.emplace_back(std::move(remote_name), std::move(msg));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    if (!sharedThread_) {
        progress();
    }

    getNotifsImpl(notif_list);
    std::lock_guard<std::mutex> lock(notifMutex_);
    moveNotifList(notifThread_, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(init_params);
    } else if (init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(init_params);
    } else {
        engine = new nixlUcxEngine(init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = str_split((*custom_params)["device_list"], ", ");

    size_t num_workers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t num_threads = nixl_b_params_get(custom_params, "num_threads", 0);

    if (num_workers <= num_threads) {
        /* There must be at least one shared worker */
        num_workers = num_threads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_PEER;
    } else {
        err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
    }

    uc = std::make_unique<nixlUcxContext>(devs,
                                          sizeof(nixlUcxIntReq),
                                          _internalRequestInit,
                                          _internalRequestFini,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode);

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    auto &uw = uws.front();
    workerAddr = uw->epAddr();
    uw->regAmCallback(NOTIF_STR, notifAmCb, this);

    // Temp fixup
    if (getenv("NIXL_DISABLE_CUDA_ADDR_WA")) {
        NIXL_INFO << "disabling CUDA address workaround";
        cuda_addr_wa = false;
    } else {
        cuda_addr_wa = true;
    }

    m_cudaPrimaryCtx = std::make_shared<nixlUcxCudaDevicePrimaryCtx>();
    vramInitCtx();
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

static std::unordered_map<const nixlUcxEngine *, size_t> &
tlsSharedWorkerMap() {
    static thread_local std::unordered_map<const nixlUcxEngine *, size_t> map;
    return map;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    vramFiniCtx();
    tlsSharedWorkerMap().erase(this);
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return remoteConnMap.count(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }

    return (remoteConnMap.find(remote_agent) == remoteConnMap.end()) ? NIXL_ERR_NOT_FOUND :
                                                                       NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    // thread safety?
    remoteConnMap.erase(search);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    std::vector<char> addr(size);

    if(remoteConnMap.count(remote_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);
    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>();
    bool error = false;
    for (auto &uw: uws) {
        auto result = uw->connect(addr.data(), size);
        if (!result.ok()) {
            error = true;
            break;
        }
        conn->eps.push_back(std::move(*result));
    }

    if (error)
        return NIXL_ERR_BACKEND;

    conn->remoteAgent = remote_agent;

    remoteConnMap.insert({remote_agent, conn});

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    if (nixl_mem == VRAM_SEG) {
        bool need_restart;
        if (vramUpdateCtx((void*)mem.addr, mem.devId, need_restart)) {
            return NIXL_ERR_NOT_SUPPORTED;
            //TODO Add to logging
        }
        if (need_restart) {
            vramApplyCtx();
        }
    }

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}


// To be cleaned up
nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        auto md = std::make_unique<nixlUcxPublicMetadata>();
        size_t size = blob.size();

        auto search = remoteConnMap.find(agent);

        if (search == remoteConnMap.end()) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        md->conn = search->second;

        std::vector<char> addr(size);
        nixlSerDes::_stringToBytes(addr.data(), blob, size);

        for (size_t wid = 0; wid < uws.size(); wid++) {
            md->addRkey(*md->conn->getEp(wid), addr.data());
        }

        output = (nixlBackendMD *)md.release();

        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    // Set CUDA context of first device, UCX will anyways detect proper device when sending
    nixlUcxCudaCtxGuard guard(nixl_mem, m_cudaPrimaryCtx);
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

static nixl_status_t
_retHelper(nixl_status_t ret, nixlUcxBackendH *hndl, nixlUcxReq &req, ucx_connection_ptr_t conn) {
    /* if transfer wasn't immediately completed */
    switch(ret) {
    case NIXL_IN_PROG:
        // TODO: this cast does not look safe
        // We need to allocate a vector of nixlUcxIntReq and set nixlUcxReqt
        hndl->append((nixlUcxIntReq *)req);
        nixlUcxReqSetConnection(req, conn);
    case NIXL_SUCCESS:
        // Nothing to do
        break;
    default:
        // Error. Release all previously initiated ops and exit:
        hndl->release();
        return ret;
    }

    return NIXL_SUCCESS;
}

size_t
nixlUcxEngine::getWorkerId() const {
    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (!opt_args || opt_args->customParam.empty()) {
        return std::nullopt;
    }

    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args->customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args->customParam.substr(pos + worker_id_key.length()));

        if (worker_id >= getSharedWorkersSize()) {
            NIXL_WARN << "Invalid worker_id " << worker_id << " (must be < "
                      << getSharedWorkersSize() << ")";
            return std::nullopt;
        }

        return worker_id;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Failed to parse worker_id from customParam: " << e.what();
        return std::nullopt;
    }
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    /* TODO: try to get from a pool first */
    const auto opt_worker_id = getWorkerIdFromOptArgs(opt_args);
    size_t worker_id = opt_worker_id.value_or(getWorkerId());
    auto *ucx_handle = new nixlUcxBackendH(getWorker(worker_id).get(), worker_id);

    handle = ucx_handle;

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;

        nixlUcxPrivateMetadata *lmd = static_cast<nixlUcxPrivateMetadata*>(local[i].metadataP);
        nixlUcxPublicMetadata *rmd = static_cast<nixlUcxPublicMetadata*>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        nixl_status_t ret = rmd->conn->getEp(workerId)->estimateCost(lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixlUcxPrivateMetadata *lmd;
    nixlUcxPublicMetadata *rmd;
    nixl_status_t ret;
    nixlUcxReq req;
    size_t workerId = intHandle->getWorkerId();

    // Reserve space for the requests, +2 for flush and completion
    intHandle->reserve(end_idx - start_idx + 2);

    for (size_t i = start_idx; i < end_idx; i++) {
        void *laddr = (void*) local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = (uint64_t)remote[i].addr;
        size_t rsize = remote[i].len;

        lmd = (nixlUcxPrivateMetadata*) local[i].metadataP;
        rmd = (nixlUcxPublicMetadata*) remote[i].metadataP;
        auto &ep = rmd->conn->getEp(workerId);

        if (lsize != rsize) {
            return NIXL_ERR_INVALID_PARAM;
        }

        switch (operation) {
        case NIXL_READ:
            ret = ep->read(raddr, rmd->getRkey(workerId), laddr, lmd->mem, lsize, req);
            break;
        case NIXL_WRITE:
            ret = ep->write(laddr, lmd->mem, raddr, rmd->getRkey(workerId), lsize, req);
            break;
        default:
            return NIXL_ERR_INVALID_PARAM;
        }

        if (_retHelper(ret, intHandle, req, rmd->conn)) {
            return ret;
        }
    }

    /*
     * Flush keeps intHandle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     */
    rmd = (nixlUcxPublicMetadata *)remote[0].metadataP;
    ret = rmd->conn->getEp(workerId)->flushEp(req);

    if (_retHelper(ret, intHandle, req, rmd->conn)) {
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();
    nixlUcxBackendH *int_handle = static_cast<nixlUcxBackendH *>(handle);
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: assert that handle is empty/completed, as we can't post request before completion

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = int_handle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            auto rmd = (nixlUcxPublicMetadata *)remote[0].metadataP;
            ret = notifSendPriv(
                remote_agent, opt_args->notifMsg, req, rmd->conn->getEp(int_handle->getWorkerId()));
            if (_retHelper(ret, int_handle, req, rmd->conn)) {
                return ret;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notification().emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    auto& notif = intHandle->notification();
    nixl_status_t handle_status = intHandle->status();

    if ((handle_status != NIXL_SUCCESS) || !notif.has_value()) {
        if (handle_status != NIXL_IN_PROG) { // error flow
            notif.reset();
        }

        return handle_status;
    }

    ucx_connection_ptr_t conn = getConnection(notif->agent);
    if (!conn) {
        notif.reset();
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    nixl_status_t status =
        notifSendPriv(notif->agent, notif->payload, req, conn->getEp(intHandle->getWorkerId()));
    notif.reset();
    status = _retHelper(status, intHandle, req, conn);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    return intHandle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t status = intHandle->release();

    /* TODO: return to a pool instead. */
    delete intHandle;

    return status;
}

nixl_status_t
nixlUcxEngine::createGpuXferReq(const nixlBackendReqH &req_hndl,
                                const nixl_meta_dlist_t &local_descs,
                                const nixl_meta_dlist_t &remote_descs,
                                nixlGpuXferReqH &gpu_req_hndl) const {
    auto intHandle = static_cast<const nixlUcxBackendH *>(&req_hndl);

    if (local_descs.descCount() != remote_descs.descCount()) {
        NIXL_ERROR << "Mismatch between local and remote descriptor counts";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local_descs.descCount() == 0) {
        NIXL_ERROR << "Empty descriptor lists";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto remoteMd = static_cast<nixlUcxPublicMetadata *>(remote_descs[0].metadataP);
    if (!remoteMd || !remoteMd->conn) {
        NIXL_ERROR << "No connection found in remote metadata";
        return NIXL_ERR_NOT_FOUND;
    }

    size_t workerId = intHandle->getWorkerId();
    nixlUcxEp *ep = remoteMd->conn->getEp(workerId).get();

    std::vector<nixlUcxMem> local_mems;
    std::vector<const nixl::ucx::rkey *> remote_rkeys;
    std::vector<uint64_t> remote_addrs;
    local_mems.reserve(local_descs.descCount());
    remote_rkeys.reserve(remote_descs.descCount());
    remote_addrs.reserve(remote_descs.descCount());

    for (size_t i = 0; i < static_cast<size_t>(local_descs.descCount()); i++) {
        auto localMd = static_cast<nixlUcxPrivateMetadata *>(local_descs[i].metadataP);
        auto remoteMdDesc = static_cast<nixlUcxPublicMetadata *>(remote_descs[i].metadataP);

        local_mems.push_back(localMd->mem);
        remote_rkeys.push_back(&remoteMdDesc->getRkey(workerId));
        remote_addrs.push_back(static_cast<uint64_t>(remote_descs[i].addr));
    }

    try {
        gpu_req_hndl = nixl::ucx::createGpuXferReq(*ep, local_mems, remote_rkeys, remote_addrs);
        NIXL_TRACE << "Created device memory list: ep=" << ep->getEp() << " handle=" << gpu_req_hndl
                   << " worker_id=" << workerId << " num_elements=" << local_mems.size();
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create device memory list for GPU transfer: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseGpuXferReq(nixlGpuXferReqH gpu_req_hndl) const {
    nixl::ucx::releaseGpuXferReq(gpu_req_hndl);
}

nixl_status_t
nixlUcxEngine::getGpuSignalSize(size_t &signal_size) const {
    if (gpuSignalSize_) {
        signal_size = *gpuSignalSize_;
        return NIXL_SUCCESS;
    }

    try {
        gpuSignalSize_ = signal_size = uc->getGpuSignalSize();
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepGpuSignal(const nixlBackendMD &meta,
                             void *signal,
                             const nixl_opt_b_args_t *opt_args) const {
    try {
        auto *ucx_meta = static_cast<const nixlUcxPrivateMetadata *>(&meta);

        const auto opt_worker_id = getWorkerIdFromOptArgs(opt_args);
        if (opt_worker_id) {
            getWorker(*opt_worker_id)->prepGpuSignal(ucx_meta->mem, signal);
        } else {
            getWorker(getWorkerId())->prepGpuSignal(ucx_meta->mem, signal);
        }

        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

int nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    int ret = 0;
    for (auto &uw: uws)
        ret += uw->progress();
    return ret;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t
nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                             const std::string &msg,
                             nixlUcxReq &req,
                             const std::unique_ptr<nixlUcxEp> &ep) const {
    nixlSerDes ser_des;
    nixl_status_t ret;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    auto buffer = std::make_unique<std::string>(ser_des.exportStr());
    ret = ep->sendAm(
        NOTIF_STR, NULL, 0, (void *)buffer->data(), buffer->size(), UCP_AM_SEND_FLAG_EAGER, req);
    if (ret == NIXL_IN_PROG) {
        nixlUcxIntReq* nReq = (nixlUcxIntReq*)req;
        nReq->amBuffer = std::move(buffer);
    }
    return ret;
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    auto search = remoteConnMap.find(remote_agent);
    return (search != remoteConnMap.end()) ? search->second : nullptr;
}

void
nixlUcxEngine::appendNotif(std::string remote_name, std::string msg) {
    notifMainList.emplace_back(std::move(remote_name), std::move(msg));
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    // send_am should be forcing EAGER protocol
    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    ser_des.importStr(ser_str);
    std::string remote_name = ser_des.getStr("name");
    std::string msg = ser_des.getStr("msg");

    engine->appendNotif(std::move(remote_name), std::move(msg));
    return UCS_OK;
}

void
nixlUcxEngine::getNotifsImpl(notif_list_t &notif_list) {
    moveNotifList(notifMainList, notif_list);
}

nixl_status_t nixlUcxEngine::getNotifs(notif_list_t &notif_list)
{
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    while (progress())
        ;
    getNotifsImpl(notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const
{
    nixl_status_t ret;
    nixlUcxReq req;

    auto conn = getConnection(remote_agent);
    if (!conn) {
        return NIXL_ERR_NOT_FOUND;
    }

    ret = notifSendPriv(remote_agent, msg, req, conn->getEp(getWorkerId()));
    switch(ret) {
    case NIXL_IN_PROG:
        /* do not track the request */
        getWorker(getWorkerId())->reqRelease(req);
    case NIXL_SUCCESS:
        break;
    default:
        /* error case */
        return ret;
    }
    return NIXL_SUCCESS;
}
