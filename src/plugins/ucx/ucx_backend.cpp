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

#include <optional>
#include <limits>
#include <future>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
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

int nixlUcxCudaCtx::cudaUpdateCtxPtr(void *address, int expected_dev, bool &was_updated)
{
    bool is_dev;
    CUdevice dev;
    CUcontext ctx;
    int ret;

    was_updated = false;

    /* TODO: proper error codes and log outputs through this method */
    if (expected_dev == -1)
        return -1;

    // incorrect dev id from first registration
    if (myDevId != -1 && expected_dev != myDevId)
        return -1;

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

    if (pthrCudaCtx) {
        // Context was already set previously, and does not match new context
        if (pthrCudaCtx != ctx) {
            return -1;
        }
        return 0;
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

int nixlUcxEngine::vramUpdateCtx(void *address, uint64_t  devId, bool &restart_reqd)
{
    int ret;
    bool was_updated;

    restart_reqd = false;

    if(!cuda_addr_wa) {
        // Nothing to do
        return 0;
    }

    ret = cudaCtx->cudaUpdateCtxPtr(address, devId, was_updated);
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


class nixlUcxIntReq : public nixlLinkElem<nixlUcxIntReq> {
    private:
        int _completed;
    public:
        std::unique_ptr<std::string> amBuffer;

        nixlUcxIntReq() : nixlLinkElem() {
            _completed = 0;
        }

        bool is_complete() const { return _completed; }
        void completed() { _completed = 1; }
};

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
    // TODO: use std::vector here for a single allocation and cache friendly
    // traversal
    nixlUcxIntReq head;
    nixlUcxWorker *worker;
    size_t worker_id;
    std::atomic<size_t> *pending_reqs;

    // Notification to be sent after completion of all requests
    struct Notif {
        std::string agent;
        nixl_blob_t payload;

        Notif(const std::string &remote_agent, const nixl_blob_t &msg)
            : agent(remote_agent),
              payload(msg) {}
    };
    std::optional<Notif> notif;

public:
    auto& notification() {
        return notif;
    }

    nixlUcxBackendH(std::atomic<size_t> *pending_reqs_)
        : worker(nullptr),
          worker_id(UINT64_MAX),
          pending_reqs(pending_reqs_) {}

    nixlUcxBackendH(nixlUcxWorker *worker_,
                    size_t worker_id_,
                    std::atomic<size_t> *pending_reqs_ = nullptr)
        : worker(worker_),
          worker_id(worker_id_),
          pending_reqs(pending_reqs_) {}

    void
    init(nixlUcxWorker *worker_, size_t worker_id_) {
        NIXL_ASSERT(worker == nullptr);
        worker = worker_;
        worker_id = worker_id_;
    }

    void
    append(nixlUcxIntReq *req) {
        head.link(req);
    }

    bool
    isComposite() const {
        return pending_reqs != nullptr;
    }

    void
    complete() {
        pending_reqs->fetch_sub(1);
        worker = nullptr;
        worker_id = UINT64_MAX;
    }

    virtual nixl_status_t
    release() {
        nixlUcxIntReq *req = head.next();

        // TODO: Error log: uncompleted requests found! Cancelling ...
        while (req) {
            nixlUcxIntReq *cur = req;
            bool done = cur->is_complete();
            req = cur->unlink();
            if (!done) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker->reqCancel((nixlUcxReq)cur);
            }
            _internalRequestReset(cur);
            worker->reqRelease((nixlUcxReq)cur);
        }
        if (isComposite() && worker) {
            complete();
        }
        return NIXL_SUCCESS;
    }

    virtual nixl_status_t
    status() {
        nixlUcxIntReq *req = head.next();
        nixl_status_t out_ret = NIXL_SUCCESS;

        if (NULL == req) {
            /* No pending transmissions */
            return NIXL_SUCCESS;
        }

        /* Maximum progress */
        while (worker->progress())
            ;

        /* Go over all request updating their status */
        while (req) {
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
                        /* Any other ret value is ERR and will be returned */
                        return ret;
                }
            }
            req = req->next();
        }

        /* Remove completed requests keeping the first one as
        request representative */
        req = head.unlink();
        while (req) {
            nixlUcxIntReq *next_req = req->unlink();
            if (req->is_complete()) {
                _internalRequestReset(req);
                worker->reqRelease((nixlUcxReq)req);
            } else {
                /* Enqueue back */
                append(req);
            }
            req = next_req;
        }

        return out_ret;
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
    nixlUcxThread(const nixlUcxEngine *engine, std::function<void()> init, size_t numWorkers)
        : m_engine(engine),
          m_init(std::move(init)) {
        m_workers.reserve(numWorkers);
    }

    virtual ~nixlUcxThread() {
        if (m_threadActive) {
            join();
        }
    }

    void
    start() {
        NIXL_ASSERT(!m_threadActive);
        m_threadActive = std::make_unique<std::promise<void>>();
        auto active = m_threadActive->get_future();
        m_thread = std::make_unique<std::thread>(std::ref(*this));
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT(m_threadActive);
        m_threadActive.reset();
        m_thread->join();
    }

    virtual void
    addWorker(nixlUcxWorker *worker, size_t workerId) {
        NIXL_ASSERT(m_workers.size() < m_workers.capacity());
        m_workers.push_back(worker);
        m_workerIds.push_back(workerId);
    }

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return m_workers;
    }

    size_t
    getWorkerId(size_t idx = 0) const {
        return m_workerIds[idx];
    }

    void
    operator()() {
        tlsThread() = this;
        m_init();
        m_threadActive->set_value();
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
        return thread && thread->m_engine == engine;
    }

protected:
    virtual void
    run() = 0;

private:
    const nixlUcxEngine *m_engine;
    std::function<void()> m_init;
    std::vector<nixlUcxWorker *> m_workers;
    std::vector<size_t> m_workerIds;
    std::unique_ptr<std::thread> m_thread;
    std::unique_ptr<std::promise<void>> m_threadActive;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine,
                        std::function<void()> init,
                        size_t numWorkers,
                        nixlTime::us_t delay)
        : nixlUcxThread(engine, std::move(init), numWorkers) {
        if (pipe(m_controlPipe) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        m_delay = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(
            delay < std::numeric_limits<int>::max() ? delay : std::numeric_limits<int>::max()));

        m_pollFds.resize(numWorkers + 1);
        m_pollFds.back() = {m_controlPipe[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(m_controlPipe[0]);
        close(m_controlPipe[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(m_controlPipe[1], &signal, sizeof(signal));
        if (ret < 0) NIXL_PERROR << "write to progress thread control pipe failed";
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker, size_t workerId) override {
        m_pollFds[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker, workerId);
    }

protected:
    virtual void
    run() {
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthrStop = false;
        while (!pthrStop) {
            for (size_t i = 0; i < m_pollFds.size() - 1; i++) {
                if (!(m_pollFds[i].revents & POLLIN) && !timeout) continue;
                m_pollFds[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    while (worker->progress())
                        ;
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(m_pollFds.data(), m_pollFds.size(), m_delay.count())) < 0)
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";

            if (!ret) {
                timeout = true;
            } else if (m_pollFds.back().revents & POLLIN) {
                m_pollFds.back().revents = 0;

                char signal;
                int ret = read(m_pollFds.back().fd, &signal, sizeof(signal));
                if (ret < 0) NIXL_PERROR << "read() on control pipe failed";

                pthrStop = true;
            }
        }
    }

private:
    std::chrono::milliseconds m_delay;
    int m_controlPipe[2];
    std::vector<pollfd> m_pollFds;
};

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl_ucx_mt_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t numWorkers = getWorkers().size();
    thread = std::make_unique<nixlUcxSharedThread>(
        this, [this]() { nixlUcxEngine::vramApplyCtx(); }, numWorkers, init_params.pthrDelay);
    for (size_t i = 0; i < numWorkers; i++) {
        thread->addWorker(getWorkers()[i].get(), i);
    }
    thread->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    thread->join();
}

int
nixlUcxThreadEngine::vramApplyCtx() {
    thread->join();
    thread->start();
    return nixlUcxEngine::vramApplyCtx();
}

void
nixlUcxThreadEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        /* Append to the private list to allow batching */
        const std::lock_guard<std::mutex> lock(notifMtx);
        notifPthr.push_back(std::make_pair(std::move(remote_name), std::move(msg)));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    getNotifsImpl(notif_list);
    const std::lock_guard<std::mutex> lock(notifMtx);
    moveNotifList(notifPthr, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Threadpool engine
 ****************************************/

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendH : public nixlUcxBackendH {
public:
    nixlUcxCompositeBackendH(nixlUcxWorker *worker,
                             size_t workerId,
                             size_t chunkSize,
                             size_t numChunks)
        : nixlUcxBackendH(worker, workerId, &m_pendingReqs),
          m_chunkSize(chunkSize),
          m_pendingReqs(0) {
        m_chunks.reserve(numChunks);
        for (size_t i = 0; i < numChunks; i++) {
            m_chunks.emplace_back(&m_pendingReqs);
        }
    }

    size_t
    getChunkSize() const {
        return m_chunkSize;
    }

    size_t
    getNumChunks() const {
        return m_chunks.size();
    }

    void
    startXfer() {
        NIXL_ASSERT(m_pendingReqs.load() == 0);
        m_pendingReqs.store(m_chunks.size());
    }

    nixlUcxBackendH *
    getChunk(size_t idx) {
        return &m_chunks[idx];
    }

    nixl_status_t
    release() override {
        for (auto &chunk : m_chunks) {
            chunk.release();
        }
        m_chunks.clear();
        return NIXL_SUCCESS;
    }

    nixl_status_t
    status() override {
        while (getWorker()->progress())
            ;

        if (m_pendingReqs.load()) {
            return NIXL_IN_PROG;
        }
        return nixlUcxBackendH::status();
    }

private:
    std::vector<nixlUcxBackendH> m_chunks;
    size_t m_chunkSize;
    std::atomic<size_t> m_pendingReqs;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, std::function<void()> init, asio::io_context &io)
        : nixlUcxThread(engine, std::move(init), 1),
          m_io(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxBackendH *handle) {
        m_requests.push_back(handle);
    }

protected:
    void
    run() override {
        auto guard = asio::make_work_guard(m_io);

        while (!m_io.stopped()) {
            if (!m_requests.empty()) {
                m_io.poll_one();
            } else {
                m_io.run_one();
            }

            if (m_requests.empty()) {
                continue;
            }

            for (auto it = m_requests.begin(); it != m_requests.end();) {
                if ((*it)->status() == NIXL_SUCCESS) {
                    (*it)->complete();
                    it = m_requests.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

private:
    asio::io_context &m_io;
    std::vector<nixlUcxBackendH *> m_requests;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t numThreads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    m_numSharedWorkers = getWorkers().size() - numThreads;
    NIXL_ASSERT(m_numSharedWorkers > 0);

    auto init = [this]() { nixlUcxEngine::vramApplyCtx(); };

    if (init_params.enableProgTh) {
        m_sharedThread = std::make_unique<nixlUcxSharedThread>(
            this, init, m_numSharedWorkers, init_params.pthrDelay);
        for (size_t i = 0; i < m_numSharedWorkers; i++) {
            m_sharedThread->addWorker(getWorkers()[i].get(), i);
        }
        m_sharedThread->start();
    }

    if (numThreads > 0) {
        m_io.reset(new asio::io_context());
        m_dedicatedThreads.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i) {
            size_t workerId = m_numSharedWorkers + i;
            m_dedicatedThreads.emplace_back(
                std::make_unique<nixlUcxDedicatedThread>(this, init, *m_io));
            m_dedicatedThreads.back()->addWorker(getWorker(workerId).get(), workerId);
            m_dedicatedThreads.back()->start();
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    if (m_sharedThread) {
        m_sharedThread->join();
    }

    if (m_io) {
        m_io->stop();
        for (auto &thread : m_dedicatedThreads) {
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
    // TODO: find the best split strategy based on batchSize and numThreads
    // Maybe make it configurable
    const char *minEnv = getenv("NIXL_MIN_CHUNK_SIZE");
    const char *maxEnv = getenv("NIXL_MAX_CHUNK_SIZE");
    size_t MIN_CHUNK_SIZE = (minEnv != NULL) ? atoi(minEnv) : 1024;
    size_t MAX_CHUNK_SIZE = (maxEnv != NULL) ? atoi(maxEnv) : 8192;

    size_t batchSize = local.descCount();
    if (batchSize <= MIN_CHUNK_SIZE) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunkSize =
        std::clamp(batchSize / m_dedicatedThreads.size(), MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
    size_t numChunks = (batchSize + chunkSize - 1) / chunkSize;

    size_t workerId = getWorkerId();
    handle =
        new nixlUcxCompositeBackendH(getWorker(workerId).get(), workerId, chunkSize, numChunks);
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
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    if (!intHandle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    nixlUcxCompositeBackendH *compHandle = (nixlUcxCompositeBackendH *)intHandle;
    compHandle->startXfer();
    size_t chunkSize = compHandle->getChunkSize();

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{compHandle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < compHandle->getNumChunks(); i++) {
        m_io->post([&, i]() {
            auto thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxBackendH *chunkHandle = compHandle->getChunk(i);
            chunkHandle->init(thread->getWorkers()[0], thread->getWorkerId());

            size_t startIdx = i * chunkSize;
            size_t endIdx = std::min(startIdx + chunkSize, (size_t)local.descCount());
            nixl_status_t ret;
            ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunkHandle, startIdx, endIdx);
            if (ret != NIXL_SUCCESS) {
                // TODO: test error handling
                status.store(ret);
                chunkHandle->release();
            } else {
                thread->addRequest(chunkHandle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    return status.load();
}

int
nixlUcxThreadPoolEngine::vramApplyCtx() {
    if (m_sharedThread) {
        m_sharedThread->join();
        m_sharedThread->start();
    }
    if (m_io) {
        m_io->stop();
        for (auto &thread : m_dedicatedThreads) {
            thread->join();
        }
        m_io->restart();
        for (auto &thread : m_dedicatedThreads) {
            thread->start();
        }
    }
    return nixlUcxEngine::vramApplyCtx();
}

void
nixlUcxThreadPoolEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        std::lock_guard<std::mutex> lock(m_notifMutex);
        m_notifThread.emplace_back(std::move(remote_name), std::move(msg));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    if (!m_sharedThread) {
        progress();
    }

    getNotifsImpl(notif_list);
    std::lock_guard<std::mutex> lock(m_notifMutex);
    moveNotifList(m_notifThread, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    size_t numThreads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    if (numThreads > 0) {
        engine = new nixlUcxThreadPoolEngine(init_params);
    } else if (init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(init_params);
    } else {
        engine = new nixlUcxEngine(init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params) {
    size_t numWorkers;
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = str_split((*custom_params)["device_list"], ", ");

    numWorkers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t numThreads = nixl_b_params_get(custom_params, "num_threads", 0);

    if (numWorkers <= numThreads) {
        /* There must be at least one shared worker */
        numWorkers = numThreads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_NONE;
    } else {
        err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
    }

    uc = std::make_unique<nixlUcxContext>(devs,
                                          sizeof(nixlUcxIntReq),
                                          _internalRequestInit,
                                          _internalRequestFini,
                                          init_params.enableProgTh,
                                          numWorkers,
                                          init_params.syncMode);

    for (size_t i = 0; i < numWorkers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    workerAddr = uws.front()->epAddr();

    // TODO: in case of UCX error handling is enabled, we can clean up AM based connections error
    //       handling, if user requested disabled error handling, we dont care about it.
    auto &uw = uws.front();
    uw->regAmCallback(CONN_CHECK, connectionCheckAmCb, this);
    uw->regAmCallback(DISCONNECT, connectionTermAmCb, this);
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

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    vramFiniCtx();
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return remoteConnMap.count(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::endConn(const std::string &remote_agent) {

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    //thread safety?
    remoteConnMap.erase(search);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

ucs_status_t
nixlUcxEngine::connectionCheckAmCb(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    std::string remote_agent( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    if(engine->checkConn(remote_agent)) {
        NIXL_ERROR << "Received connect AM from agent we don't recognize: " << remote_agent;
        return UCS_OK;
    }

    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::connectionTermAmCb (void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
{
    std::string remote_agent( (char*) data, length);

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

/*
    // TODO: research UCX connection logic and fix.
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;
    if(NIXL_SUCCESS != engine->endConn(remote_agent)) {
        //TODO: received connect AM from agent we don't recognize
        return UCS_ERR_INVALID_PARAM;
    }
*/
    return UCS_OK;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }
    const auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    bool error = false;
    nixl_status_t ret = NIXL_SUCCESS;
    std::vector<nixlUcxReq> reqs;
    for (size_t i = 0; i < uws.size(); i++) {
        reqs.emplace_back();
        ret = search->second->getEp(i)->sendAm(CONN_CHECK, NULL, 0,
                                               (void*) localAgent.data(), localAgent.size(),
                                               UCP_AM_SEND_FLAG_EAGER, reqs.back());
        if(ret < 0) {
            error = true;
            break;
        }
    }

    //wait for AM to send
    ret = NIXL_IN_PROG;
    for (size_t i = 0; i < reqs.size(); i++)
        while(ret == NIXL_IN_PROG)
            ret = getWorker(i)->test(reqs[i]);

    return error ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    if (remote_agent != localAgent) {
        auto search = remoteConnMap.find(remote_agent);

        if(search == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixl_status_t ret = NIXL_SUCCESS;
        for (size_t i = 0; i < uws.size(); i++) {
            if (search->second->getEp(i)->checkTxState() == NIXL_SUCCESS) {
                nixlUcxReq req;
                ret = search->second->getEp(i)->sendAm(DISCONNECT, NULL, 0,
                                                       (void*) localAgent.data(), localAgent.size(),
                                                       UCP_AM_SEND_FLAG_EAGER, req);
                //don't care
                if (ret == NIXL_IN_PROG)
                    getWorker(i)->reqRelease(req);
            }
        }
    }

    endConn(remote_agent);

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

static nixl_status_t _retHelper(nixl_status_t ret,  nixlUcxBackendH *hndl, nixlUcxReq &req)
{
    /* if transfer wasn't immediately completed */
    switch(ret) {
        case NIXL_IN_PROG:
            // TODO: this cast does not look safe
            // We need to allocate a vector of nixlUcxIntReq and set nixlUcxReqt
            hndl->append((nixlUcxIntReq*)req);
        case NIXL_SUCCESS:
            // Nothing to do
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            hndl->release();
            return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
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
    size_t workerId = getWorkerId();
    handle = new nixlUcxBackendH(getWorker(workerId).get(), workerId);
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

        if (_retHelper(ret, intHandle, req)) {
            return ret;
        }
    }

    /*
     * Flush keeps intHandle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     * TODO: should we flush all distinct endpoints?
     */
    rmd = (nixlUcxPublicMetadata*) remote[0].metadataP;
    ret = rmd->conn->getEp(workerId)->flushEp(req);
    if (_retHelper(ret, intHandle, req)) {
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
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
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

    ret = intHandle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            ret = notifSendPriv(remote_agent, opt_args->notifMsg, req, intHandle->getWorkerId());
            if (_retHelper(ret, intHandle, req)) {
                return ret;
            }

            ret = intHandle->status();
        } else if (ret == NIXL_IN_PROG) {
            intHandle->notification().emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();

    nixl_status_t status = intHandle->status();
    auto& notif = intHandle->notification();
    if (status == NIXL_SUCCESS && notif.has_value()) {
        nixlUcxReq req;
        status = notifSendPriv(notif->agent, notif->payload, req, workerId);
        notif.reset();
        if (_retHelper(status, intHandle, req)) {
            return status;
        }

        status = intHandle->status();
    }

    return status;
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t status = intHandle->release();

    /* TODO: return to a pool instead. */
    delete intHandle;

    return status;
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
nixl_status_t nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                                           const std::string &msg,
                                           nixlUcxReq &req,
                                           size_t worker_id) const
{
    nixlSerDes ser_des;
    nixl_status_t ret;

    auto search = remoteConnMap.find(remote_agent);

    if(search == remoteConnMap.end()) {
        //TODO: err: remote connection not found
        return NIXL_ERR_NOT_FOUND;
    }

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    auto buffer = std::make_unique<std::string>(std::move(ser_des.exportStr()));
    ret = search->second->getEp(worker_id)->sendAm(NOTIF_STR, NULL, 0,
                                                   (void*)buffer->data(), buffer->size(),
                                                   UCP_AM_SEND_FLAG_EAGER, req);

    if (ret == NIXL_IN_PROG) {
        nixlUcxIntReq* nReq = (nixlUcxIntReq*)req;
        nReq->amBuffer = std::move(buffer);
    }
    return ret;
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
    size_t wid = getWorkerId();

    ret = notifSendPriv(remote_agent, msg, req, wid);

    switch(ret) {
    case NIXL_IN_PROG:
        /* do not track the request */
        getWorker(wid)->reqRelease(req);
    case NIXL_SUCCESS:
        break;
    default:
        /* error case */
        return ret;
    }
    return NIXL_SUCCESS;
}
