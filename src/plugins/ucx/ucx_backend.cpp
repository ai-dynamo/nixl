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

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <optional>
#include <limits>
#include <future>
#include <set>
#include <string_view>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <asio.hpp>

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendReqH : public nixlBackendReqH {
private:
    std::set<ucx_connection_ptr_t> connections_;
    std::vector<nixlUcxReq> requests_;
    nixlUcxWorker *worker_;
    size_t workerId_;

    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(!connections_.empty());
        for (const auto &conn : connections_) {
            const nixl_status_t conn_status = conn->getEp(workerId_)->checkTxState();
            if (conn_status != NIXL_SUCCESS) {
                return conn_status;
            }
        }
        return status;
    }

protected:
    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(worker_ == nullptr || worker == nullptr);
        worker_ = worker;
        workerId_ = worker_id;
    }

public:
    // Notification to be sent after completion of all requests
    struct Notif {
        const std::string agent;
        const nixl_blob_t payload;

        Notif(const std::string &remote_agent, const nixl_blob_t &msg)
            : agent(remote_agent),
              payload(msg) {}
    };

    std::optional<Notif> notif;

    nixlUcxBackendReqH(nixlUcxWorker *worker, size_t worker_id)
        : worker_(worker),
          workerId_(worker_id) {}

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(connections_.empty());
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req, const ucx_connection_ptr_t &conn) {
        switch (status) {
        case NIXL_IN_PROG:
            requests_.push_back(req);
            connections_.insert(conn);
            break;
        case NIXL_SUCCESS:
            connections_.insert(conn);
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] const std::set<ucx_connection_ptr_t> &
    getConnections() const noexcept {
        return connections_;
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        // TODO: Error log: uncompleted requests found! Cancelling ...
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker_->reqCancel(req);
            }
            worker_->reqRelease(req);
        }
        requests_.clear();
        connections_.clear();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            connections_.clear();
            return NIXL_SUCCESS;
        }

        worker_->progressLoop();

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        nixlUcxReq req = requests_.back();
        const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        } else if (ret != NIXL_SUCCESS) {
            return checkConnection(ret);
        }

        /* Last request completed successfully, all the others must be in the
         * same state. TODO: remove extra checks? */
        size_t incomplete_reqs = 0;
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (__builtin_expect(ret == NIXL_SUCCESS, 0)) {
                worker_->reqRelease(req);
            } else if (ret == NIXL_IN_PROG) {
                if (out_ret == NIXL_SUCCESS) {
                    out_ret = NIXL_IN_PROG;
                }
                requests_[incomplete_reqs++] = req;
            } else {
                // Any other ret value is ERR and will be returned
                out_ret = checkConnection(ret);
            }
        }

        requests_.resize(incomplete_reqs);
        if (requests_.empty()) {
            connections_.clear();
        }
        return out_ret;
    }

    [[nodiscard]] nixlUcxWorker *
    getWorker() const noexcept {
        return worker_;
    }

    [[nodiscard]] size_t
    getWorkerId() const noexcept {
        return workerId_;
    }
};

/****************************************
 * Staged VRAM metadata
*****************************************/

namespace {

constexpr std::string_view kUcxStagedMagic = "NIXL_UCX_STAGED_V1";

struct nixlUcxStagedSlot {
    void *hostAddr = nullptr;
    size_t size = 0;
    nixlUcxMem mem;
    nixl_blob_t rkeyStr;
};

class nixlUcxStagedPrivateMetadata : public nixlBackendMD {
public:
    explicit nixlUcxStagedPrivateMetadata(const nixlBlobDesc &mem)
        : nixlBackendMD(true),
          gpuBase(mem.addr),
          gpuLen(mem.len),
          gpuDevId(mem.devId) {}

    uintptr_t gpuBase;
    size_t gpuLen;
    uint64_t gpuDevId;
    size_t slotSize = 0;
    std::vector<nixlUcxStagedSlot> slots;
    std::mutex slotMutex;
    std::vector<bool> slotBusy;

    [[nodiscard]] std::optional<size_t>
    acquireSlot() {
        const std::lock_guard lock(slotMutex);
        for (size_t i = 0; i < slotBusy.size(); ++i) {
            if (!slotBusy[i]) {
                slotBusy[i] = true;
                return i;
            }
        }
        return std::nullopt;
    }

    void
    releaseSlot(size_t slot_id) {
        const std::lock_guard lock(slotMutex);
        if (slot_id < slotBusy.size()) {
            slotBusy[slot_id] = false;
        }
    }
};

class nixlUcxStagedPublicMetadata : public nixlBackendMD {
public:
    nixlUcxStagedPublicMetadata(const ucx_connection_ptr_t &conn,
                                uintptr_t gpu_base,
                                size_t gpu_len,
                                uint64_t gpu_dev_id,
                                size_t slot_size,
                                std::vector<uintptr_t> &&slot_addrs,
                                std::vector<std::vector<nixl::ucx::rkey>> &&slot_rkeys)
        : nixlBackendMD(false),
          conn(conn),
          gpuBase(gpu_base),
          gpuLen(gpu_len),
          gpuDevId(gpu_dev_id),
          slotSize(slot_size),
          slotAddrs(std::move(slot_addrs)),
          slotRkeys(std::move(slot_rkeys)) {}

    const ucx_connection_ptr_t conn;
    const uintptr_t gpuBase;
    const size_t gpuLen;
    const uint64_t gpuDevId;
    const size_t slotSize;
    const std::vector<uintptr_t> slotAddrs;
    const std::vector<std::vector<nixl::ucx::rkey>> slotRkeys;
};

[[nodiscard]] nixl_blob_t
serializeStagedMetadata(const nixlUcxStagedPrivateMetadata &metadata) {
    nixlSerDes ser_des;
    const std::string magic(kUcxStagedMagic);
    const size_t slot_count = metadata.slots.size();

    ser_des.addStr("magic", magic);
    ser_des.addBuf("gpu_base", &metadata.gpuBase, sizeof(metadata.gpuBase));
    ser_des.addBuf("gpu_len", &metadata.gpuLen, sizeof(metadata.gpuLen));
    ser_des.addBuf("gpu_dev", &metadata.gpuDevId, sizeof(metadata.gpuDevId));
    ser_des.addBuf("slot_size", &metadata.slotSize, sizeof(metadata.slotSize));
    ser_des.addBuf("slot_count", &slot_count, sizeof(slot_count));

    for (const auto &slot : metadata.slots) {
        const uintptr_t host_addr = reinterpret_cast<uintptr_t>(slot.hostAddr);
        ser_des.addBuf("slot_addr", &host_addr, sizeof(host_addr));
        ser_des.addStr("slot_rkey", slot.rkeyStr);
    }

    return ser_des.exportStr();
}

[[nodiscard]] bool
isStagedMetadataBlob(const nixl_blob_t &blob) {
    return blob.compare(0, std::string_view("nixlSerDes|").size(), "nixlSerDes|") == 0;
}

[[nodiscard]] bool
rangeCovers(uintptr_t base, size_t len, uintptr_t addr, size_t size) {
    return addr >= base && size <= len && (addr - base) <= (len - size);
}

[[nodiscard]] nixl_status_t
cudaSetDeviceForCopy(uint64_t dev_id, int &previous_device) {
    previous_device = -1;
    cudaError_t cuda_ret = cudaGetDevice(&previous_device);
    if (cuda_ret != cudaSuccess) {
        NIXL_ERROR << "cudaGetDevice failed: " << cudaGetErrorString(cuda_ret);
        return NIXL_ERR_BACKEND;
    }

    cuda_ret = cudaSetDevice(static_cast<int>(dev_id));
    if (cuda_ret != cudaSuccess) {
        NIXL_ERROR << "cudaSetDevice(" << dev_id << ") failed: " << cudaGetErrorString(cuda_ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
cudaRestoreDevice(int previous_device) {
    if (previous_device < 0) {
        return;
    }

    const cudaError_t cuda_ret = cudaSetDevice(previous_device);
    if (cuda_ret != cudaSuccess) {
        NIXL_WARN << "cudaSetDevice restore to " << previous_device
                  << " failed: " << cudaGetErrorString(cuda_ret);
    }
}

struct nixlUcxStagedChunk {
    enum class State {
        PENDING,
        RDMA_POSTED,
        READY_AM_POSTED,
        WAIT_ACK,
        ACKED,
        FAILED,
    };

    nixlUcxStagedChunk(uint64_t chunk_id, size_t chunk_offset, size_t chunk_size)
        : id(chunk_id),
          offset(chunk_offset),
          size(chunk_size) {}

    const uint64_t id;
    const size_t offset;
    const size_t size;
    size_t localSlotId = 0;
    bool localSlotHeld = false;
    uint64_t remoteSlotId = 0;
    bool remoteSlotHeld = false;
    State state = State::PENDING;
    std::unique_ptr<nixlUcxBackendReqH> req;
    std::atomic<nixl_status_t> ackStatus{NIXL_IN_PROG};
};

class nixlUcxStagedBackendReqH : public nixlUcxBackendReqH {
public:
    enum class State {
        INIT,
        RUNNING,
        USER_NOTIF_POSTED,
        COMPLETE,
        FAILED,
    };

    nixlUcxStagedBackendReqH(nixlUcxWorker *worker, size_t worker_id, uint64_t transfer_id)
        : nixlUcxBackendReqH(worker, worker_id),
          transferId(transfer_id) {}

    void
    releaseLocalSlot(nixlUcxStagedChunk &chunk) {
        if (localMetadata && chunk.localSlotHeld) {
            localMetadata->releaseSlot(chunk.localSlotId);
            chunk.localSlotHeld = false;
        }
    }

    void
    releaseRemoteSlot(nixlUcxStagedChunk &chunk) {
        if (chunk.remoteSlotHeld && chunk.remoteSlotId < remoteSlotBusy.size()) {
            remoteSlotBusy[chunk.remoteSlotId] = false;
            chunk.remoteSlotHeld = false;
        }
    }

    void
    releaseChunkSlots(nixlUcxStagedChunk &chunk) {
        releaseLocalSlot(chunk);
        releaseRemoteSlot(chunk);
    }

    void
    releaseAllChunkSlots() {
        for (const auto &chunk : chunks) {
            releaseChunkSlots(*chunk);
        }
    }

    [[nodiscard]] std::optional<size_t>
    acquireRemoteSlot() {
        for (size_t i = 0; i < remoteSlotBusy.size(); ++i) {
            if (!remoteSlotBusy[i]) {
                remoteSlotBusy[i] = true;
                return i;
            }
        }
        return std::nullopt;
    }

    void
    markAck(uint64_t chunk_id, nixl_status_t status) {
        if (chunk_id >= chunks.size()) {
            NIXL_WARN << "Received UCX staged ACK for out-of-range chunk id " << chunk_id
                      << " transfer_id=" << transferId << " chunks=" << chunks.size();
            return;
        }
        chunks[chunk_id]->ackStatus.store(status);
    }

    void
    release() override {
        releaseAllChunkSlots();
        for (const auto &chunk : chunks) {
            if (chunk->req) {
                chunk->req->release();
            }
        }
        nixlUcxBackendReqH::release();
    }

    const uint64_t transferId;
    std::string remoteAgent;
    nixlUcxStagedPrivateMetadata *localMetadata = nullptr;
    nixlUcxStagedPublicMetadata *remoteMetadata = nullptr;
    uintptr_t localGpuAddr = 0;
    uint64_t localGpuDev = 0;
    uintptr_t remoteGpuAddr = 0;
    uint64_t remoteGpuDev = 0;
    size_t totalSize = 0;
    size_t chunkSize = 0;
    size_t nextChunkToPost = 0;
    size_t completedChunks = 0;
    std::vector<bool> remoteSlotBusy;
    std::vector<std::unique_ptr<nixlUcxStagedChunk>> chunks;
    bool pendingRegistered = false;
    State state = State::INIT;
    nixl_status_t lastStatus = NIXL_IN_PROG;
};

} // namespace

/****************************************
 * Progress thread management
*****************************************/

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, size_t num_workers) : engine_(engine) {
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
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<std::promise<void>> threadActive_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
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
                    worker->progressLoop();
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
    if (!nixlUcxMtLevelIsSupported(nixl::ucx::mt_mode_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t num_workers = getWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(this, num_workers, init_params.pthrDelay);
    for (size_t i = 0; i < num_workers; i++) {
        thread_->addWorker(getWorkers()[i].get(), i);
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    thread_->join();
}

void
nixlUcxThreadEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
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
class nixlUcxChunkBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxChunkBackendReqH() : nixlUcxBackendReqH(nullptr, UINT64_MAX) {}

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

    [[nodiscard]] nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendReqH &chunk) {
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
    setWorker(nullptr, UINT64_MAX);
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
    nixlUcxCompositeBackendReqH(nixlUcxWorker *worker,
                                size_t worker_id,
                                size_t chunk_size,
                                size_t num_chunks)
        : nixlUcxBackendReqH(worker, worker_id),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.resize(num_chunks);
    }

    [[nodiscard]] size_t
    getChunkSize() const noexcept {
        return chunkSize_;
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
    startChunk(size_t idx, nixlUcxWorker *worker, size_t worker_id) {
        nixlUcxChunkBackendReqH *chunk = &sharedState_->chunks[idx];
        chunk->startXfer(sharedState_, worker, worker_id);
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
    size_t chunkSize_;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, asio::io_context &io)
        : nixlUcxThread(engine, 1),
          io_(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendReqH *handle) {
        requests_.push_back(handle);
    }

protected:
    void
    run() override {
        const auto guard = asio::make_work_guard(io_);
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
    std::vector<nixlUcxChunkBackendReqH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    numSharedWorkers_ = getWorkers().size() - num_threads;
    NIXL_ASSERT(numSharedWorkers_ > 0);

    splitBatchSize_ = nixl_b_params_get(init_params.customParams, "split_batch_size", 1024);

    if (init_params.enableProgTh) {
        sharedThread_ =
            std::make_unique<nixlUcxSharedThread>(this, numSharedWorkers_, init_params.pthrDelay);
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
            dedicatedThreads_.emplace_back(std::make_unique<nixlUcxDedicatedThread>(this, *io_));
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
    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t batch_size = local.descCount();
    if (batch_size < splitBatchSize_) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunk_size = std::max(batch_size / dedicatedThreads_.size(), splitBatchSize_);
    size_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    size_t worker_id = getWorkerId();
    const auto comp_handle = new nixlUcxCompositeBackendReqH(
        getWorker(worker_id).get(), worker_id, chunk_size, num_chunks);
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
    size_t chunk_size = comp_handle->getChunkSize();
    NIXL_TRACE << "sending " << *comp_handle;

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{comp_handle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < comp_handle->getNumChunks(); i++) {
        io_->post([&, i]() {
            nixlUcxDedicatedThread *thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxChunkBackendReqH *chunk_handle =
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

void
nixlUcxThreadPoolEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!sharedThread_) {
        progressLoop();
    }

    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    nixlBackendInitParams effective_init_params = init_params;
    const auto staging_config = makeVramStagingConfig(init_params.customParams);
    if (staging_config.enabled && staging_config.forceProgressThread &&
        !effective_init_params.enableProgTh) {
        NIXL_INFO << "UCX VRAM staging requested progress thread; enabling UCX progress thread";
        effective_init_params.enableProgTh = true;
    }

    size_t num_threads = nixl_b_params_get(effective_init_params.customParams, "num_threads", 0);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(effective_init_params);
    } else if (effective_init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(effective_init_params);
    } else {
        engine = new nixlUcxEngine(effective_init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::VramStagingConfig
nixlUcxEngine::makeVramStagingConfig(const nixl_b_params_t *custom_params) {
    VramStagingConfig config;
    config.enabled = nixl_b_params_get_bool(
        custom_params, nixl_ucx_vram_staging_param_name, config.enabled);
    config.chunkSize = nixl_b_params_get_size(
        custom_params, nixl_ucx_staging_chunk_size_param_name, config.chunkSize);
    config.slotsPerGpu =
        nixl_b_params_get_size(custom_params, nixl_ucx_staging_slots_param_name, config.slotsPerGpu);
    config.forceProgressThread = nixl_b_params_get_bool(custom_params,
                                                        nixl_ucx_staging_force_progress_param_name,
                                                        config.forceProgressThread);
    config.cudaCopyStreams = nixl_b_params_get_size(custom_params,
                                                    nixl_ucx_staging_cuda_streams_param_name,
                                                    config.cudaCopyStreams);
    config.enabled = nixl_env_get_bool(nixl_ucx_vram_staging_env_name, config.enabled);
    config.chunkSize = nixl_env_get_size(nixl_ucx_staging_chunk_size_env_name, config.chunkSize);
    config.slotsPerGpu = nixl_env_get_size(nixl_ucx_staging_slots_env_name, config.slotsPerGpu);
    config.forceProgressThread =
        nixl_env_get_bool(nixl_ucx_staging_force_progress_env_name, config.forceProgressThread);
    config.cudaCopyStreams =
        nixl_env_get_size(nixl_ucx_staging_cuda_streams_env_name, config.cudaCopyStreams);
    return config;
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1),
      vramStagingConfig_(makeVramStagingConfig(init_params.customParams)),
      nextStagedTransferId_(1) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = absl::StrSplit((*custom_params)["device_list"], ", ");

    size_t num_workers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t num_threads = nixl_b_params_get(custom_params, "num_threads", 0);
    size_t num_device_channels = nixl_b_params_get(custom_params, "ucx_num_device_channels", 4);

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

    const auto engine_config_it = custom_params->find("engine_config");
    const auto engine_config =
        (engine_config_it != custom_params->end()) ? engine_config_it->second : "";

    uc = std::make_unique<nixlUcxContext>(devs,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config);

    uc->warnAboutHardwareSupportMismatch();

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    auto &uw = uws.front();
    workerAddr = uw->epAddr();
    uw->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR, notifAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_WRITE_READY, stagedWriteReadyAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_ACK, stagedAckAmCb, this);

    if (vramStagingConfig_.enabled) {
        NIXL_INFO << "UCX VRAM staging enabled: chunk_size=" << vramStagingConfig_.chunkSize
                  << " slots_per_gpu=" << vramStagingConfig_.slotsPerGpu
                  << " cuda_copy_streams=" << vramStagingConfig_.cudaCopyStreams
                  << " force_progress_thread=" << vramStagingConfig_.forceProgressThread;
    }
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
    const auto it = remoteConnMap.find(remote_agent);

    if (it == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    // thread safety?
    remoteConnMap.erase(it);
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
    for (auto &uw : uws) {
        std::unique_ptr<nixlUcxEp> result = uw->connect(addr.data(), size);
        if (!result) {
            return NIXL_ERR_BACKEND;
        }
        conn->eps.push_back(std::move(result));
    }

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
    if (vramStagingEnabled() && nixl_mem == VRAM_SEG) {
        if (mem.len == 0 || vramStagingConfig_.chunkSize == 0 ||
            vramStagingConfig_.slotsPerGpu == 0) {
            NIXL_ERROR << "Invalid UCX VRAM staging configuration: chunk_size="
                       << vramStagingConfig_.chunkSize
                       << " slots_per_gpu=" << vramStagingConfig_.slotsPerGpu
                       << " mem_len=" << mem.len;
            return NIXL_ERR_INVALID_PARAM;
        }

        auto staged = std::make_unique<nixlUcxStagedPrivateMetadata>(mem);
        staged->slotSize = std::min(vramStagingConfig_.chunkSize, mem.len);
        staged->slots.resize(vramStagingConfig_.slotsPerGpu);
        staged->slotBusy.assign(vramStagingConfig_.slotsPerGpu, false);

        for (size_t i = 0; i < staged->slots.size(); ++i) {
            nixlUcxStagedSlot &slot = staged->slots[i];
            slot.size = staged->slotSize;

            const cudaError_t alloc_ret = cudaMallocHost(&slot.hostAddr, slot.size);
            if (alloc_ret != cudaSuccess) {
                NIXL_ERROR << "cudaMallocHost failed for UCX VRAM staging slot " << i << ": "
                           << cudaGetErrorString(alloc_ret);
                for (size_t j = 0; j < i; ++j) {
                    uc->memDereg(staged->slots[j].mem);
                    cudaFreeHost(staged->slots[j].hostAddr);
                }
                return NIXL_ERR_BACKEND;
            }

            const int reg_ret = uc->memReg(slot.hostAddr, slot.size, slot.mem, DRAM_SEG);
            if (reg_ret) {
                NIXL_ERROR << "UCX host staging memory registration failed for slot " << i;
                cudaFreeHost(slot.hostAddr);
                for (size_t j = 0; j < i; ++j) {
                    uc->memDereg(staged->slots[j].mem);
                    cudaFreeHost(staged->slots[j].hostAddr);
                }
                return NIXL_ERR_BACKEND;
            }

            slot.rkeyStr = uc->packRkey(slot.mem);
            if (slot.rkeyStr.empty()) {
                NIXL_ERROR << "UCX host staging rkey pack failed for slot " << i;
                uc->memDereg(slot.mem);
                cudaFreeHost(slot.hostAddr);
                for (size_t j = 0; j < i; ++j) {
                    uc->memDereg(staged->slots[j].mem);
                    cudaFreeHost(staged->slots[j].hostAddr);
                }
                return NIXL_ERR_BACKEND;
            }
        }

        NIXL_INFO << "Registered UCX staged VRAM region gpu_base=" << (void *)mem.addr
                  << " gpu_len=" << mem.len << " gpu_dev=" << mem.devId
                  << " slot_size=" << staged->slotSize << " slots=" << staged->slots.size();
        registerStagedRegion(staged.get());
        out = staged.release();
        return NIXL_SUCCESS;
    }

    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

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
    if (auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(meta)) {
        unregisterStagedRegion(staged);
        for (auto &slot : staged->slots) {
            uc->memDereg(slot.mem);
            cudaFreeHost(slot.hostAddr);
        }
        delete staged;
        return NIXL_SUCCESS;
    }

    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    if (const auto *staged = dynamic_cast<const nixlUcxStagedPrivateMetadata *>(meta)) {
        str = serializeStagedMetadata(*staged);
        return NIXL_SUCCESS;
    }

    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}

namespace {

[[nodiscard]] std::vector<nixl::ucx::rkey>
makePublicMetadataRkeys(const ucx_connection_ptr_t &conn, const size_t count, const void *buffer) {
    std::vector<nixl::ucx::rkey> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(*conn->getEp(i), buffer);
    }
    return result;
}

} // namespace

nixlUcxPublicMetadata::nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn,
                                             std::vector<nixl::ucx::rkey> &&rkeys)
    : nixlBackendMD(false),
      conn(conn),
      rkeys_(std::move(rkeys)) {}

nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        const auto it = remoteConnMap.find(agent);

        if (it == remoteConnMap.end()) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        // nixlSerDes::_stringToBytes() was used to "unpack" blob here.
        output = new nixlUcxPublicMetadata(
            it->second, makePublicMetadataRkeys(it->second, uws.size(), blob.data()));
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::internalStagedMDHelper(const nixl_blob_t &blob,
                                      const std::string &agent,
                                      nixlBackendMD *&output) {
    try {
        const auto it = remoteConnMap.find(agent);
        if (it == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixlSerDes ser_des;
        if (ser_des.importStr(blob) != NIXL_SUCCESS) {
            return NIXL_ERR_MISMATCH;
        }

        const std::string magic = ser_des.getStr("magic");
        if (magic != kUcxStagedMagic) {
            return NIXL_ERR_MISMATCH;
        }

        uintptr_t gpu_base = 0;
        size_t gpu_len = 0;
        uint64_t gpu_dev_id = 0;
        size_t slot_size = 0;
        size_t slot_count = 0;

        if (ser_des.getBuf("gpu_base", &gpu_base, sizeof(gpu_base)) != NIXL_SUCCESS ||
            ser_des.getBuf("gpu_len", &gpu_len, sizeof(gpu_len)) != NIXL_SUCCESS ||
            ser_des.getBuf("gpu_dev", &gpu_dev_id, sizeof(gpu_dev_id)) != NIXL_SUCCESS ||
            ser_des.getBuf("slot_size", &slot_size, sizeof(slot_size)) != NIXL_SUCCESS ||
            ser_des.getBuf("slot_count", &slot_count, sizeof(slot_count)) != NIXL_SUCCESS) {
            return NIXL_ERR_MISMATCH;
        }

        std::vector<uintptr_t> slot_addrs;
        std::vector<std::vector<nixl::ucx::rkey>> slot_rkeys;
        slot_addrs.reserve(slot_count);
        slot_rkeys.reserve(slot_count);

        for (size_t i = 0; i < slot_count; ++i) {
            uintptr_t slot_addr = 0;
            if (ser_des.getBuf("slot_addr", &slot_addr, sizeof(slot_addr)) != NIXL_SUCCESS) {
                return NIXL_ERR_MISMATCH;
            }
            const std::string rkey_blob = ser_des.getStr("slot_rkey");
            if (rkey_blob.empty()) {
                return NIXL_ERR_MISMATCH;
            }

            slot_addrs.push_back(slot_addr);
            slot_rkeys.emplace_back(
                makePublicMetadataRkeys(it->second, uws.size(), rkey_blob.data()));
        }

        output = new nixlUcxStagedPublicMetadata(it->second,
                                                 gpu_base,
                                                 gpu_len,
                                                 gpu_dev_id,
                                                 slot_size,
                                                 std::move(slot_addrs),
                                                 std::move(slot_rkeys));
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

uint64_t
nixlUcxEngine::nextStagedTransferId() const noexcept {
    return nextStagedTransferId_.fetch_add(1, std::memory_order_relaxed);
}

nixl_status_t
nixlUcxEngine::registerPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto [it, inserted] = pendingStagedReqs_.emplace(transfer_id, handle);
    if (!inserted) {
        NIXL_ERROR << "Duplicate UCX staged transfer id " << transfer_id;
        return NIXL_ERR_NOT_ALLOWED;
    }
    return NIXL_SUCCESS;
}

void
nixlUcxEngine::unregisterPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto it = pendingStagedReqs_.find(transfer_id);
    if (it != pendingStagedReqs_.end() && it->second == handle) {
        pendingStagedReqs_.erase(it);
    }
}

void
nixlUcxEngine::completePendingStagedReq(uint64_t transfer_id,
                                        uint64_t chunk_id,
                                        nixl_status_t status) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto it = pendingStagedReqs_.find(transfer_id);
    if (it == pendingStagedReqs_.end()) {
        NIXL_WARN << "Received UCX staged ACK for unknown transfer id " << transfer_id
                  << " chunk_id=" << chunk_id;
        return;
    }

    auto *handle = dynamic_cast<nixlUcxStagedBackendReqH *>(it->second);
    if (!handle) {
        NIXL_WARN << "Received UCX staged ACK for non-staged transfer id " << transfer_id;
        return;
    }

    handle->markAck(chunk_id, status);
}

void
nixlUcxEngine::registerStagedRegion(nixlBackendMD *metadata) {
    const std::lock_guard lock(stagedRegionMutex_);
    stagedRegions_.push_back(metadata);
}

void
nixlUcxEngine::unregisterStagedRegion(nixlBackendMD *metadata) {
    const std::lock_guard lock(stagedRegionMutex_);
    stagedRegions_.erase(std::remove(stagedRegions_.begin(), stagedRegions_.end(), metadata),
                         stagedRegions_.end());
}

nixl_status_t
nixlUcxEngine::sendStagedWriteReady(const std::string &remote_agent,
                                    uint64_t transfer_id,
                                    uint64_t chunk_id,
                                    uint64_t remote_slot_id,
                                    uintptr_t remote_gpu_addr,
                                    uint64_t remote_gpu_dev,
                                    size_t size,
                                    const std::unique_ptr<nixlUcxEp> &ep,
                                    nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("slot_id", &remote_slot_id, sizeof(remote_slot_id));
    ser_des.addBuf("gpu_addr", &remote_gpu_addr, sizeof(remote_gpu_addr));
    ser_des.addBuf("gpu_dev", &remote_gpu_dev, sizeof(remote_gpu_dev));
    ser_des.addBuf("size", &size, sizeof(size));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged WRITE_READY transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " slot_id=" << remote_slot_id << " bytes=" << size;

    return ep->sendAm(nixl::ucx::am_cb_op_t::STAGED_WRITE_READY,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER,
                      req,
                      deleter);
}

nixl_status_t
nixlUcxEngine::sendStagedAck(const std::string &remote_agent,
                             uint64_t transfer_id,
                             uint64_t chunk_id,
                             nixl_status_t status) const {
    const auto conn = getConnection(remote_agent);
    if (!conn) {
        NIXL_ERROR << "Cannot send UCX staged ACK to unknown agent " << remote_agent
                   << " transfer_id=" << transfer_id;
        return NIXL_ERR_NOT_FOUND;
    }

    nixlSerDes ser_des;
    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("status", &status, sizeof(status));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer](void *completed_request, void *ptr) {
        delete buffer;
        if (completed_request != nullptr) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged ACK transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " status=" << status;

    return conn->getEp(0)->sendAm(nixl::ucx::am_cb_op_t::STAGED_ACK,
                                  nullptr,
                                  0,
                                  (void *)buffer->data(),
                                  buffer->size(),
                                  UCP_AM_SEND_FLAG_EAGER,
                                  nullptr,
                                  deleter);
}

nixl_status_t
nixlUcxEngine::handleStagedWriteReady(const nixl_blob_t &message) const {
    nixlSerDes ser_des;
    if (ser_des.importStr(message) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged WRITE_READY message";
        return NIXL_ERR_MISMATCH;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t slot_id = 0;
    uintptr_t gpu_addr = 0;
    uint64_t gpu_dev = 0;
    size_t size = 0;

    nixl_status_t status =
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("slot_id", &slot_id, sizeof(slot_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_addr", &gpu_addr, sizeof(gpu_addr)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_dev", &gpu_dev, sizeof(gpu_dev)) == NIXL_SUCCESS &&
                ser_des.getBuf("size", &size, sizeof(size)) == NIXL_SUCCESS ?
            NIXL_SUCCESS :
            NIXL_ERR_MISMATCH;

    if (remote_agent.empty()) {
        status = NIXL_ERR_MISMATCH;
    }

    if (status == NIXL_SUCCESS) {
        const std::lock_guard lock(stagedRegionMutex_);
        status = NIXL_ERR_NOT_FOUND;
        for (nixlBackendMD *region : stagedRegions_) {
            auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region);
            if (!staged || staged->gpuDevId != gpu_dev ||
                !rangeCovers(staged->gpuBase, staged->gpuLen, gpu_addr, size)) {
                continue;
            }

            if (slot_id >= staged->slots.size() || size > staged->slots[slot_id].size) {
                status = NIXL_ERR_INVALID_PARAM;
                break;
            }

            int previous_device = -1;
            status = cudaSetDeviceForCopy(gpu_dev, previous_device);
            if (status == NIXL_SUCCESS) {
                const cudaError_t cuda_ret = cudaMemcpy((void *)gpu_addr,
                                                        staged->slots[slot_id].hostAddr,
                                                        size,
                                                        cudaMemcpyHostToDevice);
                if (cuda_ret != cudaSuccess) {
                    NIXL_ERROR << "UCX staged H2D failed for transfer_id=" << transfer_id << ": "
                               << cudaGetErrorString(cuda_ret);
                    status = NIXL_ERR_BACKEND;
                }
            }
            cudaRestoreDevice(previous_device);
            break;
        }
    }

    if (!remote_agent.empty()) {
        const nixl_status_t ack_status =
            sendStagedAck(remote_agent, transfer_id, chunk_id, status);
        if (ack_status != NIXL_SUCCESS && ack_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to send UCX staged ACK transfer_id=" << transfer_id
                       << " chunk_id=" << chunk_id << " status=" << ack_status;
            return ack_status;
        }
    }

    return status;
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    if (auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(input)) {
        return internalStagedMDHelper(serializeStagedMetadata(*staged), localAgent, output);
    }

    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    if (vramStagingEnabled() && nixl_mem == VRAM_SEG) {
        if (!isStagedMetadataBlob(input.metaInfo)) {
            NIXL_ERROR << "UCX VRAM staging expected staged VRAM metadata from " << remote_agent;
            return NIXL_ERR_MISMATCH;
        }
        return internalStagedMDHelper(input.metaInfo, remote_agent, output);
    }

    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {
    delete input;
    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

size_t
nixlUcxEngine::getWorkerId(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (opt_args) {
        const std::optional<size_t> worker_id = getWorkerIdFromOptArgs(*opt_args);
        if (worker_id) {
            return *worker_id;
        }
    }

    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        const size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept {
    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args.customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args.customParam.substr(pos + worker_id_key.length()));

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

    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        if (operation != NIXL_WRITE) {
            NIXL_ERROR << "UCX VRAM staging currently supports NIXL_WRITE only";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (local.getType() != VRAM_SEG || remote.getType() != VRAM_SEG) {
            NIXL_ERROR << "UCX VRAM staging requires both local and remote descriptors to be VRAM";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        constexpr size_t worker_id = 0;
        handle = new nixlUcxStagedBackendReqH(
            getWorker(worker_id).get(), worker_id, nextStagedTransferId());
        return NIXL_SUCCESS;
    }

    const size_t worker_id = getWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    handle = new nixlUcxBackendReqH(getWorker(worker_id).get(), worker_id);

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
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

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

    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        if (local.getType() != VRAM_SEG || remote.getType() != VRAM_SEG) {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        for (int i = 0; i < local.descCount(); i++) {
            const size_t lsize = local[i].len;
            const size_t rsize = remote[i].len;
            const auto rmd = dynamic_cast<nixlUcxStagedPublicMetadata *>(remote[i].metadataP);

            if (!rmd || lsize != rsize) {
                return NIXL_ERR_MISMATCH;
            }

            std::chrono::microseconds msg_duration;
            std::chrono::microseconds msg_err_margin;
            nixl_cost_t msg_method;
            const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
                lsize, msg_duration, msg_err_margin, msg_method);
            if (ret != NIXL_SUCCESS) {
                return ret;
            }

            duration += msg_duration;
            err_margin += msg_err_margin;
            method = msg_method;
        }
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        const size_t lsize = local[i].len;
        const size_t rsize = remote[i].len;

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
            lsize, msg_duration, msg_err_margin, msg_method);
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

nixlUcxEngine::batchResult
nixlUcxEngine::sendXferRangeBatch(nixlUcxEp &ep,
                                  nixl_xfer_op_t operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  size_t worker_id,
                                  size_t start_idx,
                                  size_t end_idx) {
    batchResult result = {NIXL_SUCCESS, 0, nullptr};

    for (size_t i = start_idx; i < end_idx; ++i) {
        void *laddr = (void *)local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = static_cast<uint64_t>(remote[i].addr);
        NIXL_ASSERT(lsize == remote[i].len);

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &rmd_ep = rmd->conn->getEp(worker_id);
        if (__builtin_expect(rmd_ep.get() != &ep, 0)) {
            break;
        }

        ++result.size;
        nixlUcxReq req;
        const nixl_status_t ret = operation == NIXL_READ ?
            ep.read(raddr, rmd->getRkey(worker_id), laddr, lmd->mem, lsize, req) :
            ep.write(laddr, lmd->mem, raddr, rmd->getRkey(worker_id), lsize, req);

        if (ret == NIXL_IN_PROG) {
            if (__builtin_expect(result.req != nullptr, 1)) {
                ucp_request_free(result.req);
            }
            result.req = req;
        } else if (ret != NIXL_SUCCESS) {
            result.status = ret;
            if (result.req != nullptr) {
                ucp_request_free(result.req);
                result.req = nullptr;
            }
            break;
        }
    }

    if (result.status == NIXL_SUCCESS && result.req) {
        result.status = NIXL_IN_PROG;
    }
    return result;
}

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

    /* Assuming we have a single EP, we need 3 requests: one pending request,
     * one flush request, and one notification request */
    int_handle->reserve(3);

    for (size_t i = start_idx; i < end_idx;) {
        /* Send requests to a single EP */
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &ep = rmd->conn->getEp(worker_id);
        const batchResult result =
            sendXferRangeBatch(*ep, operation, local, remote, worker_id, i, end_idx);

        /* Append a single pending request for the entire EP batch */
        const nixl_status_t ret = int_handle->append(result.status, result.req, rmd->conn);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }

        i += result.size;
    }

    /*
     * Flush keeps int_handle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     * We need to flush all distinct connections to ensure that the operation
     * is actually completed.
     */
    for (auto &conn : int_handle->getConnections()) {
        nixlUcxReq req;
        const nixl_status_t ret = conn->getEp(worker_id)->flushEp(req);
        if (int_handle->append(ret, req, conn) != NIXL_SUCCESS) {
            return ret;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postStagedWrite(const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *handle,
                               const nixl_opt_b_args_t *opt_args) const {
    auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle);
    if (!staged_handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local.descCount() != 1 || remote.descCount() != 1) {
        NIXL_ERROR << "UCX VRAM staging v1 supports exactly one descriptor per WRITE";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (local[0].len != remote[0].len || local[0].len == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *lmd = dynamic_cast<nixlUcxStagedPrivateMetadata *>(local[0].metadataP);
    auto *rmd = dynamic_cast<nixlUcxStagedPublicMetadata *>(remote[0].metadataP);
    if (!lmd || !rmd) {
        NIXL_ERROR << "UCX VRAM staging metadata mismatch";
        return NIXL_ERR_MISMATCH;
    }

    const size_t size = local[0].len;
    if (!rangeCovers(lmd->gpuBase, lmd->gpuLen, local[0].addr, size) ||
        !rangeCovers(rmd->gpuBase, rmd->gpuLen, remote[0].addr, size)) {
        NIXL_ERROR << "UCX VRAM staging descriptor is outside registered region";
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t chunk_size = std::min(lmd->slotSize, rmd->slotSize);
    if (chunk_size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (rmd->slotAddrs.empty() || rmd->slotRkeys.empty()) {
        return NIXL_ERR_MISMATCH;
    }

    staged_handle->remoteAgent = remote_agent;
    staged_handle->localMetadata = lmd;
    staged_handle->remoteMetadata = rmd;
    staged_handle->localGpuAddr = local[0].addr;
    staged_handle->localGpuDev = local[0].devId;
    staged_handle->remoteGpuAddr = remote[0].addr;
    staged_handle->remoteGpuDev = remote[0].devId;
    staged_handle->totalSize = size;
    staged_handle->chunkSize = chunk_size;
    staged_handle->remoteSlotBusy.assign(rmd->slotAddrs.size(), false);

    const size_t chunk_count = (size + chunk_size - 1) / chunk_size;
    staged_handle->chunks.reserve(chunk_count);
    for (size_t i = 0; i < chunk_count; ++i) {
        const size_t offset = i * chunk_size;
        const size_t this_chunk_size = std::min(chunk_size, size - offset);
        auto chunk =
            std::make_unique<nixlUcxStagedChunk>(static_cast<uint64_t>(i),
                                                 offset,
                                                 this_chunk_size);
        chunk->req =
            std::make_unique<nixlUcxBackendReqH>(staged_handle->getWorker(),
                                                 staged_handle->getWorkerId());
        chunk->req->reserve(3);
        staged_handle->chunks.push_back(std::move(chunk));
    }

    nixl_status_t ret = registerPendingStagedReq(staged_handle->transferId, staged_handle);
    if (ret != NIXL_SUCCESS) {
        staged_handle->release();
        return ret;
    }
    staged_handle->pendingRegistered = true;
    staged_handle->state = nixlUcxStagedBackendReqH::State::RUNNING;

    if (opt_args && opt_args->hasNotif) {
        staged_handle->notif.emplace(remote_agent, opt_args->notifMsg);
    }

    return checkStagedXfer(staged_handle);
}

nixl_status_t
nixlUcxEngine::checkStagedXfer(nixlBackendReqH *handle) const {
    auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle);
    if (!staged_handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto fail = [&](nixl_status_t status) {
        staged_handle->lastStatus = status;
        staged_handle->state = nixlUcxStagedBackendReqH::State::FAILED;
        staged_handle->releaseAllChunkSlots();
        if (staged_handle->pendingRegistered) {
            unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
            staged_handle->pendingRegistered = false;
        }
        staged_handle->release();
        return status;
    };

    auto schedule_chunk = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        const auto remote_slot_id = staged_handle->acquireRemoteSlot();
        if (!remote_slot_id) {
            return NIXL_IN_PROG;
        }

        const auto local_slot_id = staged_handle->localMetadata->acquireSlot();
        if (!local_slot_id) {
            staged_handle->remoteSlotBusy[*remote_slot_id] = false;
            return NIXL_IN_PROG;
        }

        chunk.remoteSlotId = *remote_slot_id;
        chunk.remoteSlotHeld = true;
        chunk.localSlotId = *local_slot_id;
        chunk.localSlotHeld = true;

        int previous_device = -1;
        nixl_status_t ret = cudaSetDeviceForCopy(staged_handle->localGpuDev, previous_device);
        if (ret != NIXL_SUCCESS) {
            staged_handle->releaseChunkSlots(chunk);
            return ret;
        }

        nixlUcxStagedSlot &local_slot = staged_handle->localMetadata->slots[chunk.localSlotId];
        const auto local_gpu_addr =
            reinterpret_cast<void *>(staged_handle->localGpuAddr + chunk.offset);
        const cudaError_t cuda_ret =
            cudaMemcpy(local_slot.hostAddr, local_gpu_addr, chunk.size, cudaMemcpyDeviceToHost);
        cudaRestoreDevice(previous_device);
        if (cuda_ret != cudaSuccess) {
            NIXL_ERROR << "UCX staged D2H failed for transfer_id="
                       << staged_handle->transferId << " chunk_id=" << chunk.id << ": "
                       << cudaGetErrorString(cuda_ret);
            staged_handle->releaseChunkSlots(chunk);
            return NIXL_ERR_BACKEND;
        }

        const size_t worker_id = staged_handle->getWorkerId();
        const auto *rmd = staged_handle->remoteMetadata;
        if (chunk.remoteSlotId >= rmd->slotRkeys.size() ||
            worker_id >= rmd->slotRkeys[chunk.remoteSlotId].size()) {
            staged_handle->releaseChunkSlots(chunk);
            return NIXL_ERR_MISMATCH;
        }

        const auto &ep = rmd->conn->getEp(worker_id);
        nixlUcxReq req = nullptr;
        ret = ep->write(local_slot.hostAddr,
                        local_slot.mem,
                        rmd->slotAddrs[chunk.remoteSlotId],
                        rmd->slotRkeys[chunk.remoteSlotId][worker_id],
                        chunk.size,
                        req);
        ret = chunk.req->append(ret, req, rmd->conn);
        if (ret != NIXL_SUCCESS) {
            staged_handle->releaseChunkSlots(chunk);
            return ret;
        }

        nixlUcxReq flush_req = nullptr;
        ret = ep->flushEp(flush_req);
        ret = chunk.req->append(ret, flush_req, rmd->conn);
        if (ret != NIXL_SUCCESS) {
            staged_handle->releaseChunkSlots(chunk);
            return ret;
        }

        chunk.state = nixlUcxStagedChunk::State::RDMA_POSTED;
        return NIXL_SUCCESS;
    };

    auto schedule_ready_chunks = [&]() -> nixl_status_t {
        while (staged_handle->nextChunkToPost < staged_handle->chunks.size()) {
            nixlUcxStagedChunk &chunk =
                *staged_handle->chunks[staged_handle->nextChunkToPost];
            const nixl_status_t status = schedule_chunk(chunk);
            if (status == NIXL_IN_PROG) {
                return NIXL_SUCCESS;
            }
            if (status != NIXL_SUCCESS) {
                return status;
            }
            ++staged_handle->nextChunkToPost;
        }
        return NIXL_SUCCESS;
    };

    while (true) {
        switch (staged_handle->state) {
        case nixlUcxStagedBackendReqH::State::INIT:
            return NIXL_ERR_NOT_POSTED;

        case nixlUcxStagedBackendReqH::State::RUNNING: {
            staged_handle->getWorker()->progressLoop();
            bool made_progress = false;

            for (const auto &chunk_ptr : staged_handle->chunks) {
                nixlUcxStagedChunk &chunk = *chunk_ptr;

                if (chunk.state == nixlUcxStagedChunk::State::RDMA_POSTED) {
                    const nixl_status_t status = chunk.req->status();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    staged_handle->releaseLocalSlot(chunk);

                    const auto conn = getConnection(staged_handle->remoteAgent);
                    if (!conn) {
                        return fail(NIXL_ERR_NOT_FOUND);
                    }

                    nixlUcxReq req = nullptr;
                    const uintptr_t remote_gpu_addr =
                        staged_handle->remoteGpuAddr + chunk.offset;
                    const nixl_status_t send_status =
                        sendStagedWriteReady(staged_handle->remoteAgent,
                                             staged_handle->transferId,
                                             chunk.id,
                                             chunk.remoteSlotId,
                                             remote_gpu_addr,
                                             staged_handle->remoteGpuDev,
                                             chunk.size,
                                             conn->getEp(staged_handle->getWorkerId()),
                                             &req);
                    const nixl_status_t append_status =
                        chunk.req->append(send_status, req, conn);
                    if (append_status != NIXL_SUCCESS) {
                        return fail(append_status);
                    }

                    chunk.state = nixlUcxStagedChunk::State::READY_AM_POSTED;
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::READY_AM_POSTED) {
                    const nixl_status_t status = chunk.req->status();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    chunk.state = nixlUcxStagedChunk::State::WAIT_ACK;
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::WAIT_ACK) {
                    const nixl_status_t status = chunk.ackStatus.load();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    staged_handle->releaseRemoteSlot(chunk);
                    chunk.state = nixlUcxStagedChunk::State::ACKED;
                    ++staged_handle->completedChunks;
                    made_progress = true;
                }
            }

            const nixl_status_t schedule_status = schedule_ready_chunks();
            if (schedule_status != NIXL_SUCCESS) {
                return fail(schedule_status);
            }

            if (staged_handle->completedChunks != staged_handle->chunks.size()) {
                if (made_progress) {
                    continue;
                }
                return NIXL_IN_PROG;
            }

            if (staged_handle->pendingRegistered) {
                unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
                staged_handle->pendingRegistered = false;
            }

            if (!staged_handle->notif) {
                staged_handle->lastStatus = NIXL_SUCCESS;
                staged_handle->state = nixlUcxStagedBackendReqH::State::COMPLETE;
                return NIXL_SUCCESS;
            }

            const nixlUcxBackendReqH::Notif notif(std::move(staged_handle->notif).value());
            staged_handle->notif.reset();

            const ucx_connection_ptr_t conn = getConnection(notif.agent);
            if (!conn) {
                return fail(NIXL_ERR_NOT_FOUND);
            }

            nixlUcxReq req = nullptr;
            const auto &ep = conn->getEp(staged_handle->getWorkerId());
            const nixl_status_t send_status = notifSendPriv(notif.agent, notif.payload, ep, &req);
            const nixl_status_t append_status = staged_handle->append(send_status, req, conn);
            if (append_status != NIXL_SUCCESS) {
                return fail(append_status);
            }

            staged_handle->state = nixlUcxStagedBackendReqH::State::USER_NOTIF_POSTED;
            continue;
        }

        case nixlUcxStagedBackendReqH::State::USER_NOTIF_POSTED: {
            const nixl_status_t status = staged_handle->nixlUcxBackendReqH::status();
            if (status == NIXL_IN_PROG) {
                return status;
            }
            if (status != NIXL_SUCCESS) {
                return fail(status);
            }
            staged_handle->lastStatus = NIXL_SUCCESS;
            staged_handle->state = nixlUcxStagedBackendReqH::State::COMPLETE;
            return NIXL_SUCCESS;
        }

        case nixlUcxStagedBackendReqH::State::COMPLETE:
            return NIXL_SUCCESS;

        case nixlUcxStagedBackendReqH::State::FAILED:
            return staged_handle->lastStatus;
        }
    }
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    const size_t lcnt = local.descCount();
    const size_t rcnt = remote.descCount();
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        if (operation != NIXL_WRITE) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        return postStagedWrite(local, remote, remote_agent, handle, opt_args);
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
            const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[0].metadataP);
            ret = notifSendPriv(remote_agent,
                                opt_args->notifMsg,
                                rmd->conn->getEp(int_handle->getWorkerId()),
                                &req);
            if (int_handle->append(ret, req, rmd->conn) != NIXL_SUCCESS) {
                return ret;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notif.emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    if (dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        return checkStagedXfer(handle);
    }

    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const nixl_status_t handle_status = int_handle->status();

    if ((handle_status == NIXL_IN_PROG) || !int_handle->notif) {
        return handle_status;
    }

    const nixlUcxBackendReqH::Notif notif(std::move(int_handle->notif).value());
    int_handle->notif.reset();

    if (__builtin_expect(handle_status != NIXL_SUCCESS, 0)) {
        return handle_status;
    }

    const ucx_connection_ptr_t conn = getConnection(notif.agent);
    if (__builtin_expect(!conn, 0)) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    const auto &ep = conn->getEp(int_handle->getWorkerId());
    const nixl_status_t status = notifSendPriv(notif.agent, notif.payload, ep, &req);

    if (int_handle->append(status, req, conn) != NIXL_SUCCESS) {
        return status;
    }

    return int_handle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    if (auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        if (staged_handle->pendingRegistered) {
            unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
            staged_handle->pendingRegistered = false;
        }
        staged_handle->release();
        delete staged_handle;
        return NIXL_SUCCESS;
    }

    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    int_handle->release();

    /* TODO: return to a pool instead. */
    delete int_handle;

    return NIXL_SUCCESS;
}

unsigned
nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    unsigned ret = 0;
    for (auto &uw : uws) {
        ret += uw->progress();
    }
    return ret;
}

void
nixlUcxEngine::progressLoop() {
    while (progress() != 0)
        ;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t
nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                             const std::string &msg,
                             const std::unique_ptr<nixlUcxEp> &ep,
                             nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            /* Caller is not interested in the request, free it */
            ucp_request_free(completed_request);
        }
    };

    return ep->sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER,
                      req,
                      deleter);
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    const auto it = remoteConnMap.find(remote_agent);
    return (it != remoteConnMap.end()) ? it->second : nullptr;
}

void
nixlUcxEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
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

ucs_status_t
nixlUcxEngine::stagedWriteReadyAmCb(void *arg,
                                    const void *header,
                                    size_t header_length,
                                    void *data,
                                    size_t length,
                                    const ucp_am_recv_param_t *param) {
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    engine->handleStagedWriteReady(ser_str);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedAckAmCb(void *arg,
                             const void *header,
                             size_t header_length,
                             void *data,
                             size_t length,
                             const ucp_am_recv_param_t *param) {
    nixlSerDes ser_des;
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    if (ser_des.importStr(ser_str) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged ACK message";
        return UCS_OK;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    nixl_status_t status = NIXL_ERR_MISMATCH;
    if (remote_agent.empty() ||
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("status", &status, sizeof(status)) != NIXL_SUCCESS) {
        NIXL_ERROR << "Malformed UCX staged ACK message";
        return UCS_OK;
    }

    engine->completePendingStagedReq(transfer_id, chunk_id, status);
    return UCS_OK;
}

nixl_status_t
nixlUcxEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    progressLoop();

    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    const auto conn = getConnection(remote_agent);
    if (!conn) {
        return NIXL_ERR_NOT_FOUND;
    }

    const nixl_status_t ret = notifSendPriv(remote_agent, msg, conn->getEp(getWorkerId()));
    if (ret == NIXL_IN_PROG) {
        return NIXL_SUCCESS;
    }
    return ret;
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, worker_id, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare remote memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare local memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseMemView(nixlMemViewH mem_view) const {
    nixl::ucx::releaseMemList(mem_view);
}
