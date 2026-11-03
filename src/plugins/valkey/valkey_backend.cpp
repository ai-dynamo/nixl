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

// Same model as the OBJ backend: an OBJ_SEG descriptor is a Valkey key
// (metaInfo, or devId), NIXL_WRITE is BLOB.RDMA_SET, NIXL_READ is
// BLOB.RDMA_GET, queryMem is BLOB.INFO LEN. A descriptor must cover the whole
// object: BLOB has no range commands.
//
// Every DRAM region is registered with each worker's GLIDE client at
// registerMem and a transfer passes its window of that region.

#include "valkey_backend.h"

#include "common/nixl_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>

class nixlValkeyReqH : public nixlBackendReqH {
public:
    std::atomic<int> pending{0};
    std::atomic<int> status{NIXL_SUCCESS};

    void
    fail(nixl_status_t st) {
        int expected = NIXL_SUCCESS;
        status.compare_exchange_strong(expected, st);
    }
};

// A DRAM region (its id in liveRegions_) or an object (key and length).
class nixlValkeyMD : public nixlBackendMD {
public:
    nixlValkeyMD(nixl_mem_t nixl_mem, uint64_t region_id, size_t len, std::string key)
        : nixlBackendMD(true),
          mem(nixl_mem),
          regionId(region_id),
          len(len),
          key(std::move(key)) {}

    nixl_mem_t mem;
    uint64_t regionId;
    size_t len;
    std::string key;
};

namespace {

int64_t
nowNanos() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template<typename T, typename Fn>
void
inParallel(const std::vector<T *> &items, Fn fn) {
    std::vector<std::thread> threads;
    threads.reserve(items.size());
    for (T *item : items) {
        threads.emplace_back([&fn, item] { fn(*item); });
    }
    for (auto &t : threads) {
        t.join();
    }
}

} // namespace

nixlValkeyEngine::nixlValkeyEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      config_(nixlValkeyConfig::fromBackendParams(init_params->customParams)) {
    if (!nixl::valkey::rdmaUsable()) {
        NIXL_ERROR << "VALKEY: GLIDE RDMA unavailable (built without the rdma feature, or "
                      "libfabric not loadable)";
        initErr = true;
        return;
    }

    // A wrong address or a server without the module fails here.
    ctrl_ = client::open(config_, false);
    if (!ctrl_ ||
        ctrl_->blobLen(config_.keyPrefix + "nixl-valkey-probe") == nixl::valkey::BLOB_ERROR) {
        initErr = true;
        return;
    }

    // Every worker opens its fabric session before the backend is usable.
    workerState_.assign(config_.numWorkers, nullptr);
    for (int i = 0; i < config_.numWorkers; i++) {
        workers_.emplace_back(&nixlValkeyEngine::workerLoop, this, i);
    }
    {
        std::unique_lock<std::mutex> lk(queueMutex_);
        startCv_.wait(lk, [&] {
            return std::all_of(
                workerState_.begin(), workerState_.end(), [](worker *w) { return w; });
        });
        initErr = std::any_of(
            workerState_.begin(), workerState_.end(), [](worker *w) { return !w->conn; });
    }
    if (initErr) {
        return;
    }
    if (config_.transferTimeoutMs > 0) {
        watchdog_ = std::thread(&nixlValkeyEngine::watchdogLoop, this);
    }
    NIXL_INFO << "VALKEY: backend " << config_.host << ":" << config_.port
              << " rdma_provider=" << config_.rdmaProvider << " workers=" << config_.numWorkers;
}

nixlValkeyEngine::~nixlValkeyEngine() {
    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        stop_ = true;
    }
    queueCv_.notify_all();
    if (watchdog_.joinable()) {
        watchdog_.join();
    }
    for (auto &t : workers_) {
        t.join();
    }
    if (ctrl_) {
        ctrl_->close();
    }
}

std::string
nixlValkeyEngine::objectKey(const nixlBlobDesc &desc) const {
    return config_.keyPrefix + (desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo);
}

// ---- registration ----------------------------------------------------------

nixl_status_t
nixlValkeyEngine::registerMem(const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) {
    out = nullptr;
    if (nixl_mem == OBJ_SEG) {
        out = new nixlValkeyMD(OBJ_SEG, 0, mem.len, objectKey(mem));
        return NIXL_SUCCESS;
    }
    if (nixl_mem == DRAM_SEG) {
        return registerDram(mem, out);
    }
    return NIXL_ERR_NOT_SUPPORTED;
}

// Every worker's client registers the region now, in parallel; any failure
// rolls back.
nixl_status_t
nixlValkeyEngine::registerDram(const nixlBlobDesc &mem, nixlBackendMD *&out) {
    const memRegion reg = {mem.addr, mem.len};
    uint64_t id;
    {
        std::lock_guard<std::mutex> lk(regMutex_);
        id = nextRegion_++;
    }
    std::atomic<bool> failed{false};
    inParallel(workers(), [&](worker &w) {
        std::lock_guard<std::mutex> lk(w.lock);
        if (!w.conn) {
            return; // registers on its next transfer
        }
        if (auto *r = w.conn->registerHost(reg.addr, reg.len)) {
            w.regions.emplace(id, r);
        } else {
            failed = true;
        }
    });
    if (failed) {
        dropRegistrations(id);
        return NIXL_ERR_BACKEND;
    }
    {
        std::lock_guard<std::mutex> lk(regMutex_);
        liveRegions_[id] = reg;
    }
    out = new nixlValkeyMD(DRAM_SEG, id, 0, std::string());
    return NIXL_SUCCESS;
}

nixl_status_t
nixlValkeyEngine::deregisterMem(nixlBackendMD *meta) {
    auto *md = static_cast<nixlValkeyMD *>(meta);
    if (md && md->mem == DRAM_SEG) {
        {
            std::lock_guard<std::mutex> lk(regMutex_);
            liveRegions_.erase(md->regionId);
        }
        dropRegistrations(md->regionId);
    }
    delete md;
    return NIXL_SUCCESS;
}

void
nixlValkeyEngine::dropRegistrations(uint64_t id) const {
    inParallel(workers(), [&](worker &w) {
        glide::RdmaRegion *r = nullptr;
        {
            std::lock_guard<std::mutex> lk(w.lock);
            if (auto it = w.regions.find(id); it != w.regions.end()) {
                r = it->second;
                w.regions.erase(it);
            }
        }
        if (r) {
            nixl::valkey::releaseRegion(r);
        }
    });
}

std::vector<nixlValkeyEngine::worker *>
nixlValkeyEngine::workers() const {
    std::lock_guard<std::mutex> lk(queueMutex_);
    std::vector<worker *> ws;
    std::copy_if(workerState_.begin(), workerState_.end(), std::back_inserter(ws), [](worker *w) {
        return w != nullptr;
    });
    return ws;
}

nixl_status_t
nixlValkeyEngine::queryMem(const nixl_reg_dlist_t &descs,
                           std::vector<nixl_query_resp_t> &resp) const {
    std::vector<std::string> keys;
    keys.reserve(descs.descCount());
    for (int i = 0; i < descs.descCount(); i++) {
        keys.push_back(objectKey(descs[i]));
    }
    std::vector<long long> lens;
    {
        std::lock_guard<std::mutex> lk(ctrlMutex_);
        lens = ctrl_->blobLens(keys);
    }
    resp.assign(descs.descCount(), std::nullopt);
    for (size_t i = 0; i < lens.size(); i++) {
        if (lens[i] == nixl::valkey::BLOB_ERROR) {
            return NIXL_ERR_BACKEND;
        }
        if (lens[i] > 0) {
            resp[i] = nixl_b_params_t{{"size", std::to_string(lens[i])}};
        }
    }
    return NIXL_SUCCESS;
}

// ---- transfer requests -----------------------------------------------------

nixl_status_t
nixlValkeyEngine::prepXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *) const {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote_agent != localAgent) {
        NIXL_WARN << "VALKEY: remote agent " << remote_agent << " is not the local agent "
                  << localAgent;
    }
    if (local.getType() != DRAM_SEG || remote.getType() != OBJ_SEG) {
        NIXL_ERROR << "VALKEY: local must be DRAM_SEG and remote OBJ_SEG";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() != remote.descCount()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    handle = new nixlValkeyReqH();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlValkeyEngine::postXfer(const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *) const {
    auto *req = static_cast<nixlValkeyReqH *>(handle);
    if (!req) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (req->pending.load(std::memory_order_acquire) > 0) {
        return NIXL_ERR_REPOST_ACTIVE;
    }

    // Validate everything before queuing anything.
    std::vector<job> jobs;
    jobs.reserve(local.descCount());
    for (int i = 0; i < local.descCount(); i++) {
        const auto *obj = static_cast<const nixlValkeyMD *>(remote[i].metadataP);
        const auto *region = static_cast<const nixlValkeyMD *>(local[i].metadataP);
        if (!obj || !region) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (remote[i].addr != 0 || local[i].len < obj->len) {
            NIXL_ERROR << "VALKEY: object " << obj->key
                       << ": a descriptor must cover the whole object (offset " << remote[i].addr
                       << ", " << local[i].len << " of " << obj->len << " bytes)";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        jobs.push_back({operation, region->regionId, obj->key, local[i].addr, local[i].len, req});
    }
    if (jobs.empty()) {
        return NIXL_SUCCESS;
    }

    req->status.store(NIXL_SUCCESS, std::memory_order_relaxed);
    req->pending.store((int)jobs.size(), std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        for (auto &j : jobs) {
            queue_.push_back(std::move(j));
        }
    }
    queueCv_.notify_all();
    return NIXL_IN_PROG;
}

nixl_status_t
nixlValkeyEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlValkeyReqH *>(handle);
    if (req->pending.load(std::memory_order_acquire) > 0) {
        return NIXL_IN_PROG;
    }
    return (nixl_status_t)req->status.load(std::memory_order_relaxed);
}

nixl_status_t
nixlValkeyEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlValkeyReqH *>(handle);
    // Workers hold the handle until their jobs finish.
    while (req->pending.load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    delete req;
    return NIXL_SUCCESS;
}

// ---- workers ---------------------------------------------------------------

void
nixlValkeyEngine::workerLoop(size_t idx) {
    worker w;
    w.conn = client::open(config_, true);
    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        workerState_[idx] = &w;
    }
    startCv_.notify_all();

    while (true) {
        job j;
        {
            std::unique_lock<std::mutex> lk(queueMutex_);
            queueCv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                break;
            }
            j = std::move(queue_.front());
            queue_.pop_front();
        }
        nixl_status_t st = runJob(w, j);
        if (st != NIXL_SUCCESS) {
            j.req->fail(st);
        }
        j.req->pending.fetch_sub(1, std::memory_order_acq_rel);
    }
    dropConn(w);
    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        workerState_[idx] = nullptr;
    }
}

nixl_status_t
nixlValkeyEngine::runJob(worker &w, const job &j) const {
    {
        std::lock_guard<std::mutex> lk(w.lock);
        if (!w.conn && !(w.conn = client::open(config_, true))) {
            return NIXL_ERR_BACKEND;
        }
    }
    memRegion reg;
    {
        std::lock_guard<std::mutex> lk(regMutex_);
        auto live = liveRegions_.find(j.regionId);
        if (live == liveRegions_.end()) {
            NIXL_ERROR << "VALKEY: " << j.key << ": local memory was deregistered";
            return NIXL_ERR_INVALID_PARAM;
        }
        reg = live->second;
    }
    glide::RdmaRegion *region = nullptr;
    {
        std::lock_guard<std::mutex> lk(w.lock);
        nixl_status_t st = regionFor(w, j.regionId, reg, region);
        if (st != NIXL_SUCCESS) {
            return st;
        }
    }

    const bool write = j.op == NIXL_WRITE;
    bool closed = false;
    if (config_.transferTimeoutMs > 0) {
        w.deadline = nowNanos() + int64_t(config_.transferTimeoutMs) * 1000000;
    }
    nixl_status_t st = w.conn->transfer(
        write, j.key, region, j.addr - reg.addr, j.len, config_.writeChecksum, closed);
    w.deadline = 0;
    if (closed) {
        dropConn(w);
    }
    // A value larger than the buffer is refused before anything is written.
    if (st == NIXL_ERR_BACKEND && !write && w.conn && w.conn->blobLen(j.key) > (long long)j.len) {
        st = NIXL_ERR_MISMATCH;
    }
    return st;
}

// Cancel transfers that outlived transfer_timeout_ms: the worker's transfer
// then returns closed and it drops the client (dropConn).
void
nixlValkeyEngine::watchdogLoop() {
    const auto period =
        std::chrono::milliseconds(std::clamp(config_.transferTimeoutMs / 10, 1, 100));
    while (true) {
        {
            std::unique_lock<std::mutex> lk(queueMutex_);
            if (queueCv_.wait_for(lk, period, [&] { return stop_; })) {
                return;
            }
        }
        for (worker *w : workers()) {
            int64_t deadline = w->deadline.load();
            if (deadline == 0 || nowNanos() < deadline) {
                continue;
            }
            std::lock_guard<std::mutex> lk(w->lock);
            if (w->conn && w->deadline.exchange(0) != 0) {
                NIXL_ERROR << "VALKEY: transfer exceeded " << config_.transferTimeoutMs
                           << " ms, cancelling";
                w->conn->closeRegions();
            }
        }
    }
}

// Registers a region the worker lost on reconnect. Caller holds w.lock.
nixl_status_t
nixlValkeyEngine::regionFor(worker &w,
                            uint64_t id,
                            const memRegion &reg,
                            glide::RdmaRegion *&region) const {
    auto it = w.regions.find(id);
    if (it == w.regions.end()) {
        glide::RdmaRegion *r = w.conn->registerHost(reg.addr, reg.len);
        if (!r) {
            return NIXL_ERR_BACKEND;
        }
        it = w.regions.emplace(id, r).first;
    }
    region = it->second;
    return NIXL_SUCCESS;
}

// Drop the client with all its registrations; a transfer that ended early may
// have left the server able to reach the memory.
void
nixlValkeyEngine::dropConn(worker &w) const {
    std::vector<glide::RdmaRegion *> regions;
    client *c;
    {
        std::lock_guard<std::mutex> lk(w.lock);
        for (auto &kv : w.regions) {
            regions.push_back(kv.second);
        }
        w.regions.clear();
        c = w.conn;
        w.conn = nullptr;
    }
    nixl::valkey::closeWithRegions(c, regions, config_.timeoutMs);
}
