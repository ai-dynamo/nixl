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
#include "mooncake_backend.h"
#include "serdes/serdes.h"
#include "common/backend.h"
#include "common/configuration.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <condition_variable>
#include <functional>
#include <future>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <optional>
#include <queue>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace {
std::vector<std::string>
findLocalIpAddresses() {
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        return ips;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET) {
            if (strcmp(ifa->ifa_name, "lo") == 0) {
                continue;
            }

            // Check if interface is UP and RUNNING
            if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) {
                NIXL_INFO << "Skipping interface " << ifa->ifa_name << " (not UP or not RUNNING)";
                continue;
            }

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr,
                            sizeof(struct sockaddr_in),
                            host,
                            NI_MAXHOST,
                            nullptr,
                            0,
                            NI_NUMERICHOST) == 0) {
                ips.push_back(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

[[nodiscard]] std::string
chooseIpAddress() {
    static const std::string local = "127.0.0.1";
    static const std::vector<std::string> ips = findLocalIpAddresses();
    static const std::string &fallback = ips.empty() ? local : ips[0];
    return nixl::config::getValueDefaulted("NIXL_MOONCAKE_IP_ADDR", fallback);
}

// Returns NIXL_IN_PROG while the batch is in flight, then NIXL_SUCCESS if every
// request completed or NIXL_ERR_BACKEND otherwise. Observing completion also
// lets the Transfer Engine send the notification attached to the batch.
[[nodiscard]] nixl_status_t
queryBatchStatus(transfer_engine_t engine,
                 uint64_t batch_id,
                 [[maybe_unused]] size_t request_count) {
#ifdef HAVE_MOONCAKE_BATCH_STATUS
    // One call per poll instead of one per request. Unlike the per-request
    // walk, a failure may be reported before the other requests drain.
    transfer_status_t status;
    if (getBatchTransferStatus(engine, batch_id, &status) != 0) {
        return NIXL_ERR_BACKEND;
    }
    if (status.status == STATUS_PENDING || status.status == STATUS_WAITING) {
        return NIXL_IN_PROG;
    }
    return status.status == STATUS_COMPLETED ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
#else
    bool has_failed = false;
    for (size_t index = 0; index < request_count; ++index) {
        transfer_status_t status;
        int rc = getTransferStatus(engine, batch_id, index, &status);
        if (rc || status.status == STATUS_FAILED) {
            has_failed = true;
        } else if (status.status == STATUS_PENDING || status.status == STATUS_WAITING) {
            return NIXL_IN_PROG;
        }
    }
    return has_failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
#endif
}

// BATCH_CLEANUP_DEFERRED in newer transfer_engine_c.h: the engine took over a
// batch that is still busy and frees it once its requests drain.
constexpr int kBatchCleanupDeferred = 5;

void
freeBatch(transfer_engine_t engine, uint64_t batch_id) {
    const int rc = freeBatchID(engine, batch_id);
    if (rc != 0 && rc != kBatchCleanupDeferred) {
        NIXL_WARN << "Mooncake freeBatchID failed with " << rc << ", leaking batch " << batch_id;
    }
}

constexpr size_t kDefaultSplitBatchSize = 1024;

// A backend parameter takes precedence over its environment variable.
[[nodiscard]] size_t
getSizeParam(const nixl_b_params_t *params,
             const std::string &key,
             const std::string &env,
             size_t fallback) {
    if (const auto value = nixl::getBackendParamOptional<size_t>(params, key)) {
        return *value;
    }
    return nixl::config::getValueDefaulted<size_t>(env, fallback);
}

} // namespace

// Submits the child batches of one split postXfer in parallel.
class nixlMooncakeSubmitPool {
public:
    explicit nixlMooncakeSubmitPool(size_t num_threads) {
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] { worker(); });
        }
    }

    nixlMooncakeSubmitPool(const nixlMooncakeSubmitPool &) = delete;
    nixlMooncakeSubmitPool &
    operator=(const nixlMooncakeSubmitPool &) = delete;

    ~nixlMooncakeSubmitPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &thread : threads_) {
            thread.join();
        }
    }

    [[nodiscard]] size_t
    size() const {
        return threads_.size();
    }

    [[nodiscard]] std::future<int>
    submit(std::function<int()> fn) {
        std::packaged_task<int()> task(std::move(fn));
        std::future<int> result = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
        return result;
    }

private:
    void
    worker() {
        for (;;) {
            std::packaged_task<int()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::packaged_task<int()>> tasks_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

nixlMooncakeEngine::nixlMooncakeEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      local_agent_name_(init_params->localAgent) {
    const size_t num_threads =
        getSizeParam(init_params->customParams, "num_threads", "NIXL_MOONCAKE_NUM_THREADS", 0);
    split_batch_size_ = getSizeParam(init_params->customParams,
                                     "split_batch_size",
                                     "NIXL_MOONCAKE_SPLIT_BATCH_SIZE",
                                     kDefaultSplitBatchSize);
    if (split_batch_size_ == 0) {
        split_batch_size_ = kDefaultSplitBatchSize;
    }
    if (num_threads > 1) {
        submit_pool_ = std::make_unique<nixlMooncakeSubmitPool>(num_threads);
    }
    NIXL_INFO << "Mooncake submit threads=" << num_threads
              << " split_batch_size=" << split_batch_size_;
    const std::string segment_name = chooseIpAddress();
    engine_ = createTransferEngine("P2PHANDSHAKE", segment_name.c_str(), "", 0, true);
}

nixl_mem_list_t
nixlMooncakeEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

// Through parent destructor the unregister will be called.
nixlMooncakeEngine::~nixlMooncakeEngine() {
    submit_pool_.reset();
    destroyTransferEngine(engine_);
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
//
// Mooncake Transfer Engine exchanges metadata by itself without any explicit interface,
// and it does not need to connect remote agent before transferring data.
// Instead, getConnInfo() obtains the mapping between agent name and connect info
// (segment name in the context of Mooncake Transfer Engine).
// loadRemoteConnInfo() opens the segment, which implicitly retrieves metadata
// (such as QP numbers) of the remote agent.
nixl_status_t
nixlMooncakeEngine::connect(const std::string &remote_agent) {
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::disconnect(const std::string &remote_agent) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::getConnInfo(std::string &str) const {
    const static size_t kBufLen = 64;
    char buf_out[kBufLen];
    getLocalIpAndPort(engine_, buf_out, kBufLen);
    str = buf_out;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                       const std::string &remote_conn_info) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto segment_id = openSegment(engine_, remote_conn_info.c_str());
    if (segment_id < 0) return NIXL_ERR_BACKEND;
    connected_agents_[remote_agent].segment_id = segment_id;
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendMD : public nixlBackendMD {
    nixlMooncakeBackendMD(bool isPrivate) : nixlBackendMD(isPrivate) {}

    virtual ~nixlMooncakeBackendMD() {}

    void *addr;
    size_t length;
    int ref_cnt;
};

nixl_status_t
nixlMooncakeEngine::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mem_reg_info_.count(mem.addr)) {
        auto priv = mem_reg_info_[mem.addr];
        priv->ref_cnt++;
        out = priv;
        return NIXL_SUCCESS;
    }
    int err = registerLocalMemory(engine_, (void *)mem.addr, mem.len, "*", 1);
    if (err) return NIXL_ERR_BACKEND;
    auto priv = new nixlMooncakeBackendMD(true);
    priv->addr = (void *)mem.addr;
    priv->length = mem.len;
    priv->ref_cnt = 1;
    out = priv;
    mem_reg_info_[mem.addr] = priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::deregisterMem(nixlBackendMD *meta) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto priv = (nixlMooncakeBackendMD *)meta;
    priv->ref_cnt--;
    if (priv->ref_cnt) return NIXL_SUCCESS;
    int err = unregisterLocalMemory(engine_, priv->addr);
    mem_reg_info_.erase((uint64_t)priv->addr);
    delete priv;
    return err == 0 ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
//
// Mooncake Transfer Engine exchanges metadata by itself without any explicit interface,
// which is different from NIXL's paradigm.
// Therefore no metadata needs to be exposed to the outside.
nixl_status_t
nixlMooncakeEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = nullptr;
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::loadRemoteMD(const nixlBlobDesc &input,
                                 const nixl_mem_t &nixl_mem,
                                 const std::string &remote_agent,
                                 nixlBackendMD *&output) {
    output = nullptr;
    return NIXL_SUCCESS;
}

// TODO We purposely set this function as empty.
// Will be changed to follow NIXL's paradigm after refactoring Mooncake Transfer Engine.
nixl_status_t
nixlMooncakeEngine::unloadMD(nixlBackendMD *input) {
    return NIXL_SUCCESS;
}

struct nixlMooncakeBackendReqH : public nixlBackendReqH {
    nixlMooncakeBackendReqH() : nixlBackendReqH() {}

    virtual ~nixlMooncakeBackendReqH() {}

    struct Batch {
        uint64_t id = INVALID_BATCH;
        size_t request_count = 0;
    };

    // Batches of the last transfer: one, or one per chunk when postXfer split
    // it. Only allocated ids are kept, since the engine dereferences them.
    std::vector<Batch> batches;
    // A split transfer has no single batch to carry its notification, so
    // checkXfer sends it to {agent, message} once every batch completed.
    std::optional<std::pair<std::string, std::string>> pending_notif;
};

namespace {
void
releaseBatches(transfer_engine_t engine, nixlMooncakeBackendReqH *priv) {
    for (const auto &batch : priv->batches) {
        freeBatch(engine, batch.id);
    }
    priv->batches.clear();
    priv->pending_notif.reset();
}

// checkXfer leaves finished batches allocated so that the poll observing
// completion does not also destroy every task. Free them before the handle
// starts another transfer.
nixl_status_t
recycleBatches(transfer_engine_t engine, nixlMooncakeBackendReqH *priv) {
    for (const auto &batch : priv->batches) {
        if (queryBatchStatus(engine, batch.id, batch.request_count) == NIXL_IN_PROG) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
    }
    releaseBatches(engine, priv);
    return NIXL_SUCCESS;
}
} // namespace

nixl_status_t
nixlMooncakeEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    handle = new nixlMooncakeBackendReqH();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    int segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
    if (local.descCount() != remote.descCount()) return NIXL_ERR_INVALID_PARAM;
    const nixl_status_t recycle_status = recycleBatches(engine_, priv);
    if (recycle_status != NIXL_SUCCESS) {
        return recycle_status;
    }

    const size_t request_count = local.descCount();
    std::vector<transfer_request_t> request(request_count);
    for (size_t index = 0; index < request_count; ++index) {
        if (local[index].len != remote[index].len) return NIXL_ERR_INVALID_PARAM;
        request[index].opcode = (operation == NIXL_READ) ? OPCODE_READ : OPCODE_WRITE;
        request[index].source = (void *)local[index].addr;
        request[index].target_offset = remote[index].addr;
        request[index].length = local[index].len;
        request[index].target_id = segment_id;
    }

    const bool has_notif = opt_args && opt_args->hasNotif;
    size_t chunk_size = request_count;
    size_t num_chunks = 1;
    if (submit_pool_ && request_count >= split_batch_size_) {
        const size_t threads = submit_pool_->size();
        chunk_size = std::max((request_count + threads - 1) / threads, split_batch_size_);
        num_chunks = (request_count + chunk_size - 1) / chunk_size;
    }
    if (num_chunks > 1) {
        // Each chunk is an independent batch submitted by a pool thread. Wait
        // for all of them: they read from the request array.
        priv->batches.resize(num_chunks);
        std::vector<std::future<int>> results;
        results.reserve(num_chunks);
        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
            const size_t begin = chunk * chunk_size;
            const size_t count = std::min(chunk_size, request_count - begin);
            auto *batch = &priv->batches[chunk];
            auto *entries = request.data() + begin;
            results.push_back(submit_pool_->submit([engine = engine_, batch, entries, count] {
                batch->id = allocateBatchID(engine, count);
                if (batch->id == INVALID_BATCH) {
                    return -1;
                }
                const int rc = submitTransfer(engine, batch->id, entries, count);
                if (rc == 0) {
                    batch->request_count = count;
                }
                return rc;
            }));
        }
        bool failed = false;
        for (auto &result : results) {
            failed |= result.get() != 0;
        }
        std::erase_if(priv->batches, [](const auto &batch) { return batch.id == INVALID_BATCH; });
        if (failed) {
            return NIXL_ERR_BACKEND;
        }
        if (has_notif) {
            priv->pending_notif.emplace(remote_agent, opt_args->notifMsg);
        }
        NIXL_DEBUG << "Mooncake split " << request_count << " requests into " << num_chunks
                   << " batches";
        return NIXL_IN_PROG;
    }

    const uint64_t batch_id = allocateBatchID(engine_, request_count);
    if (batch_id == INVALID_BATCH) {
        return NIXL_ERR_BACKEND;
    }
    priv->batches.push_back({batch_id});
    int rc = 0;
    if (has_notif) {
        notify_msg_t notify_msg;
        notify_msg.name = const_cast<char *>(local_agent_name_.c_str());
        notify_msg.msg = const_cast<char *>(opt_args->notifMsg.c_str());
        rc = submitTransferWithNotify(engine_, batch_id, request.data(), request_count, notify_msg);
    } else {
        rc = submitTransfer(engine_, batch_id, request.data(), request_count);
    }
    if (rc) return NIXL_ERR_BACKEND;
    priv->batches.back().request_count = request_count;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlMooncakeEngine::checkXfer(nixlBackendReqH *handle) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    // A failed batch is reported once no other batch is still in flight.
    nixl_status_t status = NIXL_SUCCESS;
    for (const auto &batch : priv->batches) {
        const nixl_status_t batch_status = queryBatchStatus(engine_, batch.id, batch.request_count);
        if (batch_status == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }
        if (batch_status != NIXL_SUCCESS) {
            status = batch_status;
        }
    }
    if (status == NIXL_SUCCESS && priv->pending_notif) {
        const auto [agent, msg] = std::move(*priv->pending_notif);
        priv->pending_notif.reset();
        if (genNotif(agent, msg) != NIXL_SUCCESS) {
            return NIXL_ERR_BACKEND;
        }
    }
    return status;
}

nixl_status_t
nixlMooncakeEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto priv = (nixlMooncakeBackendReqH *)handle;
    releaseBatches(engine_, priv);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::getNotifs(notif_list_t &notif_list) {
    if (notif_list.size() != 0) return NIXL_ERR_INVALID_PARAM;
    int size = 0;
    notify_msg_t *notify_msgs = getNotifsFromEngine(engine_, &size);
    for (int i = 0; i < size; i++) {
        notif_list.push_back(std::make_pair(notify_msgs[i].name, notify_msgs[i].msg));
    }
    freeNotifsMsgBuf(notify_msgs, size);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMooncakeEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    int segment_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto agent = connected_agents_.find(remote_agent);
        if (agent == connected_agents_.end()) return NIXL_ERR_INVALID_PARAM;
        segment_id = agent->second.segment_id;
    }
    notify_msg_t notify_msg;
    notify_msg.name = const_cast<char *>(local_agent_name_.c_str());
    notify_msg.msg = const_cast<char *>(msg.c_str());
    int ret = genNotifyInEngine(engine_, segment_id, notify_msg);
    return nixl_status_t(ret);
}
