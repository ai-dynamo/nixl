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

#include "wqskv_backend.h"

#include <sys/uio.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <absl/strings/str_format.h>

#include "common/nixl_log.h"
#include "kv_interface.h"
#include "nixl_types.h"
#include "wqskv_helpers.h"

namespace {

// Process-wide guard for wds_kvcache_init: vendor lib only tolerates one init
// per process; subsequent createBackend calls must reuse it.
std::once_flag g_init_once;
std::atomic<int> g_init_rc{0};

class nixlWqskvBackendReqH : public nixlBackendReqH {
public:
    // Per-descriptor scratch space whose lifetime must outlive the vendor
    // callback. The vendor takes key/iovec by const reference; we keep them
    // alive in this vector until releaseReqH (after pending hits zero).
    struct DescState {
        std::string key;
        std::vector<iovec> ioVec;
    };

    std::vector<DescState> descStates;
    std::atomic<int> pending{0};
    // 0 == no error; first non-zero rc wins via compare_exchange_strong.
    std::atomic<int> first_error{0};
    std::mutex mu;
    std::condition_variable cv;

    // Per-iteration key override stashed at prepXfer time. The agent only
    // propagates customParam through createXferReq->prepXfer, never through
    // postXferReq->postXfer, so we must capture it here for postXfer to use.
    bool have_per_iter_keys{false};
    std::vector<std::string> per_iter_keys;
};

class nixlWQSKVMetadata : public nixlBackendMD {
public:
    nixlWQSKVMetadata(uint64_t dev_id, std::string k)
        : nixlBackendMD(true),
          devId(dev_id),
          key(std::move(k)) {}

    ~nixlWQSKVMetadata() override = default;

    uint64_t devId;
    std::string key;
};

void
wqskv_xfer_cb(int rc, void *arg) {
    auto *req = static_cast<nixlWqskvBackendReqH *>(arg);
    if (rc != 0) {
        int expected = 0;
        req->first_error.compare_exchange_strong(expected, rc);
    }
    int prev = req->pending.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        std::lock_guard<std::mutex> lk(req->mu);
        req->cv.notify_all();
    }
}

bool
resolveKey(const nixlMetaDesc &remote_desc,
           const std::unordered_map<uint64_t, std::string> &devIdToKey,
           std::string &out_key) {
    if (remote_desc.metadataP != nullptr) {
        auto *md = dynamic_cast<nixlWQSKVMetadata *>(remote_desc.metadataP);
        if (md != nullptr) {
            out_key = md->key;
            return true;
        }
    }
    auto it = devIdToKey.find(remote_desc.devId);
    if (it == devIdToKey.end()) {
        return false;
    }
    out_key = it->second;
    return true;
}

} // namespace

nixlWQSKVEngine::nixlWQSKVEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    localAgent = init_params->localAgent;

    const std::string conf_path = wqskv::resolveConfigPath(init_params);
    if (conf_path.empty()) {
        NIXL_ERROR << "WQSKV: no config path: customParams[\"config_path\"] not set "
                      "and WDS_BACKEND_CONFIG_PATH env var not set";
        initErr = true;
        return;
    }

    std::call_once(g_init_once, [&]() {
        KVCacheOptions opts;
        if (!wqskv::loadKVCacheOptionsFromJson(conf_path, opts)) {
            g_init_rc.store(-EINVAL, std::memory_order_release);
            return;
        }
        int rc = wds_kvcache_init(opts);
        g_init_rc.store(rc, std::memory_order_release);
        if (rc != 0) {
            NIXL_ERROR << "WQSKV: wds_kvcache_init failed, rc=" << rc;
        } else {
            NIXL_INFO << "WQSKV: wds_kvcache_init succeeded with config " << conf_path;
        }
    });

    if (g_init_rc.load(std::memory_order_acquire) != 0) {
        NIXL_ERROR << "WQSKV: vendor init previously failed (rc=" << g_init_rc.load()
                   << "), backend unusable";
        initErr = true;
        return;
    }

    NIXL_INFO << "WQSKV backend initialized for agent=" << localAgent;
}

nixl_status_t
nixlWQSKVEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    NIXL_INFO << "WQSKV registerMem: type=" << nixl_mem << ", devId=" << mem.devId
              << ", metaInfo=" << (mem.metaInfo.empty() ? "<empty>" : mem.metaInfo);

    auto supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        NIXL_ERROR << "WQSKV registerMem: unsupported memory type " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    std::string key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
    auto md = std::make_unique<nixlWQSKVMetadata>(mem.devId, key);
    devIdToKey_[mem.devId] = key;
    NIXL_INFO << "WQSKV registerMem: registered devId=" << mem.devId << " -> key=" << key;
    out = md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::deregisterMem(nixlBackendMD *meta) {
    auto *md = static_cast<nixlWQSKVMetadata *>(meta);
    if (md != nullptr) {
        NIXL_INFO << "WQSKV deregisterMem: removing devId=" << md->devId << ", key=" << md->key;
        devIdToKey_.erase(md->devId);
        std::unique_ptr<nixlWQSKVMetadata> ptr(md);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // No vendor query of arbitrary keys without a callback round-trip; return a
    // best-effort answer based on what we registered locally. Keys put by other
    // processes are reported as missing here -- callers needing strong existence
    // semantics should issue a READ and check the result.
    resp.reserve(descs.descCount());
    for (auto &desc : descs) {
        std::string key = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
        const bool exists = devIdToKey_.find(desc.devId) != devIdToKey_.end();
        resp.emplace_back(exists ? nixl_query_resp_t{nixl_b_params_t{}} : std::nullopt);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    (void)remote_agent;
    NIXL_INFO << "WQSKV prepXfer: op=" << (operation == NIXL_WRITE ? "WRITE" : "READ")
              << ", local_count=" << local.descCount() << ", remote_count=" << remote.descCount();

    if (!wqskv::isValidPrepXferParams(operation, local, remote)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "WQSKV prepXfer: local/remote desc count mismatch (" << local.descCount()
                   << " vs " << remote.descCount() << ")";
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlWqskvBackendReqH>();
    // Capture per-iter key override here -- nixlAgent::createXferReq is the
    // only hook where customParam is propagated to the backend; postXferReq
    // builds a fresh opt_b_args_t and never copies customParam through.
    if (opt_args != nullptr) {
        req_h->have_per_iter_keys = wqskv::parseCustomParamKeys(
            opt_args->customParam, local.descCount(), req_h->per_iter_keys);
    }
    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    (void)remote_agent;
    (void)opt_args;

    auto *req = static_cast<nixlWqskvBackendReqH *>(handle);
    const int n = local.descCount();
    NIXL_INFO << "WQSKV postXfer: starting "
              << (operation == NIXL_WRITE ? "WRITE (PUT)" : "READ (GET)") << " with " << n
              << " descriptor(s)";

    // Stage all per-descriptor state and arm the pending counter BEFORE issuing
    // any vendor call -- once dispatched the callback may fire on another thread
    // immediately, and we must not race a decrement against an unset counter.
    req->descStates.assign(n, nixlWqskvBackendReqH::DescState{});
    req->first_error.store(0, std::memory_order_release);
    req->pending.store(n, std::memory_order_release);

    // Per-iter keys (if any) were stashed onto req at prepXfer time. Copy
    // (not move) since postXfer can be called multiple times on the same
    // handle (recreate_per_iteration=false path) and the keys must persist.
    const bool have_per_iter_keys =
        req->have_per_iter_keys && static_cast<int>(req->per_iter_keys.size()) == n;

    for (int i = 0; i < n; ++i) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];
        std::string key;
        if (have_per_iter_keys) {
            key = req->per_iter_keys[i];
        } else if (!resolveKey(remote_desc, devIdToKey_, key)) {
            NIXL_ERROR << "WQSKV postXfer: no key for devId=" << remote_desc.devId;
            // Drain remaining slots so checkXfer/releaseReqH can finish. Treat
            // unresolved key as a synchronous error for this slot.
            int expected = 0;
            req->first_error.compare_exchange_strong(expected, -EINVAL);
            int prev = req->pending.fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(req->mu);
                req->cv.notify_all();
            }
            continue;
        }

        auto &state = req->descStates[i];
        state.key = std::move(key);
        state.ioVec = {iovec{reinterpret_cast<void *>(local_desc.addr), local_desc.len}};

        int rc = 0;
        if (operation == NIXL_WRITE) {
            rc = wds_kvcache_put(state.key, state.ioVec, wqskv_xfer_cb, req);
        } else {
            rc = wds_kvcache_get_vec(state.key, state.ioVec, local_desc.len, wqskv_xfer_cb, req);
        }

        if (rc != 0) {
            // Vendor signaled synchronous failure -> callback will not fire.
            // Compensate the pending counter ourselves so checkXfer can complete.
            NIXL_ERROR << "WQSKV postXfer: vendor " << (operation == NIXL_WRITE ? "put" : "get_vec")
                       << " returned rc=" << rc << " for key=" << state.key;
            int expected = 0;
            req->first_error.compare_exchange_strong(expected, rc);
            int prev = req->pending.fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(req->mu);
                req->cv.notify_all();
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlWqskvBackendReqH *>(handle);
    if (req->pending.load(std::memory_order_acquire) > 0) {
        return NIXL_IN_PROG;
    }
    const int err = req->first_error.load(std::memory_order_acquire);
    if (err != 0) {
        NIXL_ERROR << "WQSKV checkXfer: completed with error rc=" << err;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *req = static_cast<nixlWqskvBackendReqH *>(handle);
    {
        std::unique_lock<std::mutex> lk(req->mu);
        req->cv.wait(lk, [&] { return req->pending.load(std::memory_order_acquire) == 0; });
    }
    delete req;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = input;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::connect(const std::string &remote_agent) {
    (void)remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::disconnect(const std::string &remote_agent) {
    (void)remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWQSKVEngine::unloadMD(nixlBackendMD *input) {
    (void)input;
    return NIXL_SUCCESS;
}
