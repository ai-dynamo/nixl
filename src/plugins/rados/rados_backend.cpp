/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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
#include "rados_backend.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "common/nixl_log.h"

namespace {

bool
checkRadosResult(const char *operation, int result) {
    if (result == 0) {
        return true;
    }
    NIXL_ERROR << "RADOS " << operation << " failed with error " << result;
    return false;
}

bool
applyCephOverrides(nixl::rados::cApi &capi,
                   rados_t cluster,
                   const nixl::rados::ceph_options_t &overrides) {
    for (const auto &[option, value] : overrides) {
        const std::string operation = std::string("rados_conf_set(") + option + ')';
        if (!checkRadosResult(operation.c_str(),
                              capi.conf_set(cluster, option.c_str(), value.c_str()))) {
            return false;
        }
    }
    return true;
}

// librados yields delivered bytes for an AIO read but only 0/-errno for a write,
// so the mapped NIXL status must key off direction and requested length.
nixl_status_t
radosResultToNixl(nixl_xfer_op_t operation, int result, uint64_t requested_len) {
    if (operation == NIXL_READ && result == -ENOENT) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (result < 0) {
        NIXL_ERROR << "RADOS object I/O failed with error " << result;
        return NIXL_ERR_BACKEND;
    }
    if (operation == NIXL_READ) {
        if (static_cast<uint64_t>(result) == requested_len) {
            return NIXL_SUCCESS;
        }
        if (static_cast<uint64_t>(result) < requested_len) {
            return NIXL_ERR_MISMATCH;
        }
        NIXL_ERROR << "RADOS read reported " << result << " delivered bytes for a " << requested_len
                   << " byte request";
        return NIXL_ERR_BACKEND;
    }
    if (result == 0) {
        return NIXL_SUCCESS;
    }
    NIXL_ERROR << "RADOS write reported unexpected positive result " << result;
    return NIXL_ERR_BACKEND;
}

// Initialization-only, exception-safe owner of the cluster/ioctx handles:
// take_*()/commit_view() hand ownership to the engine on success, while any
// failed path unwinds here as view -> ioctx -> cluster.
class bootstrapResources {
public:
    explicit bootstrapResources(nixl::rados::cApi &capi) : capi_(capi) {}

    bootstrapResources(const bootstrapResources &) = delete;
    bootstrapResources &
    operator=(const bootstrapResources &) = delete;

    ~bootstrapResources() {
        if (view_held_) {
            capi_.ioCtxClose(view_);
        }
        if (ioctx_ != nullptr) {
            capi_.ioctx_destroy(ioctx_);
        }
        if (cluster_ != nullptr) {
            capi_.shutdown(cluster_);
        }
    }

    int
    create_cluster(const char *name) {
        return capi_.create2(&cluster_, nullptr, name, 0);
    }

    int
    conf_read_file(const char *path) {
        return capi_.conf_read_file(cluster_, path);
    }

    int
    connect() {
        return capi_.connect(cluster_);
    }

    int
    create_ioctx(const char *pool) {
        return capi_.ioctx_create(cluster_, pool, &ioctx_);
    }

    void
    set_namespace(const std::string &ns) {
        capi_.ioctx_set_namespace(ioctx_, ns.c_str());
    }

    void
    create_view(librados::IoCtx *ctx) {
        view_ = ctx;
        view_held_ = true;
        capi_.ioctxFromRaw(ioctx_, view_);
    }

    void
    commit_view() {
        view_ = nullptr;
        view_held_ = false;
    }

    rados_t
    cluster() const {
        return cluster_;
    }

    rados_ioctx_t
    take_ioctx() {
        return std::exchange(ioctx_, nullptr);
    }

    rados_t
    take_cluster() {
        return std::exchange(cluster_, nullptr);
    }

private:
    nixl::rados::cApi &capi_;
    rados_t cluster_ = nullptr;
    rados_ioctx_t ioctx_ = nullptr;
    librados::IoCtx *view_ = nullptr;
    bool view_held_ = false;
};

// Bring up one rados client (create2 -> conf -> overrides -> connect -> pool
// ioctx -> optional namespace -> non-owning view) from the parsed config. On
// success the raw handles move into the out-params for engine ownership; on
// any failure everything acquired so far unwinds here and nothing is taken.
bool
bootstrapClient(nixl::rados::cApi &capi,
                const nixl::rados::backendConfig &config,
                librados::IoCtx *view,
                rados_t &out_cluster,
                rados_ioctx_t &out_ioctx) {
    bootstrapResources resources(capi);
    if (!checkRadosResult("rados_create2", resources.create_cluster(config.name.c_str()))) {
        return false;
    }
    if (!checkRadosResult("rados_conf_read_file", resources.conf_read_file(config.conf.c_str())) ||
        !applyCephOverrides(capi, resources.cluster(), config.ceph_options) ||
        !checkRadosResult("rados_connect", resources.connect())) {
        return false;
    }
    if (!checkRadosResult("rados_ioctx_create", resources.create_ioctx(config.pool.c_str()))) {
        return false;
    }
    if (config.ns) {
        resources.set_namespace(*config.ns);
    }
    resources.create_view(view);
    out_ioctx = resources.take_ioctx();
    out_cluster = resources.take_cluster();
    resources.commit_view();
    return true;
}

// Production seam: a plain per-call timeout for the configured budget;
// scheduler delivery delay lies outside it (see the plugin README).
class thisThreadStatusWaitTimer final : public nixl::rados::statusWaitTimer {
public:
    void
    sleep(uint64_t budget_us) override {
        std::this_thread::sleep_for(std::chrono::microseconds(budget_us));
    }
};

} // namespace

class directRequestState {
public:
    // Only forward transitions are legal: PREPARED -> ACTIVE -> TERMINAL (a terminal
    // round may repost to ACTIVE); release/teardown demotes to drain-only ABANDONED.
    enum class phase_t { PREPARED, ACTIVE, TERMINAL, ABANDONED };

    struct cRead {
        rados_completion_t handle = nullptr;
    };

    struct cppWrite {
        librados::AioCompletion *handle = nullptr;
    };

    using owned_completion_t = std::variant<std::monostate, cRead, cppWrite>;

    struct requestPair {
        std::string oid;
        uintptr_t dram = 0;
        uint64_t offset = 0;
        uint64_t len = 0;
        int result = 0;
        // Write-only, non-owning view of snapshotted caller DRAM; librados keeps
        // borrowing it until retirement, and repost rebuilds it from the pointer.
        librados::bufferlist payload;
        owned_completion_t owned;
    };

    explicit directRequestState(nixl_xfer_op_t operation) : operation_(operation) {}

    ~directRequestState() {
        for (const requestPair &pair : pairs_) {
            if (!std::holds_alternative<std::monostate>(pair.owned)) {
                NIXL_ERROR << "RADOS request state destroyed with an unreleased "
                              "completion on object '"
                           << pair.oid << "'";
            }
        }
    }

    directRequestState(const directRequestState &) = delete;
    directRequestState &
    operator=(const directRequestState &) = delete;

    bool
    reservePairs(size_t count) {
        try {
            pairs_.resize(count);
        }
        catch (const std::bad_alloc &) {
            return false;
        }
        return true;
    }

    nixl_xfer_op_t
    operation() const {
        return operation_;
    }

    std::vector<requestPair> &
    pairs() {
        return pairs_;
    }

    const std::vector<requestPair> &
    pairs() const {
        return pairs_;
    }

    void
    resetForRepostLocked() {
        for (requestPair &pair : pairs_) {
            pair.owned = owned_completion_t{};
            pair.payload.clear();
            pair.result = 0;
        }
        terminalStatus = NIXL_SUCCESS;
    }

    std::mutex mtx;
    phase_t phase = phase_t::PREPARED;
    nixl_status_t terminalStatus = NIXL_SUCCESS;

private:
    const nixl_xfer_op_t operation_;
    std::vector<requestPair> pairs_;
};

namespace {

bool
pairUnproven(nixl::rados::cApi &capi, const directRequestState::requestPair &pair) {
    if (const auto *read = std::get_if<directRequestState::cRead>(&pair.owned)) {
        return capi.cIsCompleteAndCb(read->handle) == 0;
    }
    if (const auto *write = std::get_if<directRequestState::cppWrite>(&pair.owned)) {
        return capi.cppIsCompleteAndCb(write->handle) == 0;
    }
    return false;
}

bool
pairOwned(const directRequestState::requestPair &pair) {
    return !std::holds_alternative<std::monostate>(pair.owned);
}

// wait=false runs only after the caller proved callback retirement, so it stays
// nonblocking; an empty slot (never posted or already harvested) yields its cached result.
int
retirePair(nixl::rados::cApi &capi, directRequestState::requestPair &pair, bool wait) {
    if (auto *read = std::get_if<directRequestState::cRead>(&pair.owned)) {
        if (wait) {
            capi.cWaitForCompleteAndCb(read->handle);
        }
        const int result = capi.cGetReturnValue(read->handle);
        capi.cRelease(read->handle);
        pair.owned = std::monostate{};
        return result;
    }
    if (auto *write = std::get_if<directRequestState::cppWrite>(&pair.owned)) {
        if (wait) {
            capi.cppWaitForCompleteAndCb(write->handle);
        }
        const int result = capi.cppGetReturnValue(write->handle);
        capi.cppRelease(write->handle);
        pair.payload.clear();
        pair.owned = std::monostate{};
        return result;
    }
    return pair.result;
}

void
logRetiredResult(const directRequestState &state,
                 const directRequestState::requestPair &pair,
                 int result) {
    if (result >= 0) {
        return;
    }
    NIXL_ERROR << "RADOS abandoned " << (state.operation() == NIXL_READ ? "read of" : "write to")
               << " object '" << pair.oid << "' at offset " << pair.offset << " failed with error "
               << result;
}

constexpr size_t kAioStatWindow = 256;

struct statSlot {
    std::string oid;
    std::optional<uint64_t> size;
    std::optional<int> prval;
    std::optional<librados::bufferlist> merged;
    librados::AioCompletion *comp = nullptr;
    bool submitted = false;
    int submit_error = 0;
};

// Backstop releasing every still-held completion exactly once on unwinding:
// accepted work through the retirement wait, never-accepted work directly.
// Drained slots are nulled before this runs, so the happy path is a no-op.
class statWindow {
public:
    statWindow(nixl::rados::cApi &capi, size_t width) : capi_(capi), width_(width) {}

    statWindow(const statWindow &) = delete;
    statWindow &
    operator=(const statWindow &) = delete;

    ~statWindow() {
        for (size_t i = 0; i < width_; ++i) {
            statSlot &slot = slots_[i];
            if (slot.comp == nullptr) {
                continue;
            }
            if (slot.submitted) {
                capi_.cppWaitForCompleteAndCb(slot.comp);
            }
            capi_.cppRelease(slot.comp);
            slot.comp = nullptr;
        }
    }

    statSlot &
    operator[](size_t i) {
        return slots_[i];
    }

private:
    nixl::rados::cApi &capi_;
    const size_t width_;
    std::array<statSlot, kAioStatWindow> slots_;
};

} // namespace

// Opaque token; its real state lives in the engine's handles_, so registry lookup
// is what rejects null, unknown, and already-released handles.
class nixlRadosBackendReqH final : public nixlBackendReqH {};

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params)
    : nixlRadosEngine(init_params, std::make_unique<nixl::rados::libRados>()) {}

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params,
                                 std::unique_ptr<nixl::rados::cApi> capi,
                                 std::unique_ptr<nixl::rados::statusWaitTimer> status_wait_timer)
    : nixlBackendEngine(init_params),
      capi_(std::move(capi)) {
    if (capi_ == nullptr) {
        NIXL_ERROR << "RADOS engine requires a non-null librados seam";
        initErr = true;
        return;
    }
    const auto config = nixl::rados::parseCustomParams(getCustomParams());
    if (!config) {
        NIXL_ERROR << "RADOS custom parameter validation failed: 'conf', 'pool', and "
                      "'name' must be non-empty, 'status_wait_us' must be a decimal in "
                      "[0, 1000000], 'namespace' when present must be non-empty, NUL-free, "
                      "and not the all-namespaces sentinel, and every other parameter must be "
                      "'ceph.<option>'";
        initErr = true;
        return;
    }
    // Only a positive budget adopts the injected timer or creates the
    // production one; with a zero budget an injected timer is dropped.
    statusWaitBudgetUs_ = config->status_wait_us;
    if (statusWaitBudgetUs_ > 0) {
        statusWaitTimer_ = status_wait_timer ? std::move(status_wait_timer) :
                                               std::make_unique<thisThreadStatusWaitTimer>();
    }
    initialize(*config);
}

void
nixlRadosEngine::initialize(const nixl::rados::backendConfig &config) {
    if (!bootstrapClient(*capi_, config, &cppCtx_, cluster_, ioctx_)) {
        initErr = true;
        return;
    }
    clusterCreated_ = true;
    ioctxOpen_ = true;
    cppCtxOpen_ = true;
    if (!bootstrapClient(*capi_, config, &metaCppCtx_, metaCluster_, metaIoctx_)) {
        NIXL_ERROR << "RADOS metadata stat client initialization failed";
        initErr = true;
        return;
    }
    metaClientOpen_ = true;
}

nixlRadosEngine::~nixlRadosEngine() {
    std::unique_lock alock(activeMutex_);
    shuttingDown_ = true;
    while (!handles_.empty()) {
        const auto it = handles_.begin();
        nixlBackendReqH *handle = it->first;
        std::shared_ptr<directRequestState> state = std::move(it->second);
        // Erase before draining so each abandoned state is claimed and drained
        // exactly once, then unlock: the drain blocks on completions.
        handles_.erase(it);
        alock.unlock();
        drainRequestState(state);
        delete static_cast<nixlRadosBackendReqH *>(handle);
        alock.lock();
    }
    alock.unlock();
    if (metaClientOpen_) {
        capi_->ioCtxClose(&metaCppCtx_);
        capi_->ioctx_destroy(metaIoctx_);
        capi_->shutdown(metaCluster_);
        metaClientOpen_ = false;
    }
    if (cppCtxOpen_) {
        capi_->ioCtxClose(&cppCtx_);
        cppCtxOpen_ = false;
    }
    if (ioctxOpen_) {
        capi_->ioctx_destroy(ioctx_);
        ioctxOpen_ = false;
    }
    if (clusterCreated_) {
        capi_->shutdown(cluster_);
        clusterCreated_ = false;
    }
}

nixl_status_t
nixlRadosEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    out = nullptr;
    if (nixl_mem != DRAM_SEG && nixl_mem != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    if (nixl_mem == DRAM_SEG) {
        // Metadata-free by design: out stays null deliberately because the
        // backend never pins, tracks, or validates caller DRAM.
        return NIXL_SUCCESS;
    }
    if (mem.metaInfo.find('\0') != std::string::npos) {
        NIXL_ERROR << "RADOS OBJ_SEG metaInfo contains an embedded NUL byte";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::string object_key;
    std::unique_ptr<nixlRadosObjectMD> md;
    try {
        object_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
        md = std::make_unique<nixlRadosObjectMD>(mem.devId, object_key);
        std::lock_guard lock(objectsMutex_);
        if (objects_.count(mem.devId) != 0) {
            NIXL_ERROR << "RADOS OBJ_SEG devId " << mem.devId << " is already registered";
            return NIXL_ERR_INVALID_PARAM;
        }
        objects_.emplace(mem.devId, object_key);
    }
    catch (const std::bad_alloc &) {
        NIXL_ERROR << "RADOS backend failed to allocate registration metadata";
        return NIXL_ERR_BACKEND;
    }
    out = md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosEngine::deregisterMem(nixlBackendMD *meta) {
    if (meta == nullptr) {
        return NIXL_SUCCESS;
    }
    const std::unique_ptr<nixlRadosObjectMD> md(static_cast<nixlRadosObjectMD *>(meta));
    std::lock_guard lock(objectsMutex_);
    const auto it = objects_.find(md->devId);
    if (it != objects_.end() && it->second == md->objectKey) {
        objects_.erase(it);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    resp.assign(static_cast<size_t>(descs.descCount()), std::nullopt);
    if (descs.getType() != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    for (int i = 0; i < descs.descCount(); ++i) {
        if (descs[i].metaInfo.find('\0') != std::string::npos) {
            NIXL_ERROR << "RADOS query rejected an OBJ_SEG descriptor whose metaInfo "
                          "contains an embedded NUL byte";
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    const size_t count = static_cast<size_t>(descs.descCount());
    bool stat_failed = false;
    for (size_t base = 0; base < count; base += kAioStatWindow) {
        const size_t width = std::min(kAioStatWindow, count - base);
        statWindow window(*capi_, width);
        for (size_t i = 0; i < width; ++i) {
            const nixlBlobDesc &desc = descs[static_cast<int>(base + i)];
            statSlot &slot = window[i];
            slot.oid = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
            const int create_ret = capi_->cppCreateCompletion(&slot.comp);
            if (create_ret != 0 || slot.comp == nullptr) {
                slot.comp = nullptr;
                slot.submit_error = create_ret < 0 ? create_ret : -ENOMEM;
                continue;
            }
            const int submit_ret = capi_->cppAioStat(
                &metaCppCtx_, slot.oid.c_str(), slot.comp, &slot.size, &slot.prval, &slot.merged);
            if (submit_ret != 0) {
                capi_->cppRelease(slot.comp);
                slot.comp = nullptr;
                slot.submit_error = submit_ret < 0 ? submit_ret : -EIO;
                continue;
            }
            slot.submitted = true;
        }
        for (size_t i = 0; i < width; ++i) {
            statSlot &slot = window[i];
            if (slot.submit_error != 0) {
                NIXL_ERROR << "RADOS query failed to submit an async stat for object '" << slot.oid
                           << "' with error " << slot.submit_error;
                stat_failed = true;
                continue;
            }
            capi_->cppWaitForCompleteAndCb(slot.comp);
            const int prval = *slot.prval;
            const int aggregate = capi_->cppGetReturnValue(slot.comp);
            capi_->cppRelease(slot.comp);
            slot.comp = nullptr;
            if (prval == 0 && aggregate == 0) {
                resp[base + i] = nixl_b_params_t{{"size", std::to_string(*slot.size)}};
                continue;
            }
            // Absent stat objects surface -ENOENT in the completion return value while
            // librados fills the untouched per-op slot with -EIO.
            if (aggregate == -ENOENT) {
                continue;
            }
            const int fault = aggregate != 0 ? aggregate : prval;
            NIXL_ERROR << "RADOS query failed to stat object '" << slot.oid << "' with error "
                       << fault;
            stat_failed = true;
        }
    }
    return stat_failed ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

std::shared_ptr<directRequestState>
nixlRadosEngine::lookupActiveLocked(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        return {};
    }
    const auto it = handles_.find(handle);
    return it == handles_.end() ? std::shared_ptr<directRequestState>{} : it->second;
}

void
nixlRadosEngine::drainRequestState(const std::shared_ptr<directRequestState> &state) const {
    std::lock_guard slock(state->mtx);
    for (auto &pair : state->pairs()) {
        if (!pairOwned(pair)) {
            continue;
        }
        const int result = retirePair(*capi_, pair, true);
        logRetiredResult(*state, pair, result);
    }
    state->phase = directRequestState::phase_t::ABANDONED;
}

nixl_status_t
nixlRadosEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *) const {
    handle = nullptr;
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    if ((operation != NIXL_WRITE && operation != NIXL_READ) || local.getType() != DRAM_SEG ||
        remote.getType() != OBJ_SEG || remote_agent != localAgent) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const int count = local.descCount();
    if (count == 0 || count != remote.descCount()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    std::shared_ptr<directRequestState> state;
    try {
        state = std::make_shared<directRequestState>(operation);
    }
    catch (const std::bad_alloc &) {
        return NIXL_ERR_BACKEND;
    }
    if (!state->reservePairs(static_cast<size_t>(count))) {
        NIXL_ERROR << "RADOS backend failed to allocate request pairs";
        return NIXL_ERR_BACKEND;
    }

    try {
        std::lock_guard lock(objectsMutex_);
        for (int i = 0; i < count; ++i) {
            const auto &local_desc = local[i];
            const auto &remote_desc = remote[i];
            if (local_desc.len == 0 || local_desc.len != remote_desc.len) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (local_desc.len > static_cast<size_t>(INT_MAX)) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (local_desc.addr == 0 ||
                local_desc.addr > std::numeric_limits<uintptr_t>::max() - local_desc.len) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (remote_desc.addr > std::numeric_limits<uint64_t>::max() - remote_desc.len) {
                return NIXL_ERR_INVALID_PARAM;
            }
            const auto key = objects_.find(remote_desc.devId);
            if (key == objects_.end()) {
                NIXL_ERROR << "RADOS OBJ_SEG devId " << remote_desc.devId
                           << " is not registered for transfer";
                return NIXL_ERR_INVALID_PARAM;
            }
            auto &pair = state->pairs()[i];
            pair.oid = key->second;
            pair.dram = local_desc.addr;
            pair.offset = remote_desc.addr;
            pair.len = local_desc.len;
        }

        if (operation == NIXL_WRITE) {
            for (int i = 0; i < count; ++i) {
                const auto &a = state->pairs()[i];
                for (int j = i + 1; j < count; ++j) {
                    const auto &b = state->pairs()[j];
                    const bool overlaps = a.oid == b.oid && a.offset < b.offset + b.len &&
                        b.offset < a.offset + a.len;
                    if (overlaps) {
                        NIXL_ERROR << "RADOS request contains overlapping writes to object '"
                                   << a.oid << "'";
                        return NIXL_ERR_INVALID_PARAM;
                    }
                }
            }
        }
    }
    catch (const std::bad_alloc &) {
        NIXL_ERROR << "RADOS backend failed to allocate request state";
        return NIXL_ERR_BACKEND;
    }

    std::unique_ptr<nixlRadosBackendReqH> req;
    try {
        req = std::make_unique<nixlRadosBackendReqH>();
        std::lock_guard alock(activeMutex_);
        if (shuttingDown_) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        handles_.emplace(req.get(), std::move(state));
    }
    catch (const std::bad_alloc &) {
        return NIXL_ERR_BACKEND;
    }
    handle = req.release();
    return NIXL_SUCCESS;
}

// The operation/descriptor/agent arguments are ignored on purpose: prepXfer already
// snapshotted them into the request state, which alone drives this (re)submission.
nixl_status_t
nixlRadosEngine::postXfer(const nixl_xfer_op_t &,
                          const nixl_meta_dlist_t &,
                          const nixl_meta_dlist_t &,
                          const std::string &,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *) const {
    if (initErr) {
        return NIXL_ERR_BACKEND;
    }
    std::shared_ptr<directRequestState> state;
    {
        std::lock_guard alock(activeMutex_);
        state = lookupActiveLocked(handle);
        if (state == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (shuttingDown_) {
            return NIXL_ERR_NOT_ALLOWED;
        }
    }

    std::lock_guard slock(state->mtx);
    if (shuttingDown_) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    switch (state->phase) {
    case directRequestState::phase_t::PREPARED:
        break;
    case directRequestState::phase_t::TERMINAL:
        state->resetForRepostLocked();
        break;
    case directRequestState::phase_t::ACTIVE:
        return NIXL_IN_PROG;
    case directRequestState::phase_t::ABANDONED:
        return NIXL_ERR_INVALID_PARAM;
    }

    const bool is_read = state->operation() == NIXL_READ;
    for (auto &pair : state->pairs()) {
        if (is_read) {
            rados_completion_t completion = nullptr;
            bool completion_held = false;
            try {
                const int create_ret = capi_->cCreateCompletion(&completion);
                if (create_ret != 0 || completion == nullptr) {
                    pair.result = create_ret < 0 ? create_ret : -ENOMEM;
                    NIXL_ERROR << "RADOS failed to create a C completion for object '" << pair.oid
                               << "'; submitting no further pairs";
                    break;
                }
                completion_held = true;
                const int ret = capi_->cRead(ioctx_,
                                             pair.oid.c_str(),
                                             completion,
                                             reinterpret_cast<char *>(pair.dram),
                                             pair.len,
                                             pair.offset);
                if (ret != 0) {
                    capi_->cRelease(completion);
                    completion_held = false;
                    pair.result = ret < 0 ? ret : -EIO;
                    NIXL_ERROR << "RADOS failed to submit a C read from object '" << pair.oid
                               << "' with error " << ret << "; submitting no further pairs";
                    break;
                }
                pair.owned = directRequestState::cRead{completion};
                completion_held = false;
            }
            catch (const std::bad_alloc &) {
                if (completion_held) {
                    capi_->cRelease(completion);
                }
                pair.result = -ENOMEM;
                NIXL_ERROR << "RADOS C read allocation failed for object '" << pair.oid
                           << "'; submitting no further pairs";
                break;
            }
        } else {
            librados::AioCompletion *completion = nullptr;
            bool completion_held = false;
            try {
                pair.payload = librados::bufferlist::static_from_mem(
                    reinterpret_cast<char *>(pair.dram), pair.len);
                const int create_ret = capi_->cppCreateCompletion(&completion);
                if (create_ret != 0 || completion == nullptr) {
                    pair.payload.clear();
                    pair.result = create_ret < 0 ? create_ret : -ENOMEM;
                    NIXL_ERROR << "RADOS failed to create a C++ completion for object '" << pair.oid
                               << "'; submitting no further pairs";
                    break;
                }
                completion_held = true;
                const int ret = capi_->cppWrite(
                    &cppCtx_, pair.oid.c_str(), completion, pair.payload, pair.len, pair.offset);
                if (ret != 0) {
                    capi_->cppRelease(completion);
                    completion_held = false;
                    pair.payload.clear();
                    pair.result = ret < 0 ? ret : -EIO;
                    NIXL_ERROR << "RADOS failed to submit a C++ write to object '" << pair.oid
                               << "' with error " << ret << "; submitting no further pairs";
                    break;
                }
                pair.owned = directRequestState::cppWrite{completion};
                completion_held = false;
            }
            catch (const std::bad_alloc &) {
                if (completion_held) {
                    capi_->cppRelease(completion);
                }
                pair.payload.clear();
                pair.result = -ENOMEM;
                NIXL_ERROR << "RADOS C++ write allocation failed for object '" << pair.oid
                           << "'; submitting no further pairs";
                break;
            }
        }
    }

    size_t accepted = 0;
    for (const auto &pair : state->pairs()) {
        if (pairOwned(pair)) {
            ++accepted;
        }
    }

    if (accepted > 0) {
        state->phase = directRequestState::phase_t::ACTIVE;
        return NIXL_IN_PROG;
    }
    for (const auto &pair : state->pairs()) {
        if (pair.result != 0) {
            state->phase = directRequestState::phase_t::TERMINAL;
            state->terminalStatus = radosResultToNixl(state->operation(), pair.result, pair.len);
            return state->terminalStatus;
        }
    }
    state->phase = directRequestState::phase_t::TERMINAL;
    state->terminalStatus = NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlRadosEngine::checkXfer(nixlBackendReqH *handle) const {
    std::shared_ptr<directRequestState> state;
    {
        std::lock_guard alock(activeMutex_);
        state = lookupActiveLocked(handle);
    }
    if (state == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    std::unique_lock slock(state->mtx);
    switch (state->phase) {
    case directRequestState::phase_t::PREPARED:
        return NIXL_ERR_NOT_POSTED;
    case directRequestState::phase_t::TERMINAL:
        return state->terminalStatus;
    case directRequestState::phase_t::ABANDONED:
        return NIXL_ERR_INVALID_PARAM;
    case directRequestState::phase_t::ACTIVE:
        break;
    }

    for (auto &pair : state->pairs()) {
        if (pairUnproven(*capi_, pair)) {
            // Status wait: one sleep per call with the state lock released,
            // no rescan; unproven pairs are harvested on the next poll.
            if (statusWaitTimer_ != nullptr) {
                slock.unlock();
                statusWaitTimer_->sleep(statusWaitBudgetUs_);
            }
            return NIXL_IN_PROG;
        }
        if (pairOwned(pair)) {
            pair.result = retirePair(*capi_, pair, false);
        }
    }

    nixl_status_t overall = NIXL_SUCCESS;
    for (const auto &pair : state->pairs()) {
        const auto status = radosResultToNixl(state->operation(), pair.result, pair.len);
        if (status != NIXL_SUCCESS) {
            overall = status;
            break;
        }
    }
    state->terminalStatus = overall;
    state->phase = directRequestState::phase_t::TERMINAL;
    return overall;
}

nixl_status_t
nixlRadosEngine::releaseReqH(nixlBackendReqH *handle) const {
    std::shared_ptr<directRequestState> state;
    {
        std::lock_guard alock(activeMutex_);
        state = lookupActiveLocked(handle);
        if (state == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (shuttingDown_) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        handles_.erase(handle);
    }
    drainRequestState(state);
    delete static_cast<nixlRadosBackendReqH *>(handle);
    return NIXL_SUCCESS;
}
