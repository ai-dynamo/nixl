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
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
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
applyCephOverrides(nixl::rados::radosApi &api,
                   librados::Rados &cluster,
                   const nixl::rados::ceph_options_t &overrides) {
    for (const auto &[option, value] : overrides) {
        const std::string operation = std::string("conf_set(") + option + ')';
        if (!checkRadosResult(operation.c_str(),
                              api.conf_set(cluster, option.c_str(), value.c_str()))) {
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

bool
bootstrapClient(nixl::rados::radosApi &api,
                const nixl::rados::backendConfig &config,
                librados::Rados &rados,
                librados::IoCtx &ioctx) {
    if (!checkRadosResult("init2", api.init2(rados, config.name.c_str(), nullptr, 0))) {
        return false;
    }
    if (!checkRadosResult("conf_read_file", api.conf_read_file(rados, config.conf.c_str())) ||
        !applyCephOverrides(api, rados, config.ceph_options) ||
        !checkRadosResult("connect", api.connect(rados))) {
        return false;
    }
    if (!checkRadosResult("ioctx_create", api.ioctx_create(rados, config.pool.c_str(), ioctx))) {
        return false;
    }
    if (config.ns) {
        api.set_namespace(ioctx, config.ns->c_str());
    }
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

    struct requestPair {
        std::string oid;
        uintptr_t dram = 0;
        uint64_t offset = 0;
        uint64_t len = 0;
        int result = 0;
        // Non-owning view of snapshotted caller DRAM; borrowed until retirement, rebuilt on repost.
        librados::bufferlist payload;
        librados::AioCompletion *completion = nullptr;
    };

    explicit directRequestState(nixl_xfer_op_t operation) : operation_(operation) {}

    ~directRequestState() {
        for (const requestPair &pair : pairs_) {
            if (pair.completion != nullptr) {
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
            pair.completion = nullptr;
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
pairUnproven(nixl::rados::radosApi &api, const directRequestState::requestPair &pair) {
    return pair.completion != nullptr && api.isCompleteAndCb(pair.completion) == 0;
}

// Reconcile a completed read's direct or replacement output before release.
int
settleReadOutput(directRequestState::requestPair &pair, int result) {
    if (result < 0) {
        return result;
    }
    const auto &payload = pair.payload;
    if (payload.is_provided_buffer(reinterpret_cast<const char *>(pair.dram))) {
        return result;
    }
    if (static_cast<uint64_t>(result) <= pair.len &&
        static_cast<uint64_t>(result) <= payload.length()) {
        auto *dest = reinterpret_cast<unsigned char *>(pair.dram);
        size_t remaining = static_cast<size_t>(result);
        for (const auto &chunk : payload.buffers()) {
            if (remaining == 0) {
                break;
            }
            const size_t take = std::min(remaining, static_cast<size_t>(chunk.length()));
            std::memcpy(dest, chunk.c_str(), take);
            dest += take;
            remaining -= take;
        }
        return result;
    }
    NIXL_ERROR << "RADOS read delivered " << result
               << " bytes through a replacement bufferlist beyond the caller destination bounds "
                  "for object '"
               << pair.oid << "'";
    return -ERANGE;
}

// wait=false runs only after the caller proved callback retirement, so it stays
// nonblocking; an empty slot (never posted or already harvested) yields its cached result.
int
retirePair(nixl::rados::radosApi &api,
           nixl_xfer_op_t operation,
           directRequestState::requestPair &pair,
           bool wait) {
    if (pair.completion == nullptr) {
        return pair.result;
    }
    if (wait) {
        api.waitForCompleteAndCb(pair.completion);
    }
    int result = api.getReturnValue(pair.completion);
    if (operation == NIXL_READ) {
        result = settleReadOutput(pair, result);
    }
    api.release(pair.completion);
    pair.payload.clear();
    pair.completion = nullptr;
    return result;
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
    statWindow(nixl::rados::radosApi &api, size_t width) : api_(api), width_(width) {}

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
                api_.waitForCompleteAndCb(slot.comp);
            }
            api_.release(slot.comp);
            slot.comp = nullptr;
        }
    }

    statSlot &
    operator[](size_t i) {
        return slots_[i];
    }

private:
    nixl::rados::radosApi &api_;
    const size_t width_;
    std::array<statSlot, kAioStatWindow> slots_;
};

} // namespace

// Opaque token; its real state lives in the engine's handles_, so registry lookup
// is what rejects null, unknown, and already-released handles.
class nixlRadosBackendReqH final : public nixlBackendReqH {};

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params)
    : nixlRadosEngine(init_params, std::make_unique<nixl::rados::realRadosApi>()) {}

nixlRadosEngine::nixlRadosEngine(const nixlBackendInitParams *init_params,
                                 std::unique_ptr<nixl::rados::radosApi> api,
                                 std::unique_ptr<nixl::rados::statusWaitTimer> status_wait_timer)
    : nixlBackendEngine(init_params),
      api_(std::move(api)) {
    if (api_ == nullptr) {
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
    if (!bootstrapClient(*api_, config, data_.rados, data_.ioctx)) {
        initErr = true;
        return;
    }
    if (!bootstrapClient(*api_, config, meta_.rados, meta_.ioctx)) {
        NIXL_ERROR << "RADOS metadata stat client initialization failed";
        initErr = true;
    }
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
    // Teardown is RAII only, in the member order fixed by rados_backend.h.
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
        statWindow window(*api_, width);
        for (size_t i = 0; i < width; ++i) {
            const nixlBlobDesc &desc = descs[static_cast<int>(base + i)];
            statSlot &slot = window[i];
            slot.oid = desc.metaInfo.empty() ? std::to_string(desc.devId) : desc.metaInfo;
            slot.comp = api_->createCompletion();
            if (slot.comp == nullptr) {
                slot.submit_error = -ENOMEM;
                continue;
            }
            const int submit_ret = api_->aioStat(
                &meta_.ioctx, slot.oid.c_str(), slot.comp, &slot.size, &slot.prval, &slot.merged);
            if (submit_ret != 0) {
                api_->release(slot.comp);
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
            api_->waitForCompleteAndCb(slot.comp);
            const int prval = *slot.prval;
            const int aggregate = api_->getReturnValue(slot.comp);
            api_->release(slot.comp);
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
        if (pair.completion == nullptr) {
            continue;
        }
        const int result = retirePair(*api_, state->operation(), pair, true);
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
    bool accepted = false;
    for (auto &pair : state->pairs()) {
        librados::AioCompletion *completion = nullptr;
        bool completion_held = false;
        try {
            pair.payload = librados::bufferlist::static_from_mem(
                reinterpret_cast<char *>(pair.dram), pair.len);
            completion = api_->createCompletion();
            if (completion == nullptr) {
                pair.payload.clear();
                pair.result = -ENOMEM;
                NIXL_ERROR << "RADOS failed to create a completion for object '" << pair.oid
                           << "'; submitting no further pairs";
                break;
            }
            completion_held = true;
            const int ret = is_read ? api_->aioRead(&data_.ioctx,
                                                    pair.oid.c_str(),
                                                    completion,
                                                    &pair.payload,
                                                    pair.len,
                                                    pair.offset) :
                                      api_->aioWrite(&data_.ioctx,
                                                     pair.oid.c_str(),
                                                     completion,
                                                     pair.payload,
                                                     pair.len,
                                                     pair.offset);
            if (ret != 0) {
                api_->release(completion);
                completion_held = false;
                pair.payload.clear();
                pair.result = ret < 0 ? ret : -EIO;
                NIXL_ERROR << "RADOS failed to submit " << (is_read ? "a read from" : "a write to")
                           << " object '" << pair.oid << "' with error " << ret
                           << "; submitting no further pairs";
                break;
            }
            pair.completion = completion;
            completion_held = false;
            accepted = true;
        }
        catch (const std::bad_alloc &) {
            if (completion_held) {
                api_->release(completion);
            }
            pair.payload.clear();
            pair.result = -ENOMEM;
            NIXL_ERROR << "RADOS " << (is_read ? "read" : "write")
                       << " allocation failed for object '" << pair.oid
                       << "'; submitting no further pairs";
            break;
        }
    }

    if (accepted) {
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
        if (pairUnproven(*api_, pair)) {
            // Status wait: one sleep per call with the state lock released,
            // no rescan; unproven pairs are harvested on the next poll.
            if (statusWaitTimer_ != nullptr) {
                slock.unlock();
                statusWaitTimer_->sleep(statusWaitBudgetUs_);
            }
            return NIXL_IN_PROG;
        }
        if (pair.completion != nullptr) {
            pair.result = retirePair(*api_, state->operation(), pair, false);
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
