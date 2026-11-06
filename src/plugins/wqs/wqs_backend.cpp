// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "wqs_backend.h"

#include "common/nixl_log.h"

#include <climits>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {
struct wqsWireDescriptor {
    void *address;
    size_t length;
};

constexpr size_t wire_overhead = sizeof(size_t) + sizeof(wqsWireDescriptor);

struct wqsRegistration {
    wqsRegistration(const nixlBasicDesc &value, nixl_mem_t mem_type, std::string object_key)
        : desc(value),
          type(mem_type),
          key(std::move(object_key)) {}

    nixlBasicDesc desc;
    nixl_mem_t type;
    std::string key;
    bool registered = true;
    std::mutex mutex;
    std::condition_variable idle;
    size_t active = 0;
};

class wqsMetadata final : public nixlBackendMD {
public:
    wqsMetadata(std::shared_ptr<wqsRegistration> value, bool is_primary)
        : nixlBackendMD(true),
          registration(std::move(value)),
          primary(is_primary) {}

    std::shared_ptr<wqsRegistration> registration;
    bool primary;
};

struct wqsLease {
    explicit wqsLease(std::shared_ptr<wqsRegistration> value) : registration(std::move(value)) {
        const std::lock_guard lock(registration->mutex);
        ++registration->active;
    }

    ~wqsLease() {
        reset();
    }

    wqsLease(const wqsLease &) = delete;
    wqsLease &
    operator=(const wqsLease &) = delete;

    wqsLease(wqsLease &&other) noexcept : registration(std::move(other.registration)) {}

    wqsLease &
    operator=(wqsLease &&other) noexcept {
        if (this != &other) {
            reset();
            registration = std::move(other.registration);
        }
        return *this;
    }

private:
    void
    reset() {
        auto value = std::move(registration);
        if (value) {
            const std::lock_guard lock(value->mutex);
            --value->active;
            value->idle.notify_all();
        }
    }

    std::shared_ptr<wqsRegistration> registration;
};

struct wqsEntry {
    std::string key;
    uintptr_t address;
    size_t length;
};

struct wqsWork {
    std::vector<wqsEntry> entries;
    std::vector<wqsLease> leases;
};

struct wqsRequestState {
    std::mutex mutex;
    std::condition_variable done;
    nixl_status_t status = NIXL_ERR_NOT_POSTED;
    bool active = false;
    bool started = false;
    bool canceled = false;
};

class wqsRequest final : public nixlBackendReqH {
public:
    explicit wqsRequest(std::shared_ptr<nixlWqsOwner> value)
        : owner(std::move(value)),
          state(std::make_shared<wqsRequestState>()) {}

    ~wqsRequest() override {
        std::unique_lock lock(state->mutex);
        if (!state->started) {
            state->canceled = true;
        }
        state->done.wait(lock, [this] { return !state->active; });
    }

    std::shared_ptr<nixlWqsOwner> owner;
    std::shared_ptr<wqsRequestState> state;
};

bool
fitsKey(const std::string &key, const nixlWqsConfig &config) {
    return !key.empty() && config.payloadBytes >= wire_overhead &&
        key.size() <= config.payloadBytes - wire_overhead;
}

bool
covers(const wqsRegistration &registration, const nixlMetaDesc &desc) {
    const auto &base = registration.desc;
    return desc.devId == base.devId && desc.addr >= base.addr && desc.len <= base.len &&
        desc.addr - base.addr <= base.len - desc.len;
}

nixl_status_t
execute(nixlWqsSession &session, nixl_xfer_op_t operation, const wqsWork &work) {
    const auto &config = session.config();
    auto &pool = session.pool();
    size_t next = 0;
    while (next < work.entries.size()) {
        pool.reset();
        const size_t start = next;
        size_t payload = 0;
        std::vector<std::string> keys;
        std::vector<void *> buffers;
        std::vector<size_t> sizes;
        while (next < work.entries.size() && keys.size() < config.maxBatchEntries) {
            const auto &entry = work.entries[next];
            const size_t cost = wire_overhead + entry.key.size();
            if (cost > config.payloadBytes - payload || !pool.canFit(entry.length)) {
                break;
            }
            size_t length = entry.length;
            if (operation == NIXL_READ) {
                const int64_t actual = session.client().getSize(entry.key);
                if (actual <= 0) {
                    NIXL_ERROR << "WQS getSize failed before READ, result=" << actual;
                    return NIXL_ERR_BACKEND;
                }
                if (static_cast<uint64_t>(actual) != entry.length) {
                    NIXL_ERROR << "WQS READ length mismatch: requested=" << entry.length
                               << ", actual=" << actual;
                    return NIXL_ERR_MISMATCH;
                }
                length = static_cast<size_t>(actual);
            }
            keys.push_back(entry.key);
            buffers.push_back(pool.allocate(length));
            sizes.push_back(length);
            payload += cost;
            ++next;
        }
        if (keys.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (operation == NIXL_WRITE) {
            for (size_t i = 0; i < keys.size(); ++i) {
                std::memcpy(buffers[i],
                            reinterpret_cast<const void *>(work.entries[start + i].address),
                            sizes[i]);
            }
        }
        const auto results = operation == NIXL_READ ? session.client().read(keys, buffers, sizes) :
                                                      session.client().write(keys, buffers, sizes);
        if (results.size() != keys.size()) {
            return NIXL_ERR_BACKEND;
        }
        for (size_t i = 0; i < results.size(); ++i) {
            if (operation == NIXL_WRITE) {
                if (results[i] != 0) {
                    return NIXL_ERR_BACKEND;
                }
            } else if (results[i] <= 0) {
                return NIXL_ERR_BACKEND;
            } else if (static_cast<size_t>(results[i]) != sizes[i]) {
                return NIXL_ERR_MISMATCH;
            }
        }
        if (operation == NIXL_READ) {
            for (size_t i = 0; i < keys.size(); ++i) {
                std::memcpy(reinterpret_cast<void *>(work.entries[start + i].address),
                            buffers[i],
                            sizes[i]);
            }
        }
    }
    return NIXL_SUCCESS;
}
} // namespace

class nixlWqsEngine::impl {
public:
    impl(nixlWqsConfig value, std::shared_ptr<nixlWqsSession> shared_session)
        : config(std::move(value)),
          session(std::move(shared_session)) {
        if (!session || !config.sameSession(session->config())) {
            throw std::invalid_argument("WQS session configuration mismatch");
        }
    }

    ~impl() {
        std::unique_lock lock(owner->mutex);
        owner->closing = true;
        owner->idle.wait(lock, [this] { return owner->pending == 0; });
    }

    nixl_status_t
    prepare(nixl_xfer_op_t operation,
            const nixl_meta_dlist_t &local,
            const nixl_meta_dlist_t &remote,
            bool acquire,
            wqsWork &work) {
        if ((operation != NIXL_READ && operation != NIXL_WRITE) || local.getType() != DRAM_SEG ||
            remote.getType() != OBJ_SEG || local.descCount() <= 0 ||
            local.descCount() != remote.descCount()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        std::unordered_set<std::string> write_keys;
        if (operation == NIXL_WRITE) {
            write_keys.reserve(local.descCount());
        }
        if (acquire) {
            work.entries.reserve(local.descCount());
            work.leases.reserve(size_t{2} * local.descCount());
        }
        const std::lock_guard lock(metadataMutex);
        for (int i = 0; i < local.descCount(); ++i) {
            const auto &buffer = local[i];
            const auto &object = remote[i];
            const auto local_md = metadata.find(buffer.metadataP);
            const auto remote_md = metadata.find(object.metadataP);
            if (local_md == metadata.end() || remote_md == metadata.end()) {
                return NIXL_ERR_INVALID_PARAM;
            }
            const auto &dram = local_md->second->registration;
            const auto &obj = remote_md->second->registration;
            if (!dram->registered || !obj->registered || dram->type != DRAM_SEG ||
                obj->type != OBJ_SEG || !covers(*dram, buffer) || !covers(*obj, object) ||
                object.addr != 0 || object.len != obj->desc.len || buffer.len != object.len ||
                buffer.addr == 0 || buffer.len == 0 || buffer.len >= INT_MAX ||
                buffer.len > config.stagingPoolBytes) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (operation == NIXL_WRITE && !write_keys.insert(obj->key).second) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (acquire) {
                work.entries.push_back(wqsEntry{obj->key, buffer.addr, buffer.len});
                work.leases.emplace_back(dram);
                work.leases.emplace_back(obj);
            }
        }
        return NIXL_SUCCESS;
    }

    nixlWqsConfig config;
    std::shared_ptr<nixlWqsSession> session;
    std::shared_ptr<nixlWqsOwner> owner = std::make_shared<nixlWqsOwner>();
    std::mutex metadataMutex;
    std::unordered_map<nixlBackendMD *, std::unique_ptr<wqsMetadata>> metadata;
};

nixlWqsEngine::nixlWqsEngine(const nixlBackendInitParams *params) : nixlBackendEngine(params) {
    auto config = nixlWqsConfig::fromParams(getCustomParams());
    auto session = getWqsSession(config);
    impl_ = std::make_unique<impl>(std::move(config), std::move(session));
}

nixlWqsEngine::nixlWqsEngine(const nixlBackendInitParams *params,
                             std::shared_ptr<nixlWqsSession> session)
    : nixlBackendEngine(params),
      impl_(std::make_unique<impl>(nixlWqsConfig::fromParams(getCustomParams()),
                                   std::move(session))) {}

nixlWqsEngine::~nixlWqsEngine() = default;

nixl_status_t
nixlWqsEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &type, nixlBackendMD *&out) {
    out = nullptr;
    if (type != DRAM_SEG && type != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (mem.len == 0 || mem.len > std::numeric_limits<uintptr_t>::max() - mem.addr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::string key = type == OBJ_SEG ? impl_->config.keyPrefix + mem.metaInfo : "";
    if ((type == DRAM_SEG && mem.addr == 0) ||
        (type == OBJ_SEG &&
         (mem.addr != 0 || mem.metaInfo.empty() || !fitsKey(key, impl_->config) ||
          mem.len >= INT_MAX || mem.len > impl_->config.stagingPoolBytes))) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto registration = std::make_shared<wqsRegistration>(mem, type, key);
    auto md = std::make_unique<wqsMetadata>(std::move(registration), true);
    auto *pointer = md.get();
    const std::lock_guard lock(impl_->metadataMutex);
    if (type == OBJ_SEG) {
        for (const auto &[existing, value] : impl_->metadata) {
            (void)existing;
            const auto &reg = *value->registration;
            if (value->primary && reg.type == OBJ_SEG && reg.desc.devId == mem.devId) {
                return NIXL_ERR_INVALID_PARAM;
            }
        }
    }
    impl_->metadata.emplace(pointer, std::move(md));
    out = pointer;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::deregisterMem(nixlBackendMD *meta) {
    std::shared_ptr<wqsRegistration> registration;
    {
        const std::lock_guard lock(impl_->metadataMutex);
        const auto it = impl_->metadata.find(meta);
        if (it == impl_->metadata.end() || !it->second->primary) {
            return NIXL_ERR_INVALID_PARAM;
        }
        registration = it->second->registration;
        registration->registered = false;
        impl_->metadata.erase(it);
    }
    std::unique_lock lock(registration->mutex);
    registration->idle.wait(lock, [&registration] { return registration->active == 0; });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    output = nullptr;
    const std::lock_guard lock(impl_->metadataMutex);
    const auto it = impl_->metadata.find(input);
    if (it == impl_->metadata.end() || !it->second->registration->registered) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto md = std::make_unique<wqsMetadata>(it->second->registration, false);
    auto *pointer = md.get();
    impl_->metadata.emplace(pointer, std::move(md));
    output = pointer;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::unloadMD(nixlBackendMD *meta) {
    const std::lock_guard lock(impl_->metadataMutex);
    const auto it = impl_->metadata.find(meta);
    if (it == impl_->metadata.end() || it->second->primary) {
        return NIXL_ERR_INVALID_PARAM;
    }
    impl_->metadata.erase(it);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::connect(const std::string &agent) {
    return agent == localAgent ? NIXL_SUCCESS : NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlWqsEngine::disconnect(const std::string &agent) {
    if (agent != localAgent) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    std::unique_lock lock(impl_->owner->mutex);
    impl_->owner->idle.wait(lock, [this] { return impl_->owner->pending == 0; });
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    resp.clear();
    if (descs.getType() != OBJ_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    std::vector<std::string> keys;
    for (const auto &desc : descs) {
        std::string key = impl_->config.keyPrefix + desc.metaInfo;
        if (desc.metaInfo.empty() || !fitsKey(key, impl_->config)) {
            return NIXL_ERR_INVALID_PARAM;
        }
        keys.push_back(std::move(key));
    }
    if (keys.empty()) {
        return NIXL_SUCCESS;
    }
    using query_result_t = std::pair<nixl_status_t, std::vector<nixl_query_resp_t>>;
    auto promise = std::make_shared<std::promise<query_result_t>>();
    auto future = promise->get_future();
    const auto session = impl_->session;
    const auto status = session->submit(impl_->owner, [session, keys = std::move(keys), promise] {
        try {
            query_result_t result{NIXL_SUCCESS, {}};
            for (const auto &key : keys) {
                const int64_t length = session->client().getSize(key);
                if (length > 0) {
                    result.second.emplace_back(nixl_b_params_t{{"size", std::to_string(length)}});
                } else {
                    result.second.emplace_back(std::nullopt);
                    if (length < 0) {
                        result.first = NIXL_ERR_BACKEND;
                    }
                }
            }
            promise->set_value(std::move(result));
        }
        catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    if (status != NIXL_SUCCESS) {
        return status;
    }
    try {
        auto result = future.get();
        resp = std::move(result.second);
        return result.first;
    }
    catch (const std::exception &error) {
        NIXL_ERROR << "WQS queryMem failed: " << error.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlWqsEngine::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *args) const {
    if (handle || remote_agent != localAgent) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (args && (args->hasNotif || !args->customParam.empty())) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    wqsWork work;
    const auto status = impl_->prepare(operation, local, remote, false, work);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    handle = new wqsRequest(impl_->owner);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlWqsEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *args) const {
    auto *request = dynamic_cast<wqsRequest *>(handle);
    if (!request || request->owner != impl_->owner || remote_agent != localAgent) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (args && (args->hasNotif || !args->customParam.empty())) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    auto state = request->state;
    std::unique_lock lock(state->mutex);
    if (state->active) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
    auto work = std::make_shared<wqsWork>();
    const auto status = impl_->prepare(operation, local, remote, true, *work);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    state->active = true;
    state->started = false;
    state->canceled = false;
    state->status = NIXL_IN_PROG;
    auto session = impl_->session;
    nixl_status_t submitted;
    try {
        submitted = session->submit(impl_->owner, [session, state, work, operation] {
            nixl_status_t result = NIXL_ERR_CANCELED;
            bool canceled;
            {
                const std::lock_guard state_lock(state->mutex);
                canceled = state->canceled;
                state->started = true;
            }
            if (!canceled) {
                try {
                    result = execute(*session, operation, *work);
                }
                catch (const std::exception &error) {
                    NIXL_ERROR << "WQS transfer failed: " << error.what();
                    result = NIXL_ERR_BACKEND;
                }
                catch (...) {
                    result = NIXL_ERR_BACKEND;
                }
            }
            work->leases.clear();
            {
                const std::lock_guard state_lock(state->mutex);
                state->status = result;
                state->active = false;
                state->done.notify_all();
            }
        });
    }
    catch (...) {
        state->active = false;
        state->status = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }
    if (submitted != NIXL_SUCCESS) {
        state->active = false;
        state->status = NIXL_ERR_NOT_POSTED;
        return submitted;
    }
    return NIXL_IN_PROG;
}

nixl_status_t
nixlWqsEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *request = dynamic_cast<wqsRequest *>(handle);
    if (!request || request->owner != impl_->owner) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::lock_guard lock(request->state->mutex);
    return request->state->status;
}

nixl_status_t
nixlWqsEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *request = dynamic_cast<wqsRequest *>(handle);
    if (!request || request->owner != impl_->owner) {
        return NIXL_ERR_INVALID_PARAM;
    }
    delete request;
    return NIXL_SUCCESS;
}
