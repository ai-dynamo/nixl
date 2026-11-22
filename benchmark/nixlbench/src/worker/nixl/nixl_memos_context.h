/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXLBENCH_MEMOS_CONTEXT_H
#define NIXLBENCH_MEMOS_CONTEXT_H

#include <nixl.h>
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nixlbench {

// Drain requests before releasing their handles;
// never deregister/free a buffer while the backend still owns an active request.
// An unresponsive backend can therefore delay shutdown rather than cause a UAF.
inline void
releaseRequest(nixlAgent &agent, nixlXferReqH *&request) {
    if (!request) {
        return;
    }
    // Do not call releaseXferReq on an active request: an unsuccessful
    // cancellation can change its status while the backend still owns buffers.
    while (agent.getXferStatus(request) == NIXL_IN_PROG) {
        std::this_thread::yield();
    }
    while (agent.releaseXferReq(request) != NIXL_SUCCESS) {
        std::this_thread::yield();
    }
    request = nullptr;
}

inline void
releasePreparedList(nixlAgent &agent, nixlDlistH *&list) {
    if (!list) {
        return;
    }
    while (agent.releasedDlistH(list) != NIXL_SUCCESS) {
        std::this_thread::yield();
    }
    list = nullptr;
}

// One logical benchmark worker owns this context. Other workers never borrow
// its registrations or transfer handles. Shared buffer ownership covers partial
// initialization (including an exception before registerMem is called).
class MemosWorkerContext {
    struct Registration {
        nixl_reg_dlist_t descriptors;
        std::shared_ptr<void> buffer;
        bool registered = false;
    };

    std::string name_;
    std::unique_ptr<nixlAgent> agent_;
    nixlBackendH *backend_ = nullptr; // Owned by agent_.
    nixl_opt_args_t options_;
    std::vector<Registration> registrations_;

public:
    explicit MemosWorkerContext(const std::string &prefix, size_t worker) {
        static std::atomic<size_t> sequence{0};
        name_ = prefix + "-worker-" + std::to_string(worker) + "-" +
            std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        nixlAgentConfig config;
        config.useProgThread = false;
        // Retain synchronization even with one logical owner. Ownership is not
        // permission to weaken NIXL or backend threading contracts.
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        options_.backends.resize(1);
        agent_ = std::make_unique<nixlAgent>(name_, config);
    }

    MemosWorkerContext(const MemosWorkerContext &) = delete;
    MemosWorkerContext &
    operator=(const MemosWorkerContext &) = delete;

    ~MemosWorkerContext() {
        for (auto it = registrations_.rbegin(); it != registrations_.rend(); ++it) {
            if (it->registered) {
                if (agent_->deregisterMem(it->descriptors, &options_) != NIXL_SUCCESS) {
                    std::cerr << "DOCA_MEMOS: deregistration failed during context cleanup"
                              << std::endl;
                }
            }
        }
        // All requests were drained by their owning transfer loops. Destroy
        // the agent/backend before freeing allocations, including on a failed
        // deregistration; no agent-bound registration survives this context.
        agent_.reset();
        registrations_.clear();
    }

    nixl_status_t
    initialize(const nixl_b_params_t &overrides) {
        nixl_mem_list_t memory_types;
        nixl_b_params_t parameters;
        auto status = agent_->getPluginParams("DOCA_MEMOS", memory_types, parameters);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        for (const auto &[key, value] : overrides) {
            parameters[key] = value;
        }
        status = agent_->createBackend("DOCA_MEMOS", parameters, backend_);
        options_.backends[0] = backend_;
        return status;
    }

    nixl_status_t
    registerMemory(nixl_reg_dlist_t descriptors, std::shared_ptr<void> buffer = {}) {
        registrations_.push_back({std::move(descriptors), std::move(buffer), false});
        auto &registration = registrations_.back();
        auto status = agent_->registerMem(registration.descriptors, &options_);
        registration.registered = status == NIXL_SUCCESS;
        return status;
    }

    nixlAgent *
    agent() const {
        return agent_.get();
    }

    nixlBackendH *
    backend() const {
        return backend_;
    }

    const std::string &
    name() const {
        return name_;
    }
};

inline nixl_status_t
createMemosContexts(size_t count,
                    const std::string &prefix,
                    const nixl_b_params_t &parameters,
                    std::vector<std::unique_ptr<MemosWorkerContext>> &contexts) {
    std::vector<std::unique_ptr<MemosWorkerContext>> pending;
    pending.reserve(count);
    for (size_t worker = 0; worker < count; ++worker) {
        auto context = std::make_unique<MemosWorkerContext>(prefix, worker);
        auto status = context->initialize(parameters);
        if (status != NIXL_SUCCESS) {
            return status; // RAII also destroys previously initialized contexts.
        }
        pending.push_back(std::move(context));
    }
    contexts = std::move(pending);
    return NIXL_SUCCESS;
}

} // namespace nixlbench
#endif
