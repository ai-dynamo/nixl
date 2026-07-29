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
#include "nixl_md_manager.h"

#include "agent_data.h"
#include "nixl_p2p_metadata_backend.h"
#include "nixl_tcpstore_metadata_backend.h"

#if HAVE_ETCD
#include "nixl_etcd_metadata_backend.h"
#endif

#include "common/configuration.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"

#include <chrono>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

// Definition of the default metadata label (declared in nixl_types.h). It used
// to live in the now-deleted nixl_listener.cpp; the manager is a natural home.
const std::string default_metadata_label = "metadata";

namespace {

// How long one worker pass spends on queued tasks before polling the backends.
// A task can block on store I/O, so draining the queue unconditionally would
// hold back inbound servicing (P2P accepts, etcd invalidations) for as long as
// the slowest backend takes; the remainder stays queued for the next pass.
constexpr auto task_budget = std::chrono::milliseconds(100);

// The name-addressed backend for this run, chosen from the environment (null
// when none is configured, i.e. address-only / P2P). Adding a name-addressed
// transport is one class plus a branch here; it is not tied to any storage kind.
// ETCD and TCPStore are mutually exclusive (a run uses exactly one).
[[nodiscard]] std::unique_ptr<nixlMetadataBackend>
makeBackend([[maybe_unused]] nixlMetadataContext &ctx,
            [[maybe_unused]] const nixlMDConfig &config) {
    const bool use_tcpstore = nixl::config::checkExistence("NIXL_TCPSTORE_ENDPOINT");
#if HAVE_ETCD
    if (nixlMDManager::etcdConfigured()) {
        if (use_tcpstore) {
            throw std::runtime_error(
                "NIXL_ETCD_ENDPOINTS and NIXL_TCPSTORE_ENDPOINT are mutually exclusive");
        }
        return std::make_unique<nixlEtcdMetadataBackend>(ctx, config);
    }
#endif
    if (use_tcpstore) {
        return std::make_unique<nixlTcpStoreMetadataBackend>(ctx);
    }
    return nullptr;
}

// A call is address-routed (P2P) when it carries a peer address; otherwise it is
// name-addressed and handled by the configured backend.
[[nodiscard]] bool
hasAddress(const nixl_opt_args_t *extra_params) {
    return extra_params && !extra_params->ipAddr.empty();
}

// Error when a call carries no peer address and no name-addressed backend
// (etcd/tcpstore) is configured.
[[nodiscard]] nixl_status_t
noTransport() {
#if HAVE_ETCD
    NIXL_ERROR_FUNC << "no peer address provided and no centralized store configured "
                       "(set NIXL_ETCD_ENDPOINTS or NIXL_TCPSTORE_ENDPOINT)";
    return NIXL_ERR_INVALID_PARAM;
#else
    NIXL_ERROR_FUNC << "no peer address provided and no centralized store configured "
                       "(set NIXL_TCPSTORE_ENDPOINT; this build has no ETCD)";
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

} // namespace

bool
nixlMDManager::etcdConfigured() {
#if HAVE_ETCD
    return nixl::config::checkExistence("NIXL_ETCD_ENDPOINTS");
#else
    return false;
#endif
}

// The manager holds the P2P backend (always) plus an optional name-addressed
// backend, and routes each call by precedence: a peer address selects P2P,
// otherwise the configured backend. This preserves the agent's original per-call
// precedence (address wins over a configured backend).
nixlMDManager::nixlMDManager(nixlMetadataContext &ctx, const nixlMDConfig &config)
    : config_(config),
      p2pBackend_(std::make_unique<nixlP2PMetadataBackend>(ctx, config)),
      backend_(makeBackend(ctx, config)),
      workerNeeded_(p2pBackend_->needsWorker() || (backend_ && backend_->needsWorker())) {}

// worker_ is declared after both backends, so ~nixlMetadataWorker joins before
// they are destroyed; nothing extra is needed here.
nixlMDManager::~nixlMDManager() = default;

template<typename Prepare>
nixl_status_t
nixlMDManager::route(const nixl_opt_args_t *extra_params, Prepare prepare) {
    // Address wins per call: a peer address selects P2P, otherwise the configured
    // name backend (which may be null when none is configured).
    nixlMetadataBackend *b = hasAddress(extra_params) ?
        static_cast<nixlMetadataBackend *>(p2pBackend_.get()) :
        backend_.get();
    if (!b) {
        return noTransport();
    }
    const nixlPreparedOp op = prepare(*b); // caller thread: validate + serialize
    if (op.status != NIXL_SUCCESS) {
        return op.status;
    }
    if (op.task) {
        if (workerNeeded_) {
            worker_.submit(std::move(op.task)); // worker thread: the transport I/O
        } else {
            // No backend asked for a worker, so there is no thread to run this
            // on and queueing it would drop it silently. Only P2P without a
            // listener reaches this (the store backends always need the worker),
            // and prepare() has already done every agent-side step, so what is
            // left is a socket send: safe to run here, at the cost of making the
            // call synchronous. The lock keeps the backend's promise that its
            // transport state is touched by one thread at a time, which the
            // worker provides in the other branch; metadata calls are not
            // serialized by the agent lock.
            const std::lock_guard lk(inlineTaskMutex_);
            op.task();
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlMDManager::sendLocalMD(const nixl_opt_args_t *extra_params) {
    return route(extra_params,
                 [&](nixlMetadataBackend &b) { return b.prepareSendLocal(extra_params); });
}

nixl_status_t
nixlMDManager::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                                  const nixl_opt_args_t *extra_params) {
    return route(extra_params, [&](nixlMetadataBackend &b) {
        return b.prepareSendLocalPartial(descs, extra_params);
    });
}

nixl_status_t
nixlMDManager::fetchRemoteMD(const std::string &remote_name, const nixl_opt_args_t *extra_params) {
    return route(extra_params, [&](nixlMetadataBackend &b) {
        return b.prepareFetchRemote(remote_name, extra_params);
    });
}

nixl_status_t
nixlMDManager::invalidateLocalMD(const nixl_opt_args_t *extra_params) {
    return route(extra_params,
                 [&](nixlMetadataBackend &b) { return b.prepareInvalidateLocal(extra_params); });
}

std::string_view
nixlMDManager::backendName() const noexcept {
    return backend_ ? backend_->name() : p2pBackend_->name();
}

void
nixlMDManager::pollBackends() {
    p2pBackend_->serviceEvents();
    if (backend_) {
        backend_->serviceEvents();
    }
}

bool
nixlMDManager::needsWorker() const noexcept {
    return workerNeeded_;
}

void
nixlMDManager::start() {
    if (workerNeeded_) {
        worker_.start([this] { pollBackends(); }, config_.workerDelay);
    }
}

void
nixlMDManager::stop() {
    worker_.stop();
}

// ---- nixlMetadataWorker ----

nixlMetadataWorker::~nixlMetadataWorker() {
    stop();
}

void
nixlMetadataWorker::start(poll_t poll, nixlTime::us_t delay) {
    if (thread_.joinable()) {
        return;
    }
    poll_ = std::move(poll);
    delay_ = delay;
    stop_.store(false);
    thread_ = std::thread([this] {
        try {
            loop();
        }
        catch (...) {
            exception_ = std::current_exception();
        }
    });
}

void
nixlMetadataWorker::stop() {
    if (!thread_.joinable()) {
        return;
    }
    // Let the loop drain what is queued so a send/invalidate issued just before
    // shutdown still reaches the peer/store. Nothing can enqueue meanwhile: only
    // route() submits, and it is reachable only from the agent's public methods.
    while (true) {
        {
            const std::lock_guard lk(mutex_);
            if (tasks_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stop_.store(true);
    thread_.join();
    if (exception_) {
        try {
            std::rethrow_exception(exception_);
        }
        catch (const std::exception &e) {
            NIXL_WARN << "Metadata worker thread threw an exception: " << e.what();
        }
        exception_ = nullptr;
    }
}

void
nixlMetadataWorker::submit(nixl_worker_task_t task) {
    const std::lock_guard lk(mutex_);
    tasks_.push_back(std::move(task));
}

void
nixlMetadataWorker::runQueuedTasks(std::chrono::steady_clock::time_point until) {
    std::deque<nixl_worker_task_t> batch;
    {
        const std::lock_guard lk(mutex_);
        batch.swap(tasks_);
    }
    while (!batch.empty()) {
        // Isolate each unit of work: one throwing task is logged and the worker
        // keeps running, rather than tearing down all metadata processing.
        try {
            batch.front()();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker task threw an exception: " << e.what();
        }
        batch.pop_front();
        if (std::chrono::steady_clock::now() >= until) {
            break;
        }
    }
    if (!batch.empty()) {
        // Put the remainder back in front of anything submitted meanwhile, so
        // tasks still run in the order they were issued.
        const std::lock_guard lk(mutex_);
        tasks_.insert(tasks_.begin(),
                      std::make_move_iterator(batch.begin()),
                      std::make_move_iterator(batch.end()));
    }
}

void
nixlMetadataWorker::loop() {
    while (!stop_.load()) {
        // Spend a bounded slice on tasks, then poll the backends: a task can
        // block on I/O (an etcd fetch waits on a watch), and draining the whole
        // queue first would stall inbound servicing behind it. At least one task
        // runs per pass, so the queue still drains.
        runQueuedTasks(std::chrono::steady_clock::now() + task_budget);
        try {
            if (poll_) {
                poll_();
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Metadata worker poll threw an exception: " << e.what();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(delay_));
    }
    // stop() waits for the queue to look empty, which it can while a pass still
    // holds tasks the budget deferred; run those before leaving so nothing
    // submitted before shutdown is dropped.
    runQueuedTasks(std::chrono::steady_clock::time_point::max());
}
