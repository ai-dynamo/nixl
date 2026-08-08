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
/**
 * @file nixl_md_manager.h
 * @brief Core-internal, agent-owned metadata manager that routes metadata
 *        exchange to a pluggable backend.
 */
#ifndef NIXL_SRC_CORE_NIXL_MD_MANAGER_H
#define NIXL_SRC_CORE_NIXL_MD_MANAGER_H

#include "nixl_descriptors.h"
#include "nixl_metadata_backend.h"
#include "nixl_types.h"
#include "common/nixl_time.h"

#include <atomic>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

class nixlMetadataContext;

/**
 * @class nixlMetadataWorker
 * @brief A single background thread that drains a task queue and calls a poll
 *        callback each iteration.
 *
 * This is where all blocking metadata I/O and background servicing runs, off the
 * caller thread. It knows nothing about backends or routing; nixlMDManager gives
 * it tasks to run and a poll callback to invoke.
 */
class nixlMetadataWorker {
public:
    using Poll = std::function<void()>;

    nixlMetadataWorker() = default;
    ~nixlMetadataWorker();

    nixlMetadataWorker(const nixlMetadataWorker &) = delete;
    nixlMetadataWorker &
    operator=(const nixlMetadataWorker &) = delete;

    /**
     * @brief Launch the loop (no-op if already running). Each pass runs the
     *        queued tasks, calls @p poll, then yields for @p delay.
     */
    void
    start(Poll poll, nixlTime::us_t delay);

    /** @brief Drain queued tasks, then stop and join. Idempotent. */
    void
    stop();

    /** @brief Enqueue a task; ignored once the worker is stopping/stopped. */
    void
    submit(nixlWorkerTask task);

private:
    void
    loop();

    Poll poll_;
    nixlTime::us_t delay_ = 0;
    std::deque<nixlWorkerTask> tasks_;
    std::mutex mutex_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> accepting_{true};
    std::exception_ptr exception_;
};

/**
 * @class nixlMDManager
 * @brief Core-internal: owns the metadata backends and routes each call.
 *
 * Built and owned by nixlAgentData (constructed unconditionally). Depends
 * only on the nixlMetadataContext interface (not nixlAgent), so there is no cycle.
 * Holds the address-routed backend (P2P) plus an optional name-addressed backend
 * chosen from the environment; a peer address selects P2P, otherwise the name
 * backend (address wins per call). Backend I/O runs on an owned nixlMetadataWorker.
 */
class nixlMDManager {
public:
    explicit nixlMDManager(nixlMetadataContext &ctx);
    ~nixlMDManager();

    nixlMDManager(const nixlMDManager &) = delete;
    nixlMDManager(nixlMDManager &&) = delete;
    nixlMDManager &
    operator=(const nixlMDManager &) = delete;
    nixlMDManager &
    operator=(nixlMDManager &&) = delete;

    /**
     * @brief Whether the ETCD backend is selected (NIXL_ETCD_ENDPOINTS set and
     *        this build has ETCD support). Single source of truth: the manager
     *        uses it to pick the name backend, the agent to gate the comm thread.
     */
    [[nodiscard]] static bool
    etcdConfigured();

    /**
     * @brief Publish the full local metadata blob through the active backend.
     *
     * Routes to a backend's prepareSendLocal on the caller thread; any resulting
     * transport work is scheduled on the worker thread.
     */
    [[nodiscard]] nixl_status_t
    sendLocalMD(const nixl_opt_args_t *extra_params = nullptr);

    /** @brief Publish a partial local metadata blob through the active backend. */
    [[nodiscard]] nixl_status_t
    sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t *extra_params = nullptr);

    /** @brief Initiate retrieval of a remote agent's metadata. */
    [[nodiscard]] nixl_status_t
    fetchRemoteMD(const std::string &remote_name, const nixl_opt_args_t *extra_params = nullptr);

    /** @brief Withdraw our metadata through the active backend. */
    [[nodiscard]] nixl_status_t
    invalidateLocalMD(const nixl_opt_args_t *extra_params = nullptr);

    /**
     * @brief Name of the active metadata backend: the name backend when one is
     *        configured, otherwise "P2P".
     */
    [[nodiscard]] std::string_view
    backendName() const noexcept;

    /**
     * @brief Start the worker thread if any backend needs it. Called by the
     *        agent once construction is complete (not during it).
     */
    void
    start();

    /** @brief Drain pending tasks, stop and join the worker. Idempotent. */
    void
    stop();

private:
    // Route a call: select the backend (address wins), run its caller-thread
    // prepare step, and enqueue the resulting task on the worker. `prepare` is
    // invoked as `prepare(nixlMetadataBackend&) -> nixlPreparedOp`.
    template<typename Prepare>
    [[nodiscard]] nixl_status_t
    route(const nixl_opt_args_t *extra_params, Prepare prepare);

    // Worker callback: poll each backend for background work (P2P accept/read,
    // ETCD watch). Runs on the worker thread.
    void
    pollBackends();

    nixlMetadataContext &ctx_;
    // P2P (address-routed), always present.
    const std::unique_ptr<nixlMetadataBackend> p2pBackend_;
    // Name-addressed backend (etcd/future), or null when none configured.
    const std::unique_ptr<nixlMetadataBackend> backend_;
    // Runs backend I/O and background servicing off the caller thread.
    nixlMetadataWorker worker_;
};

#endif // NIXL_SRC_CORE_NIXL_MD_MANAGER_H
