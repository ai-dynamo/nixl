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
#include "nixl_md_config.h"
#include "nixl_metadata_backend.h"
#include "nixl_types.h"

#include <memory>
#include <string>
#include <string_view>

class nixlMetadataContext;

/**
 * @class nixlMDManager
 * @brief Core-internal: owns the metadata backends and routes each call.
 *
 * Built and owned by nixlAgentData (constructed unconditionally). Reaches the
 * agent only through the nixlMetadataContext interface, and takes the settings
 * it and its backends need as a constructor parameter.
 * Holds the address-routed backend (P2P) plus an optional name-addressed backend
 * chosen from the environment; a peer address selects P2P, otherwise the name
 * backend (address wins per call). Routing is all it does: threading belongs to
 * each backend, so the manager neither owns a thread nor schedules work.
 */
class nixlMDManager {
public:
    nixlMDManager(nixlMetadataContext &ctx, const nixlMDConfig &config);
    ~nixlMDManager();

    nixlMDManager(const nixlMDManager &) = delete;
    nixlMDManager(nixlMDManager &&) = delete;
    nixlMDManager &
    operator=(const nixlMDManager &) = delete;
    nixlMDManager &
    operator=(nixlMDManager &&) = delete;

    /**
     * @brief Whether the ETCD backend is selected (NIXL_ETCD_ENDPOINTS set and
     *        this build has ETCD support), i.e. how the manager picks the
     *        name-addressed backend.
     */
    [[nodiscard]] static bool
    etcdConfigured();

    /**
     * @brief Whether any backend runs a thread of its own. Answered from the
     *        backends alone, so callers do not re-derive it from the
     *        environment; the agent uses it to decide whether its sync mode must
     *        be upgraded.
     */
    [[nodiscard]] bool
    usesThread() const noexcept;

    /**
     * @brief Publish the full local metadata blob through the active backend.
     *
     * Returns the backend's synchronous result; whether the transport I/O has
     * completed by then is the backend's business.
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
     * @brief Let each backend begin its background work. Called by the agent
     *        once construction is complete (not during it).
     */
    void
    start();

    /** @brief Stop each backend's background work. Idempotent. */
    void
    stop();

private:
    // The backend a call routes to: a peer address selects P2P, otherwise the
    // name-addressed backend, which is null when none is configured.
    [[nodiscard]] nixlMetadataBackend *
    select(const nixl_opt_args_t *extra_params) const noexcept;

    // P2P (address-routed), always present.
    const std::unique_ptr<nixlMetadataBackend> p2pBackend_;
    // Name-addressed backend (etcd/tcpstore/future), or null when none configured.
    const std::unique_ptr<nixlMetadataBackend> backend_;
};

#endif // NIXL_SRC_CORE_NIXL_MD_MANAGER_H
