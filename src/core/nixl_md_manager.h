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
#include "nixl_types.h"

#include <memory>
#include <string>
#include <string_view>

class nixlMetadataContext;
class nixlMetadataBackend;

/**
 * @class nixlMDManager
 * @brief Core-internal: owns the active backend and routes each call to it.
 *
 * Built and owned by nixlAgentData when metadata exchange is enabled. Depends
 * only on the nixlMetadataContext interface (not nixlAgent), so there is no cycle.
 */
class nixlMDManager {
public:
    // `use_etcd` / `use_tcpstore` mirror nixlAgentData's cached env checks (read
    // once in the agent ctor). When one is set the matching centralized-store
    // backend is selected, else P2P. The two stores are mutually exclusive.
    nixlMDManager(nixlMetadataContext &ctx, bool use_etcd, bool use_tcpstore);
    ~nixlMDManager();

    nixlMDManager(const nixlMDManager &) = delete;
    nixlMDManager(nixlMDManager &&) = delete;
    nixlMDManager &
    operator=(const nixlMDManager &) = delete;
    nixlMDManager &
    operator=(nixlMDManager &&) = delete;

    /**
     * @brief Publish the full local metadata blob through the active backend.
     *
     * @param extra_params Operational args (e.g. `ipAddr`/`port` for P2P).
     */
    [[nodiscard]] nixl_status_t
    sendLocalMD(const nixl_opt_args_t *extra_params = nullptr) const;

    /**
     * @brief Publish a partial local metadata blob through the active backend.
     *
     * @param descs        Descriptor list to include in the metadata.
     * @param extra_params Operational args forwarded to the backend.
     */
    [[nodiscard]] nixl_status_t
    sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                       const nixl_opt_args_t *extra_params = nullptr) const;

    /**
     * @brief Initiate retrieval of a remote agent's metadata.
     *
     * @param remote_name  Remote agent name.
     * @param extra_params Operational args (e.g. `ipAddr`/`port` for P2P).
     */
    [[nodiscard]] nixl_status_t
    fetchRemoteMD(const std::string &remote_name,
                  const nixl_opt_args_t *extra_params = nullptr) const;

    /**
     * @brief Withdraw our metadata through the active backend.
     *
     * @param extra_params Operational args (e.g. `ipAddr`/`port` for P2P).
     */
    [[nodiscard]] nixl_status_t
    invalidateLocalMD(const nixl_opt_args_t *extra_params = nullptr) const;

    /**
     * @brief Name of the active metadata backend (e.g. "P2P").
     */
    [[nodiscard]] std::string_view
    backendName() const noexcept;

private:
    const std::unique_ptr<nixlMetadataBackend> backend_;
};

#endif // NIXL_SRC_CORE_NIXL_MD_MANAGER_H
