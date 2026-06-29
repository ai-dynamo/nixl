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
 * @file nixl_metadata_backend.h
 * @brief Core-internal contract for nixlMDManager metadata backends.
 */
#ifndef NIXL_SRC_CORE_NIXL_METADATA_BACKEND_H
#define NIXL_SRC_CORE_NIXL_METADATA_BACKEND_H

#include "nixl_descriptors.h"
#include "nixl_types.h"

#include <string>
#include <string_view>

/**
 * @class nixlMetadataBackend
 * @brief Metadata-exchange operations that nixlMDManager dispatches to.
 *
 * Each transport implements this contract (P2P for now; ETCD/TCPStore later).
 * Core-internal: not part of the installed public headers, so backend
 * dependencies never leak into the public API. Operational addressing
 * (`ipAddr`/`port`, `metadataLabel`) is carried in `nixl_opt_args_t`.
 */
class nixlMetadataBackend {
public:
    virtual ~nixlMetadataBackend() = default;

    /// Stable transport name reported by nixlMDManager::getBackend().
    [[nodiscard]] virtual std::string_view
    name() const = 0;

    /// Make our full local metadata available through this backend.
    [[nodiscard]] virtual nixl_status_t
    sendLocal(const nixl_opt_args_t *extra_params) = 0;

    /// Make a partial local metadata blob available through this backend.
    [[nodiscard]] virtual nixl_status_t
    sendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) = 0;

    /// Initiate retrieval of a remote agent's metadata into the agent cache.
    [[nodiscard]] virtual nixl_status_t
    fetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) = 0;

    /// Withdraw our metadata through this backend.
    [[nodiscard]] virtual nixl_status_t
    invalidateLocal(const nixl_opt_args_t *extra_params) = 0;
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_BACKEND_H
