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
 * @brief Metadata-exchange contract that nixlMDManager dispatches to.
 *
 * Each transport implements this contract (P2P, ETCD, TCPStore). Core-internal:
 * not part of the installed public headers, so backend dependencies never leak
 * into the public API. Operational addressing (`ipAddr`/`port`, `metadataLabel`)
 * is carried in `nixl_opt_args_t`.
 *
 * Thread contract: a backend owns its own threading. The four operations are
 * called on the CALLER thread and return the status of the synchronous part
 * (validation, serialization); blocking transport I/O and background servicing
 * belong on a thread of the backend's own, for which nixlMetadataWorker is the
 * shared machinery. The manager does no scheduling and holds no thread, so a
 * backend blocked on its store does not stall the others.
 */
class nixlMetadataBackend {
public:
    virtual ~nixlMetadataBackend() = default;

    /** Stable transport name reported by nixlMDManager::backendName(). */
    [[nodiscard]] virtual std::string_view
    name() const = 0;

    /** Publish the full local metadata blob. */
    [[nodiscard]] virtual nixl_status_t
    sendLocal(const nixl_opt_args_t *extra_params) = 0;

    /** Publish a partial local metadata blob. */
    [[nodiscard]] virtual nixl_status_t
    sendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) = 0;

    /** Initiate retrieval of a remote agent's metadata. */
    [[nodiscard]] virtual nixl_status_t
    fetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) = 0;

    /** Withdraw our metadata. */
    [[nodiscard]] virtual nixl_status_t
    invalidateLocal(const nixl_opt_args_t *extra_params) = 0;

    /**
     * @brief Whether this backend runs a thread of its own. That thread shares
     *        agent state, so this is what decides the agent's effective sync
     *        mode. Default false (a backend that does everything synchronously).
     */
    [[nodiscard]] virtual bool
    usesThread() const {
        return false;
    }

    /**
     * @brief Begin background work. Called once by the agent after construction
     *        completes, so a backend thread never touches half-built agent
     *        state. Default no-op.
     */
    virtual void
    start() {}

    /**
     * @brief Finish what is pending, then stop and join. Idempotent, and called
     *        before the agent tears down the state a task could touch. Default
     *        no-op.
     */
    virtual void
    stop() {}
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_BACKEND_H
