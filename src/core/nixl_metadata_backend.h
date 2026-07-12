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

#include <functional>
#include <string>
#include <string_view>

/** A unit of transport I/O produced on the caller thread, run on the worker. */
using nixlWorkerTask = std::function<void()>;

/**
 * @struct nixlPreparedOp
 * @brief Result of a backend's caller-thread prepare step.
 *
 * @var status  Synchronous validation/serialization result, returned to the
 *              caller. Anything other than NIXL_SUCCESS means the op was rejected
 *              and no task should run.
 * @var task    The transport work to run on the manager's worker thread. Empty
 *              when there is nothing to schedule.
 */
struct nixlPreparedOp {
    nixl_status_t status = NIXL_SUCCESS;
    nixlWorkerTask task;
};

/**
 * @class nixlMetadataBackend
 * @brief Metadata-exchange contract that nixlMDManager dispatches to.
 *
 * Each transport implements this contract (P2P, ETCD, TCPStore). Core-internal:
 * not part of the installed public headers, so backend dependencies never leak
 * into the public API. Operational addressing (`ipAddr`/`port`, `metadataLabel`)
 * is carried in `nixl_opt_args_t`.
 *
 * Thread contract, encoded in the interface:
 *  - the `prepare*` methods run on the CALLER thread: they validate and
 *    serialize, return a synchronous status, and hand back the transport work as
 *    a nixlWorkerTask. They must not block on I/O.
 *  - the returned nixlWorkerTask and `serviceEvents()` run on the manager's
 *    WORKER thread: that is where all blocking transport I/O belongs.
 * The manager owns scheduling; backends never touch a queue or a thread.
 */
class nixlMetadataBackend {
public:
    virtual ~nixlMetadataBackend() = default;

    /** Stable transport name reported by nixlMDManager::backendName(). */
    [[nodiscard]] virtual std::string_view
    name() const = 0;

    /** Caller thread: prepare a full-metadata publish. */
    [[nodiscard]] virtual nixlPreparedOp
    prepareSendLocal(const nixl_opt_args_t *extra_params) = 0;

    /** Caller thread: prepare a partial-metadata publish. */
    [[nodiscard]] virtual nixlPreparedOp
    prepareSendLocalPartial(const nixl_reg_dlist_t &descs, const nixl_opt_args_t *extra_params) = 0;

    /** Caller thread: prepare retrieval of a remote agent's metadata. */
    [[nodiscard]] virtual nixlPreparedOp
    prepareFetchRemote(const std::string &remote_name, const nixl_opt_args_t *extra_params) = 0;

    /** Caller thread: prepare withdrawal of our metadata. */
    [[nodiscard]] virtual nixlPreparedOp
    prepareInvalidateLocal(const nixl_opt_args_t *extra_params) = 0;

    /**
     * @brief Whether this backend needs the manager's worker thread running
     *        (for background servicing and/or to execute its tasks). Default
     *        false (a backend that does nothing off-thread).
     */
    [[nodiscard]] virtual bool
    needsWorker() const {
        return false;
    }

    /**
     * @brief Worker thread: one pass of background servicing, called repeatedly
     *        (e.g. accept peers / read replies for P2P, drain watch invalidations
     *        for ETCD). Default no-op.
     */
    virtual void
    serviceEvents() {}
};

#endif // NIXL_SRC_CORE_NIXL_METADATA_BACKEND_H
