/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_CORE_TRANSFER_REQUEST_H
#define NIXL_SRC_CORE_TRANSFER_REQUEST_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "nixl_types.h"
#include "backend_engine.h"
#include "telemetry.h"

enum nixl_telemetry_stat_status_t {
    NIXL_TELEMETRY_POST = 0,
    NIXL_TELEMETRY_POST_AND_FINISH = 1,
    NIXL_TELEMETRY_FINISH = 2
};

// Contains pointers to corresponding backend engine and its handler, and populated
// and verified DescLists, and other state and metadata needed for a NIXL transfer
class nixlXferReqH {
private:
    nixlBackendEngine *engine = nullptr;
    nixlBackendReqH *backendHandle = nullptr;

    const std::unique_ptr<nixl_meta_dlist_t> initiatorDescs;
    const std::unique_ptr<nixl_meta_dlist_t> targetDescs;

    const std::string remoteAgent;
    nixl_blob_t notifMsg;
    bool hasNotif = false;

    const nixl_xfer_op_t backendOp;
    nixl_status_t status = NIXL_ERR_NOT_POSTED;

    nixl_xfer_telem_t telemetry;

public:
    nixlXferReqH(const std::string remote_agent,
                 const nixl_xfer_op_t backend_op,
                 const nixl_mem_t local_type,
                 const nixl_mem_t remote_type,
                 const size_t desc_count = 0);

    nixlXferReqH(nixlXferReqH &&) = delete;
    nixlXferReqH(const nixlXferReqH &) = delete;

    void
    operator=(nixlXferReqH &&) = delete;
    void
    operator=(const nixlXferReqH &) = delete;

    ~nixlXferReqH() {
        if (backendHandle != nullptr) {
            engine->releaseReqH(backendHandle);
        }
    }

    void
    updateRequestStats(nixlTelemetry *telemetry, nixl_telemetry_stat_status_t stat_status);

    friend class nixlAgent;
};

struct nixlDlistH {
    using descs_t = std::unordered_map<nixlBackendEngine *, std::unique_ptr<nixl_meta_dlist_t>>;
    const descs_t descs;

    const std::string remoteAgent;
    const bool isLocal;

    nixlDlistH(const bool is_local, const std::string &remote_agent, descs_t &&descs);

    nixlDlistH(nixlDlistH &&) = delete;
    nixlDlistH(const nixlDlistH &) = delete;

    void
    operator=(nixlDlistH &&) = delete;
    void
    operator=(const nixlDlistH &) = delete;
};

#endif
