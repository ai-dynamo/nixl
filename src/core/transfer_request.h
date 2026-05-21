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

class nixlXferReqH {
public:
    const std::string remoteAgent;
    const nixl_xfer_op_t operation;

    nixlXferReqH(nixlXferReqH &&) = delete;
    nixlXferReqH(const nixlXferReqH &) = delete;

    void
    operator=(nixlXferReqH &&) = delete;
    void
    operator=(const nixlXferReqH &) = delete;

    friend class nixlAgent;

protected:
    nixlXferReqH(const std::string &remote_agent,
                 const nixl_xfer_op_t operation)
        : remoteAgent(remote_agent),
          operation(operation) {}

    virtual ~nixlXferReqH() {
        if ((backendHandle != nullptr) && (engine != nullptr)) {
            engine->releaseReqH(backendHandle);
        }
    }

    void
    updateRequestStats(nixlTelemetry *telemetry, nixl_telemetry_stat_status_t stat_status);

    nixlBackendEngine *engine = nullptr;
    nixlBackendReqH *backendHandle = nullptr;

    nixl_status_t status = NIXL_ERR_NOT_POSTED;
    nixl_xfer_telem_t telemetry;
};

class nixlXferReqRW
    : public nixlXferReqH {
public:

    nixlXferReqRW(const std::string &remote_agent,
                  const nixl_xfer_op_t operation,
                  const nixl_mem_t local_type,
                  const nixl_mem_t remote_type,
                  const std::size_t desc_count = 0);

    ~nixlXferReqRW() = default;

    friend class nixlAgent;

private:
    nixl_meta_dlist_t initiatorDescs;
    nixl_meta_dlist_t targetDescs;

    nixl_blob_t notifMsg;
    bool hasNotif = false;
};

class nixlXferReqSR
    : public nixlXferReqH {
public:
    const std::string tag;

    nixlXferReqSR(const std::string &remote_agent,
                  const nixl_xfer_op_t operation,
                  const std::string &tag,
                  const nixl_mem_t local_type,
                  const std::size_t desc_count = 0);

    friend class nixlAgent;

private:
    nixl_meta_dlist_t initiatorDescs;
};

struct nixlDlistH {
    using descs_t = std::unordered_map<nixlBackendEngine *, std::unique_ptr<nixl_meta_dlist_t>>;

    nixlDlistH(const std::string &remote_agent, descs_t &&descs);

    const std::string remoteAgent; // Empty means "local".
    const descs_t descs;
};

#endif
