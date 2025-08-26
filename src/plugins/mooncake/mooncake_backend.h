/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef __MOONCAKE_BACKEND_H
#define __MOONCAKE_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <unordered_set>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "common/str_tools.h"
#include "common/nixl_time.h"

#include "transfer_engine_c.h"

class nixlMooncakeBackendMD;

class nixlMooncakeEngine : public nixlBackendEngine {
public:
    nixlMooncakeEngine(const nixlBackendInitParams *init_params);
    ~nixlMooncakeEngine();

    bool
    supportsRemote() const {
        return true;
    }

    bool
    supportsLocal() const {
        return true;
    }

    bool
    supportsNotif() const {
        return true;
    }

    bool
    supportsProgTh() const {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const;

    /* Object management */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const;
    nixl_status_t
    getConnInfo(std::string &str) const;
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent, const std::string &remote_conn_info);

    nixl_status_t
    connect(const std::string &remote_agent);
    nixl_status_t
    disconnect(const std::string &remote_agent);

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out);
    nixl_status_t
    deregisterMem(nixlBackendMD *meta);

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output);

    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output);
    nixl_status_t
    unloadMD(nixlBackendMD *input);

    // Data transfer
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const;

    nixl_status_t
    getNotifs(notif_list_t &notif_list);
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;

private:
    struct AgentInfo {
        int segment_id;
    };

    mutable std::mutex mutex_;
    transfer_engine_t engine_;
    std::string local_agent_name_;
    std::unordered_map<uint64_t, nixlMooncakeBackendMD *> mem_reg_info_;
    std::unordered_map<std::string, AgentInfo> connected_agents_;
};

#endif
