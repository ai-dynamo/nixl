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
#ifndef __UCCL_BACKEND_H
#define __UCCL_BACKEND_H

#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <thread>
#include <atomic>
#include <map>

#include "nixl.h"
#include <nixl_types.h>
#include "backend/backend_engine.h"
#include "common/nixl_log.h"

#include "uccl_engine.h"

#define FIFO_ITEM_SIZE 64

class nixlUcclBackendMD;
class nixlUcclReqH;

class nixlUcclEngine : public nixlBackendEngine {
public:
    nixlUcclEngine(const nixlBackendInitParams *init_params);
    ~nixlUcclEngine();

    bool
    supportsRemote() const {
        return true;
    }

    bool
    supportsLocal() const {
        // TODO: Enable this when local transfers are supported
        return false;
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

    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const;
    nixl_status_t
    getConnInfo(std::string &str) const;
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent, const std::string &remote_conn_info);

    nixl_status_t
    connect(const std::string &remote_agent) override;
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
    void
    startListener();

    mutable std::mutex mem_mutex_; // mem_reg_info_ mutex
    mutable std::mutex conn_mutex_; // connected_agents_ mutex
    uccl_engine_t *engine_;
    std::string local_agent_name_;
    std::unordered_map<uint64_t, nixlUcclBackendMD *> mem_reg_info_;
    std::unordered_map<std::string, uint64_t> connected_agents_; // agent name -> conn_id
    std::thread listener_thread_;
};

// UCCL Backend Memory Descriptor
class nixlUcclBackendMD : public nixlBackendMD {
public:
    nixlUcclBackendMD(bool isPrivate) : nixlBackendMD(isPrivate) {}

    virtual ~nixlUcclBackendMD() {}

    void *addr;
    size_t length;
    int ref_cnt;
    uint64_t mr_id; // UCCL memory region id
    char fifo_item_data[FIFO_ITEM_SIZE];
};

// UCCL Backend Request Handle
class nixlUcclReqH : public nixlBackendReqH {
public:
    nixlUcclReqH(uccl_conn_t *conn) : conn(conn) {}

    virtual ~nixlUcclReqH() {}

    uccl_conn_t *conn;
    std::unordered_set<uint64_t> pending_transfer_ids;
    nixl_blob_t notif_msg;
};

#endif
