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
#include "common/nixl_time.h"

#include "transfer_engine_c.h"

// TENT (Transfer Engine NexT) is the next-generation Mooncake engine, driven
// through its native C API when Mooncake was built with -DUSE_TENT=ON. Its
// header is included by the implementation only: nothing here may depend on
// HAVE_MOONCAKE_TENT, or this class would have a different layout in
// translation units that do not define it (the unit test is one), so the
// engine handle below is kept type-erased.

class nixlMooncakeBackendMD;
struct nixlMooncakeBackendReqH;

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
    // Which Mooncake engine the backend drives. Classic is the default and
    // keeps the original code path unchanged; Tent selects the next-gen
    // engine through its native C API (tent_*).
    enum class mode { CLASSIC, TENT };

    struct AgentInfo {
        // Classic segment ids are int32_t, TENT segment handles are uint64_t;
        // stored wide and narrowed at the classic call sites.
        uint64_t segment_id;
    };

    nixl_status_t
    postXferClassic(const nixl_xfer_op_t &operation,
                    const nixl_meta_dlist_t &local,
                    const nixl_meta_dlist_t &remote,
                    uint64_t segment_id,
                    nixlMooncakeBackendReqH *priv,
                    const nixl_opt_b_args_t *opt_args) const;
    nixl_status_t
    checkXferClassic(nixlMooncakeBackendReqH *priv) const;
    nixl_status_t
    releaseReqHClassic(nixlMooncakeBackendReqH *priv) const;

    nixl_status_t
    postXferTent(const nixl_xfer_op_t &operation,
                 const nixl_meta_dlist_t &local,
                 const nixl_meta_dlist_t &remote,
                 uint64_t segment_id,
                 nixlMooncakeBackendReqH *priv,
                 const nixl_opt_b_args_t *opt_args) const;
    nixl_status_t
    checkXferTent(nixlMooncakeBackendReqH *priv) const;
    nixl_status_t
    releaseReqHTent(nixlMooncakeBackendReqH *priv) const;

    // Frees the batches parked by releaseReqHTent() as soon as the engine
    // reports them terminal. A no-op while nothing is parked, which is the
    // common case.
    void
    reclaimParkedBatches() const;

    mode mode_ = mode::CLASSIC;
    // Sentinel for "no batch allocated": the classic engine reports failure as
    // INVALID_BATCH (UINT64_MAX), TENT as 0.
    uint64_t invalid_batch_ = INVALID_BATCH;

    mutable std::mutex mutex_;
    transfer_engine_t engine_ = nullptr;
    // tent_engine_t, kept as void * so this header stays independent of the
    // TENT headers and of HAVE_MOONCAKE_TENT (see the note above).
    void *tent_engine_ = nullptr;
    // Batches whose release was asked for while transfers were still running.
    // Kept under their own lock so the sweep never holds mutex_ across an
    // engine call.
    mutable std::mutex parked_mutex_;
    mutable std::vector<uint64_t> parked_batches_;
    const std::string local_agent_name_;
    std::unordered_map<uint64_t, nixlMooncakeBackendMD *> mem_reg_info_;
    std::unordered_map<std::string, AgentInfo> connected_agents_;
};

#endif
