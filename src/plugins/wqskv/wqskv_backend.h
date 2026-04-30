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

/**
 * @file wqskv_backend.h
 * @brief WQSKV Backend Header - NIXL backend over WDS KV cache vendor lib
 *
 * Wraps the C-style vendor API exposed by libwclient_kvcache.so
 * (wds_kvcache_init / wds_kvcache_put / wds_kvcache_get_vec, etc.) so a
 * NIXL agent can PUT/GET DRAM buffers into WDS without going through the
 * mooncake store. Local-only backend (supportsRemote=false), DRAM_SEG only.
 *
 * postXfer is asynchronous: each descriptor maps to one vendor call; the
 * vendor invokes a per-request callback that decrements a pending counter
 * shared via the request handle. checkXfer reads that counter without
 * locks; releaseReqH waits on a condition variable until the counter hits
 * zero, then frees the handle.
 *
 * wds_kvcache_init is called at most once per process via std::call_once;
 * subsequent createBackend calls reuse the global init state and only set
 * per-engine state (localAgent, devIdToKey_).
 */

#ifndef WQSKV_BACKEND_H
#define WQSKV_BACKEND_H

#include "backend/backend_engine.h"
#include "nixl_types.h"
#include <string>
#include <unordered_map>
#include <vector>

class nixlWQSKVEngine : public nixlBackendEngine {
public:
    explicit nixlWQSKVEngine(const nixlBackendInitParams *init_params);

    ~nixlWQSKVEngine() override = default;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG};
    }

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_status_t
    connect(const std::string &remote_agent) override;

    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

private:
    std::string
    getLocalAgent() const {
        return localAgent;
    }

    // Maps NIXL devId -> vendor key, populated by registerMem and consulted
    // by postXfer when the remote descriptor lacks a metadataP pointer (mirrors
    // the mockkv fallback path).
    std::unordered_map<uint64_t, std::string> devIdToKey_;

    std::string localAgent;
};

#endif // WQSKV_BACKEND_H
