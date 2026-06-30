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
#ifndef __GDS_BACKEND_H
#define __GDS_BACKEND_H

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cufile.h>

#include <nixl.h>
#include <nixl_types.h>

#include "backend/backend_engine.h"
#include "file/file_path_mode.h"
#include "gds_utils.h"

// One logical mem<->file I/O produced by the shared preparation path. The
// concrete engines turn these into their own posted, pollable request handles.
struct GdsXferReq {
    void *addr;
    size_t size;
    size_t file_offset;
    CUfileHandle_t fh;
    CUfileOpcode_t op;
};

// Abstract base for the GDS family of backends. It owns everything that is
// identical between "GDS" and "GDS_MT": the cuFile driver lifecycle, memory and
// file registration (refcounted handle cache + buffer RAII), queryMem, and the
// prepXfer preamble (validation + descriptor->GdsXferReq translation). The only
// backend-specific step in preparation is finalizePrep(); the concrete engines
// additionally implement the transfer-execution virtuals (postXfer/checkXfer/
// releaseReqH) directly.
class nixlGdsEngine : public nixlBackendEngine {
public:
    explicit nixlGdsEngine(const nixlBackendInitParams *init_params);
    ~nixlGdsEngine() override = default;

    nixlGdsEngine(const nixlGdsEngine &) = delete;
    nixlGdsEngine &
    operator=(const nixlGdsEngine &) = delete;

    bool
    supportsNotif() const override {
        return false;
    }

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, VRAM_SEG, FILE_SEG};
    }

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    // Shared: validates the request and translates descriptors into a
    // GdsXferReq list, then defers the concrete handle creation to finalizePrep.
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    // postXfer / checkXfer / releaseReqH remain pure virtual here (inherited from
    // nixlBackendEngine) and are implemented by the concrete engines.

protected:
    // The single backend-specific step of preparation: build a concrete,
    // posted-ready request handle from the validated logical request list.
    virtual nixl_status_t
    finalizePrep(std::vector<GdsXferReq> &&reqs, nixlBackendReqH *&handle) const = 0;

private:
    std::unique_ptr<gdsDriverHandle> driver_;
    std::unordered_map<int, std::weak_ptr<gdsFileHandle>> gds_file_map_;
    nixl::PathModeDevIdRegistry path_mode_devids_;
};

#endif // __GDS_BACKEND_H
