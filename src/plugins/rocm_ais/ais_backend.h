/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef NIXL_SRC_PLUGINS_ROCM_AIS_AIS_BACKEND_H
#define NIXL_SRC_PLUGINS_ROCM_AIS_AIS_BACKEND_H

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipfile.h>

#include <nixl.h>
#include <nixl_types.h>

#include "ais_utils.h"
#include "backend/backend_engine.h"
#include "file/file_engine_base.h"
#include "file/file_path_mode.h"

// One logical mem<->file I/O produced by the shared preparation path. The
// concrete engines turn these into their own posted, pollable request handles.
struct aisXferReq {
    void *addr;
    size_t size;
    size_t file_offset;
    hipFileHandle_t fh;
    hipFileOpcode_t op;
    // AMD GPUs are per-thread current, so each worker re-selects the device
    // owning the buffer before issuing I/O. -1 means "host memory, no select".
    int dev_id;
};

// Abstract base for the AIS (AMD Infinity Storage) family of backends. It owns
// everything that is independent of how I/O is submitted: the hipFile driver
// lifecycle, memory and file registration (refcounted handle cache + buffer
// RAII), queryMem, and the prepXfer preamble (validation + descriptor->aisXferReq
// translation). The only backend-specific step in preparation is finalizePrep();
// the concrete engines additionally implement the transfer-execution virtuals
// (postXfer/checkXfer/releaseReqH) directly.
//
// Today the only leaf is nixlAisMtEngine ("AIS_MT"). A batch engine mirroring
// GDS's cuFile batch path slots in alongside it once hipFile's batch API is
// adopted; finalizePrep is the hook it needs.
//
// The connection/metadata entry points and supported-memory set are shared with
// the other local file backends via FileEngineBase.
class nixlAisEngine : public FileEngineBase {
public:
    explicit nixlAisEngine(const nixlBackendInitParams *init_params);
    ~nixlAisEngine() override = default;

    nixlAisEngine(const nixlAisEngine &) = delete;
    nixlAisEngine &
    operator=(const nixlAisEngine &) = delete;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    // Shared: validates the request and translates descriptors into an
    // aisXferReq list, then defers the concrete handle creation to finalizePrep.
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
    finalizePrep(std::vector<aisXferReq> &&reqs, nixlBackendReqH *&handle) const = 0;

private:
    std::unique_ptr<aisDriverHandle> driver_;
    // Keyed on the open fd (FileFd::fd()), not on the descriptor's devId: in
    // path-mode devId is a caller-chosen key with no relation to any fd. Holds
    // weak refs so the last registration of an fd drops the hipFile handle.
    std::unordered_map<int, std::weak_ptr<aisFileHandle>> ais_file_map_;
    nixl::PathModeDevIdRegistry path_mode_devids_;
};

#endif // NIXL_SRC_PLUGINS_ROCM_AIS_AIS_BACKEND_H
