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
#ifndef __GDS_BATCH_ENGINE_H
#define __GDS_BATCH_ENGINE_H

#include <memory>
#include <mutex>
#include <vector>

#include <cufile.h>

#include "gds_backend.h"

// Wraps a cuFile batch I/O context (setup/submit/status/cancel).
class nixlGdsIOBatch {
public:
    explicit nixlGdsIOBatch(unsigned int size);
    ~nixlGdsIOBatch();

    nixl_status_t
    addToBatch(CUfileHandle_t fh,
               void *buffer,
               size_t size,
               size_t file_offset,
               size_t ptr_offset,
               CUfileOpcode_t type);
    nixl_status_t
    submitBatch(int flags);
    nixl_status_t
    checkStatus();
    nixl_status_t
    cancelBatch();
    void
    reset();

    bool
    isValid() const {
        return batch_handle != nullptr && init_err.err == CU_FILE_SUCCESS;
    }

private:
    CUfileBatchHandle_t batch_handle = nullptr;
    std::unique_ptr<CUfileIOEvents_t[]> io_batch_events;
    std::unique_ptr<CUfileIOParams_t[]> io_batch_params;
    CUfileError_t init_err{};
    unsigned int max_reqs = 0;
    unsigned int batch_size = 0;
    unsigned int entries_completed = 0;
    bool active = false;
    nixl_status_t current_status = NIXL_ERR_NOT_POSTED;
};

class nixlGdsBatchReqH : public nixlBackendReqH {
public:
    ~nixlGdsBatchReqH() override = default;

    std::vector<GdsXferReq> request_list;
    std::vector<nixlGdsIOBatch *> batch_io_list;
    nixl_status_t overall_status = NIXL_SUCCESS;
};

// "GDS" backend: cuFile batch transfers. Large transfers are split by
// max_request_size and submitted in batches of up to batch_limit entries.
//
// Inherits from nixlGdsEngine (see gds_backend.h): registerMem/deregisterMem,
// queryMem, the cuFile driver lifecycle, and the prepXfer preamble (validation +
// descriptor->GdsXferReq, which then calls finalizePrep below). This class only
// implements the transfer mechanism: finalizePrep + postXfer/checkXfer/
// releaseReqH.
class nixlGdsBatchEngine : public nixlGdsEngine {
public:
    explicit nixlGdsBatchEngine(const nixlBackendInitParams *init_params);
    ~nixlGdsBatchEngine() override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

protected:
    nixl_status_t
    finalizePrep(std::vector<GdsXferReq> &&reqs, nixlBackendReqH *&handle) const override;

private:
    nixlGdsIOBatch *
    getBatchFromPool(unsigned int size) const;
    void
    returnBatchToPool(nixlGdsIOBatch *batch) const;
    nixl_status_t
    createAndSubmitBatch(const std::vector<GdsXferReq> &requests,
                         size_t start_idx,
                         size_t batch_size,
                         nixlGdsIOBatch *&batch_out) const;
    nixl_status_t
    cancelAndReclaimBatches(std::vector<nixlGdsIOBatch *> &batch_list) const;

    mutable std::mutex batch_pool_lock_;
    mutable std::vector<nixlGdsIOBatch *> batch_pool_;
    std::vector<std::unique_ptr<nixlGdsIOBatch>> batch_storage_;
    unsigned int batch_pool_size_ = 0;
    unsigned int batch_limit_ = 0;
    unsigned int max_request_size_ = 0;
};

#endif // __GDS_BATCH_ENGINE_H
