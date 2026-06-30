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
#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_batch_engine.h"

namespace {
/** Setting the default values to check the batch limit */
constexpr unsigned DEFAULT_BATCH_LIMIT = 128;
/** Setting the max request size to 16 MB */
constexpr unsigned DEFAULT_MAX_REQUEST_SIZE = 16 * 1024 * 1024; // 16MB
/** Create a batch pool of size 16 */
constexpr unsigned DEFAULT_BATCH_POOL_SIZE = 16;

size_t
ceilDiv(size_t value, size_t divisor) {
    return (value / divisor) + ((value % divisor) != 0);
}
} // namespace

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
    : io_batch_events(std::make_unique<CUfileIOEvents_t[]>(size)),
      io_batch_params(std::make_unique<CUfileIOParams_t[]>(size)),
      max_reqs(size) {

    const CUfileError_t err = cuFileBatchIOSetUp(&batch_handle, size);
    if (err.err != 0) {
        NIXL_ERROR << "Error in setting up Batch";
        init_err = err;
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch() {
    if (active) {
        NIXL_ERROR << "GDS: destroying an active batch; canceling outstanding I/O";
        cancelBatch();
    }
    if (batch_handle != nullptr) {
        cuFileBatchIODestroy(batch_handle);
    }
}

nixl_status_t
nixlGdsIOBatch::addToBatch(CUfileHandle_t fh,
                           void *buffer,
                           size_t size,
                           size_t file_offset,
                           size_t ptr_offset,
                           CUfileOpcode_t type) {
    if (!isValid() || active || batch_size >= max_reqs) {
        return NIXL_ERR_BACKEND;
    }

    CUfileIOParams_t *params = &io_batch_params[batch_size];
    *params = {};
    params->mode = CUFILE_BATCH;
    params->fh = fh;
    params->u.batch.devPtr_base = buffer;
    params->u.batch.file_offset = file_offset;
    params->u.batch.devPtr_offset = ptr_offset;
    params->u.batch.size = size;
    params->opcode = type;
    params->cookie = params;
    batch_size++;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::cancelBatch() {
    if (!active) {
        return NIXL_SUCCESS;
    }
    const CUfileError_t err = cuFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        NIXL_ERROR << "Error in canceling batch";
        return NIXL_ERR_BACKEND;
    }
    active = false;
    current_status = NIXL_ERR_CANCELED;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::submitBatch(int flags) {
    if (!isValid() || batch_size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const CUfileError_t err =
        cuFileBatchIOSubmit(batch_handle, batch_size, io_batch_params.get(), flags);
    if (err.err != 0) {
        NIXL_ERROR << "Error submitting GDS batch";
        current_status = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }
    active = true;
    current_status = NIXL_IN_PROG;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::checkStatus() {
    if (current_status != NIXL_IN_PROG) {
        return current_status;
    }

    if (entries_completed > batch_size) {
        current_status = NIXL_ERR_UNKNOWN;
        return current_status;
    }

    const unsigned int entries_remaining = batch_size - entries_completed;
    unsigned int nr = entries_remaining;
    // TODO: A follow-up should make status polling and active release
    // nonblocking. min_nr intentionally remains equal to nr here to preserve
    // the pre-consolidation GDS completion behavior; changing it needs separate
    // API and performance validation.
    const CUfileError_t errBatch =
        cuFileBatchIOGetStatus(batch_handle, nr, &nr, io_batch_events.get(), nullptr);
    if (errBatch.err != 0) {
        NIXL_ERROR << "Error in IO Batch Get Status";
        current_status = NIXL_ERR_BACKEND;
        return current_status;
    }

    if (nr > entries_remaining) {
        current_status = NIXL_ERR_UNKNOWN;
        return current_status;
    }

    const bool all_entries_reported = (nr == entries_remaining);
    for (unsigned int i = 0; i < nr; ++i) {
        const CUfileIOEvents_t &event = io_batch_events[i];
        if (event.status != CUFILE_COMPLETE || event.cookie == nullptr) {
            NIXL_ERROR << "GDS batch entry failed with status " << event.status;
            if (all_entries_reported) {
                active = false;
            }
            current_status = NIXL_ERR_BACKEND;
            return current_status;
        }

        const auto *params = static_cast<const CUfileIOParams_t *>(event.cookie);
        if (event.ret != params->u.batch.size) {
            NIXL_ERROR << "GDS batch entry completed " << event.ret << " of "
                       << params->u.batch.size << " bytes";
            if (all_entries_reported) {
                active = false;
            }
            current_status = NIXL_ERR_BACKEND;
            return current_status;
        }
    }

    entries_completed += nr;
    if (entries_completed == batch_size) {
        active = false;
        current_status = NIXL_SUCCESS;
    } else {
        current_status = NIXL_IN_PROG;
    }

    return current_status;
}

void
nixlGdsIOBatch::reset() {
    if (active) {
        NIXL_ERROR << "GDS: attempted to reset an active batch";
        return;
    }
    entries_completed = 0;
    batch_size = 0;
    current_status = NIXL_ERR_NOT_POSTED;
}

nixlGdsBatchEngine::nixlGdsBatchEngine(const nixlBackendInitParams *init_params)
    : nixlGdsEngine(init_params) {
    // Base ctor opened the cuFile driver; bail if that failed.
    if (this->initErr) {
        return;
    }

    try {
        nixl_b_params_t *custom_params = init_params->customParams;
        batch_pool_size_ = nixl::getBackendParamDefaulted(
            custom_params, "batch_pool_size", DEFAULT_BATCH_POOL_SIZE);
        batch_limit_ =
            nixl::getBackendParamDefaulted(custom_params, "batch_limit", DEFAULT_BATCH_LIMIT);
        max_request_size_ = nixl::getBackendParamDefaulted(
            custom_params, "max_request_size", DEFAULT_MAX_REQUEST_SIZE);

        if (batch_pool_size_ == 0 || batch_limit_ == 0 || max_request_size_ == 0) {
            throw std::invalid_argument(
                "GDS: batch_pool_size, batch_limit, and max_request_size must be greater than "
                "zero");
        }

        batch_pool_.reserve(batch_pool_size_);
        batch_storage_.reserve(batch_pool_size_);
        for (unsigned int i = 0; i < batch_pool_size_; i++) {
            auto batch = std::make_unique<nixlGdsIOBatch>(batch_limit_);
            if (!batch->isValid()) {
                throw std::runtime_error("GDS: failed to initialize cuFile batch pool");
            }
            batch_pool_.push_back(batch.get());
            batch_storage_.push_back(std::move(batch));
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        this->initErr = true;
    }
}

nixlGdsBatchEngine::~nixlGdsBatchEngine() {
    batch_pool_.clear();
    batch_storage_.clear();
}

nixlGdsIOBatch *
nixlGdsBatchEngine::getBatchFromPool(unsigned int /*size*/) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock_);
    if (!batch_pool_.empty()) {
        nixlGdsIOBatch *batch = batch_pool_.back();
        batch_pool_.pop_back();
        batch->reset();
        return batch;
    }
    // Pool exhausted - don't create new batches in the data path.
    return nullptr;
}

void
nixlGdsBatchEngine::returnBatchToPool(nixlGdsIOBatch *batch) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock_);
    batch_pool_.push_back(batch);
}

nixl_status_t
nixlGdsBatchEngine::finalizePrep(std::vector<gdsXferReq> &&reqs, nixlBackendReqH *&handle) const {
    auto gds_handle = std::make_unique<nixlGdsBatchReqH>();

    size_t chunk_count = 0;
    bool can_reuse_requests = true;
    const size_t max_request_size = max_request_size_;
    for (const gdsXferReq &req : reqs) {
        if (!req.addr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const size_t chunks = (req.size / max_request_size) + ((req.size % max_request_size) != 0);
        can_reuse_requests &= (chunks == 1);
        if (chunks > std::numeric_limits<size_t>::max() - chunk_count) {
            return NIXL_ERR_INVALID_PARAM;
        }
        chunk_count += chunks;
    }

    if (chunk_count == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (can_reuse_requests) {
        gds_handle->request_list = std::move(reqs);
    } else {
        // Split large transfers into multiple requests bounded by max_request_size.
        gds_handle->request_list.reserve(chunk_count);
        for (const gdsXferReq &req : reqs) {
            size_t remaining_size = req.size;
            size_t current_offset = 0;
            while (remaining_size > 0) {
                const size_t request_size = std::min(remaining_size, max_request_size);

                gdsXferReq chunk;
                chunk.addr = (char *)req.addr + current_offset;
                chunk.size = request_size;
                chunk.file_offset = req.file_offset + current_offset;
                chunk.fh = req.fh;
                chunk.op = req.op;
                gds_handle->request_list.push_back(chunk);

                remaining_size -= request_size;
                current_offset += request_size;
            }
        }
    }

    const size_t request_count = gds_handle->request_list.size();
    const size_t batch_count = ceilDiv(request_count, batch_limit_);
    if (batch_count > batch_pool_size_) {
        NIXL_ERROR << "GDS: transfer requires " << batch_count << " batches but the pool has "
                   << batch_pool_size_;
        return NIXL_ERR_BACKEND;
    }
    gds_handle->batch_io_list.reserve(batch_count);

    handle = gds_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsBatchEngine::createAndSubmitBatch(const std::vector<gdsXferReq> &requests,
                                         size_t start_idx,
                                         size_t batch_size,
                                         nixlGdsIOBatch *&batch_out) const {
    batch_out = nullptr;
    nixlGdsIOBatch *batch = getBatchFromPool(batch_size);
    if (!batch) {
        NIXL_ERROR << "GDS batch pool exhausted";
        return NIXL_ERR_BACKEND;
    }

    for (size_t i = 0; i < batch_size; i++) {
        const auto &req = requests[start_idx + i];
        if (!req.addr || !req.fh) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl_status_t status =
            batch->addToBatch(req.fh, req.addr, req.size, req.file_offset, 0, req.op);
        if (status != NIXL_SUCCESS) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixl_status_t status = batch->submitBatch(0);
    if (status != NIXL_SUCCESS) {
        returnBatchToPool(batch);
        return NIXL_ERR_BACKEND;
    }

    batch_out = batch;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsBatchEngine::cancelAndReclaimBatches(std::vector<nixlGdsIOBatch *> &batch_list) const {
    nixl_status_t status = NIXL_SUCCESS;
    auto keep = batch_list.begin();

    for (nixlGdsIOBatch *batch : batch_list) {
        if (batch == nullptr) {
            continue;
        }
        if (batch->cancelBatch() == NIXL_SUCCESS) {
            // TODO: Establish and test cancel-to-resubmit semantics for every
            // cuFile I/O path before immediately reusing a batch handle.
            returnBatchToPool(batch);
        } else {
            *keep++ = batch;
            status = NIXL_ERR_BACKEND;
        }
    }

    batch_list.erase(keep, batch_list.end());
    return status;
}

nixl_status_t
nixlGdsBatchEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);

    if (gds_handle->request_list.empty()) {
        NIXL_ERROR << "Empty request list";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!gds_handle->batch_io_list.empty()) {
        return NIXL_ERR_REPOST_ACTIVE;
    }

    const auto &request_list = gds_handle->request_list;
    const size_t batch_count = ceilDiv(request_list.size(), batch_limit_);
    if (batch_count > batch_pool_size_) {
        return NIXL_ERR_BACKEND;
    }

    gds_handle->overall_status = NIXL_SUCCESS;
    gds_handle->batch_io_list.assign(batch_count, nullptr);

    size_t current_req = 0;
    for (size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        const size_t batch_size =
            std::min(request_list.size() - current_req, static_cast<size_t>(batch_limit_));
        const nixl_status_t status = createAndSubmitBatch(
            request_list, current_req, batch_size, gds_handle->batch_io_list[batch_index]);
        if (status != NIXL_SUCCESS) {
            gds_handle->overall_status = status;
            if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
                return NIXL_ERR_BACKEND;
            }
            return status;
        }
        current_req += batch_size;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsBatchEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);

    if (gds_handle->batch_io_list.empty()) {
        return gds_handle->overall_status;
    }

    auto current = gds_handle->batch_io_list.begin();
    while (current != gds_handle->batch_io_list.end()) {
        nixlGdsIOBatch *batch = *current;
        const nixl_status_t status = batch->checkStatus();

        if (status == NIXL_IN_PROG) {
            ++current;
            continue;
        }
        if (status == NIXL_SUCCESS) {
            returnBatchToPool(batch);
            current = gds_handle->batch_io_list.erase(current);
            continue;
        }
        if (gds_handle->overall_status == NIXL_SUCCESS) {
            gds_handle->overall_status = status;
        }
        ++current;
    }

    if (gds_handle->overall_status != NIXL_SUCCESS) {
        if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
            gds_handle->overall_status = NIXL_ERR_BACKEND;
        }
        return gds_handle->overall_status;
    }

    return gds_handle->batch_io_list.empty() ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlGdsBatchEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);
    if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    delete gds_handle;
    return NIXL_SUCCESS;
}
