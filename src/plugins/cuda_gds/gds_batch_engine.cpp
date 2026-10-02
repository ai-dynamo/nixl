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

// Teardown only. cuFile keeps running the I/O of a canceled batch, so a batch
// is never canceled to be reused
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
    // Poll without blocking: min_nr 0 with a zero timeout returns what is ready
    struct timespec poll_timeout = {0, 0};
    const CUfileError_t errBatch =
        cuFileBatchIOGetStatus(batch_handle, 0, &nr, io_batch_events.get(), &poll_timeout);
    if (errBatch.err != 0) {
        NIXL_ERROR << "Error in IO Batch Get Status";
        poll_broken = true;
        current_status = NIXL_ERR_BACKEND;
        return current_status;
    }

    if (nr > entries_remaining) {
        current_status = NIXL_ERR_UNKNOWN;
        return current_status;
    }

    // Every entry that reported counts as done, failed or not, so a failed
    // batch can still be seen to finish
    nixl_status_t failure = NIXL_SUCCESS;
    for (unsigned int i = 0; i < nr; ++i) {
        const CUfileIOEvents_t &event = io_batch_events[i];
        if (event.status == CUFILE_WAITING || event.status == CUFILE_PENDING) {
            continue;
        }
        entries_completed++;
        if (failure != NIXL_SUCCESS) {
            continue;
        }

        if (event.status != CUFILE_COMPLETE || event.cookie == nullptr) {
            NIXL_ERROR << "GDS batch entry failed with status " << event.status;
            failure = NIXL_ERR_BACKEND;
            continue;
        }

        const auto *params = static_cast<const CUfileIOParams_t *>(event.cookie);
        if (event.ret != params->u.batch.size) {
            NIXL_ERROR << "GDS batch entry completed " << event.ret << " of "
                       << params->u.batch.size << " bytes";
            failure = NIXL_ERR_BACKEND;
        }
    }

    if (entries_completed == batch_size) {
        active = false;
    }
    if (failure != NIXL_SUCCESS) {
        current_status = failure;
    } else {
        current_status = active ? NIXL_IN_PROG : NIXL_SUCCESS;
    }

    return current_status;
}

// Count the entries that have reported without acting on the results. Once
// every submitted entry has reported, cuFile is done with the batch
bool
nixlGdsIOBatch::drain() {
    if (!active) {
        return true;
    }
    if (poll_broken) {
        return false;
    }

    unsigned int nr = batch_size - entries_completed;
    struct timespec poll_timeout = {0, 0};
    const CUfileError_t errBatch =
        cuFileBatchIOGetStatus(batch_handle, 0, &nr, io_batch_events.get(), &poll_timeout);
    if (errBatch.err != 0) {
        NIXL_ERROR << "Error in IO Batch Get Status";
        poll_broken = true;
        return false;
    }

    for (unsigned int i = 0; i < nr; ++i) {
        const CUfileIOEvents_t &event = io_batch_events[i];
        if (event.status != CUFILE_WAITING && event.status != CUFILE_PENDING) {
            entries_completed++;
        }
    }
    if (entries_completed >= batch_size) {
        active = false;
    }
    return !active;
}

// Replace the cuFile context of a batch whose poll failed. Destroying the
// context is what makes the batch reusable: nothing can still complete into it
bool
nixlGdsIOBatch::recycle() {
    cuFileBatchIODestroy(batch_handle);
    batch_handle = nullptr;
    active = false;
    poll_broken = false;

    const CUfileError_t err = cuFileBatchIOSetUp(&batch_handle, max_reqs);
    if (err.err != 0) {
        NIXL_ERROR << "Error in setting up Batch";
        init_err = err;
        return false;
    }
    reset();
    return true;
}

// Test hook: destroy the context under a submitted batch so the next poll fails
void
nixlGdsIOBatch::breakContextForTest() {
    cuFileBatchIODestroy(batch_handle);
}

void
nixlGdsIOBatch::reset() {
    if (active) {
        NIXL_ERROR << "GDS: attempted to reset an active batch";
        return;
    }
    entries_completed = 0;
    batch_size = 0;
    poll_broken = false;
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
        break_first_batch_poll_ =
            nixl::getBackendParamDefaulted(custom_params, "test_break_first_batch_poll", 0u) != 0;

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

    if (break_first_batch_poll_) {
        break_first_batch_poll_ = false;
        batch->breakContextForTest();
    }

    batch_out = batch;
    return NIXL_SUCCESS;
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
            // The batches already submitted stay in the list and are polled
            // until cuFile is done with them, then the request reports the failure
            gds_handle->overall_status = status;
            gds_handle->batch_io_list.resize(batch_index);
            return batch_index == 0 ? status : NIXL_IN_PROG;
        }
        current_req += batch_size;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsBatchEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);
    auto &batches = gds_handle->batch_io_list;

    // Batches still in flight are packed to the front, so the list shrinks
    // with one resize per call
    size_t in_flight = 0;
    for (nixlGdsIOBatch *batch : batches) {
        bool done;
        if (gds_handle->overall_status != NIXL_SUCCESS) {
            // The request has already failed, wait for cuFile to finish with the batch
            done = batch->drain();
        } else {
            const nixl_status_t status = batch->checkStatus();
            if (status < 0) {
                gds_handle->overall_status = status;
                done = batch->drain();
            } else {
                done = (status == NIXL_SUCCESS);
            }
        }

        if (done) {
            returnBatchToPool(batch);
        } else if (batch->pollBroken()) {
            // The poll itself failed, so waiting for completions cannot end.
            // Give the batch a fresh context and return it, or give it up
            if (batch->recycle()) {
                returnBatchToPool(batch);
            } else {
                NIXL_ERROR << "GDS batch context could not be recreated, the pool loses one";
            }
            if (gds_handle->overall_status == NIXL_SUCCESS) {
                gds_handle->overall_status = NIXL_ERR_BACKEND;
            }
        } else {
            batches[in_flight++] = batch;
        }
    }
    batches.resize(in_flight);

    if (!batches.empty()) {
        return NIXL_IN_PROG;
    }
    return gds_handle->overall_status;
}

nixl_status_t
nixlGdsBatchEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);
    // cuFile keeps the batches until every entry has reported, so a request
    // with batches in flight cannot be released yet. The caller keeps polling
    if (!gds_handle->batch_io_list.empty()) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    delete gds_handle;
    return NIXL_SUCCESS;
}
