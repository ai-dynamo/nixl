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
#include <cassert>
#include <iostream>
#include <cufile.h>
#include "gds_backend.h"
#include "common/str_tools.h"

/** Setting the default values to check the batch limit */
#define DEFAULT_BATCH_LIMIT 128
/** Setting the max request size to 16 MB */
#define DEFAULT_MAX_REQUEST_SIZE (16 * 1024 * 1024)  // 16MB
/** Create a batch pool of size 8 */
#define DEFAULT_BATCH_POOL_SIZE 8

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
    : max_reqs(size)
{
    CUfileError_t err;

    io_batch_events = new CUfileIOEvents_t[size];
    io_batch_params = new CUfileIOParams_t[size];

    err = cuFileBatchIOSetUp(&batch_handle, size);
    if (err.err != 0) {
        std::cerr << "Error in setting up Batch\n";
        init_err = err;
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch()
{
    if (current_status == NIXL_SUCCESS ||
        current_status == NIXL_ERR_NOT_POSTED) {
            delete io_batch_events;
            delete io_batch_params;
        cuFileBatchIODestroy(batch_handle);
    } else {
            std::cerr<<"Attempting to delete a batch before completion\n";
    }
}

nixl_status_t nixlGdsIOBatch::addToBatch(CUfileHandle_t fh, void *buffer,
                                         size_t size, size_t file_offset,
                                         size_t ptr_offset,
                                         CUfileOpcode_t type)
{
    CUfileIOParams_t    *params = nullptr;

    if (batch_size >= max_reqs)
        return NIXL_ERR_BACKEND;

    params                          = &io_batch_params[batch_size];
    params->mode                    = CUFILE_BATCH;
    params->fh                      = fh;
    params->u.batch.devPtr_base     = buffer;
    params->u.batch.file_offset     = file_offset;
    params->u.batch.devPtr_offset   = ptr_offset;
    params->u.batch.size            = size;
    params->opcode                  = type;
    params->cookie                  = params;
    batch_size++;

    return NIXL_SUCCESS;
}

void nixlGdsIOBatch::destroyBatch()
{
    cuFileBatchIODestroy(batch_handle);
}


nixl_status_t nixlGdsIOBatch::cancelBatch()
{
    CUfileError_t   err;

    err = cuFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        std::cerr << "Error in canceling batch\n";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::submitBatch(int flags)
{
    CUfileError_t   err;

    err = cuFileBatchIOSubmit(batch_handle, batch_size,
                              io_batch_params, flags);
    if (err.err != 0) {
        std::cerr << "Error in setting up Batch\n" << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::checkStatus()
{
    CUfileError_t       errBatch;
    unsigned int        nr = max_reqs;

    errBatch = cuFileBatchIOGetStatus(batch_handle, 0, &nr,
                                      io_batch_events, NULL);
    if (errBatch.err != 0) {
        std::cerr << "Error in IO Batch Get Status" << std::endl;
        current_status = NIXL_ERR_BACKEND;
    }

    entries_completed += nr;
    if (entries_completed < (unsigned int)max_reqs)
        current_status = NIXL_IN_PROG;
    else if (entries_completed > max_reqs)
        current_status = NIXL_ERR_UNKNOWN;
    else
        current_status = NIXL_SUCCESS;

    return current_status;
}

void nixlGdsIOBatch::reset() {
    entries_completed = 0;
    batch_size = 0;
    current_status = NIXL_ERR_NOT_POSTED;
}

nixlGdsEngine::nixlGdsEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
{
    gds_utils = new gdsUtil();

    // Set default values
    batch_pool_size = DEFAULT_BATCH_POOL_SIZE;
    batch_limit = DEFAULT_BATCH_LIMIT;
    max_request_size = DEFAULT_MAX_REQUEST_SIZE;

    // Read custom parameters if available
    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params) {
        // Configure batch_pool_size
        if (custom_params->count("batch_pool_size") > 0) {
            try {
                batch_pool_size = std::stoi((*custom_params)["batch_pool_size"]);
            } catch (...) {
                // Keep default if conversion fails
            }
        }

        // Configure batch_limit
        if (custom_params->count("batch_limit") > 0) {
            try {
                batch_limit = std::stoi((*custom_params)["batch_limit"]);
            } catch (...) {
                // Keep default if conversion fails
            }
        }

        // Configure max_request_size
        if (custom_params->count("max_request_size") > 0) {
            try {
                max_request_size = std::stoul((*custom_params)["max_request_size"]);
            } catch (...) {
                // Keep default if conversion fails
            }
        }
    }

    this->initErr = false;
    if (gds_utils->openGdsDriver() == NIXL_ERR_BACKEND) {
        this->initErr = true;
        return;
    }

    for (unsigned int i = 0; i < batch_pool_size; i++) {
         batch_pool.push_back(new nixlGdsIOBatch(batch_limit));
    }

}


nixl_status_t nixlGdsEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    nixl_status_t status;
    nixlGdsMetadata *md  = new nixlGdsMetadata();

    if (nixl_mem == FILE_SEG) {
        // if the same file is reused - no need to re-register
        auto it = gds_file_map.find(mem.devId);
        if (it != gds_file_map.end()) {
            md->handle.cu_fhandle   = it->second.cu_fhandle;
            md->handle.fd           = mem.devId;
            md->handle.size         = mem.len;
            md->handle.metadata     = mem.metaInfo;
            md->type                = nixl_mem;
            status                  = NIXL_SUCCESS;
        } else {
            status = gds_utils->registerFileHandle(mem.devId, mem.len,
                                             mem.metaInfo, md->handle);
            if (NIXL_SUCCESS != status) {
                delete md;
                return status;
            }
            md->type                = nixl_mem;
            gds_file_map[mem.devId] = md->handle;
        }
    } else if (nixl_mem == VRAM_SEG) {
        status = gds_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
        if (NIXL_SUCCESS != status) {
            delete md;
            return status;
        }
        md->buf.base   = (void *)mem.addr;
        md->buf.size   = mem.len;
        md->type       = nixl_mem;
    } else if (nixl_mem == DRAM_SEG) {
        // For DRAM, we need to register it as a buffer with GDS
        status = gds_utils->registerBufHandle((void *)mem.addr, mem.len, 0);
        if (NIXL_SUCCESS != status) {
            delete md;
            return status;
        }
        md->buf.base   = (void *)mem.addr;
        md->buf.size   = mem.len;
        md->type       = nixl_mem;
    } else {
        // Unsupported in the backend.
        delete md;
        return NIXL_ERR_BACKEND;
    }
    out = (nixlBackendMD*) md;
    return status;
}

nixl_status_t nixlGdsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlGdsMetadata *md = (nixlGdsMetadata *)meta;
    if (md->type == FILE_SEG) {
        gds_utils->deregisterFileHandle(md->handle);
    } else {
        gds_utils->deregisterBufHandle(md->buf.base);
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    nixlGdsBackendReqH* gds_handle = new nixlGdsBackendReqH();
    nixl_status_t ret = createBatches(operation, local, remote, gds_handle);

    if (ret != NIXL_SUCCESS) {
        delete gds_handle;
        return ret;
    }

    handle = gds_handle;
    return NIXL_SUCCESS;
}

nixlGdsIOBatch* nixlGdsEngine::getBatchFromPool(unsigned int size) {
    // First try to find a batch of the right size in the pool
    for (auto it = batch_pool.begin(); it != batch_pool.end(); ++it) {
        if ((*it)->getMaxReqs() == size) {
            nixlGdsIOBatch* batch = *it;
            batch_pool.erase(it);
            batch->reset();
            return batch;
        }
    }

    // If no suitable batch found, create a new one
    nixlGdsIOBatch* batch = new nixlGdsIOBatch(size);
    return batch;
}

void nixlGdsEngine::returnBatchToPool(nixlGdsIOBatch* batch) {
    // Only keep up to batch_pool_size batches
    if (batch_pool.size() < batch_pool_size) {
        batch->reset();
        batch_pool.push_back(batch);
    } else {
        delete batch;
    }
}

nixl_status_t nixlGdsEngine::postXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    nixlGdsBackendReqH* gds_handle = (nixlGdsBackendReqH*)handle;
    nixl_status_t ret;

    if (!gds_handle) {
        std::cerr << "Invalid handle\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    // If batches are empty or request list is empty, we need to create them
    if (gds_handle->batch_io_list.empty() || gds_handle->request_list.empty()) {
        ret = createBatches(operation, local, remote, gds_handle);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
    } else {
        // Reuse existing batches by repopulating from request list
        size_t current_req = 0;
        for (auto* batch : gds_handle->batch_io_list) {
            batch->reset();  // Just reset counters

            // Repopulate this batch
            size_t batch_size = std::min(gds_handle->request_list.size() - current_req,
                                       (size_t)batch->getMaxReqs());

            for (size_t i = 0; i < batch_size; i++) {
                const auto& req = gds_handle->request_list[current_req + i];
                ret = batch->addToBatch(req.fh, req.addr, req.size,
                                      req.file_offset, 0, req.op);
                if (ret != NIXL_SUCCESS) {
                    // Clean up on error
                    for (auto* cleanup_batch : gds_handle->batch_io_list) {
                        cleanup_batch->cancelBatch();
                        returnBatchToPool(cleanup_batch);
                    }
                    gds_handle->batch_io_list.clear();
                    return ret;
                }
            }
            current_req += batch_size;
        }
    }

    // Submit all prepared batches
    for (auto* batch : gds_handle->batch_io_list) {
        ret = batch->submitBatch(0);
        if (ret != NIXL_SUCCESS) {
            // Clean up on error
            for (auto* cleanup_batch : gds_handle->batch_io_list) {
                cleanup_batch->cancelBatch();
                returnBatchToPool(cleanup_batch);
            }
            gds_handle->batch_io_list.clear();
            return ret;
        }
    }
    return NIXL_IN_PROG;
}

nixl_status_t nixlGdsEngine::checkXfer(nixlBackendReqH* handle)
{
    nixlGdsBackendReqH  *gds_handle = (nixlGdsBackendReqH *) handle;
    nixl_status_t       status = NIXL_IN_PROG;

    if (gds_handle->batch_io_list.empty()) {
        return NIXL_SUCCESS;
    }

    // First check all batches
    for (auto* batch : gds_handle->batch_io_list) {
        status = batch->checkStatus();
        if (status == NIXL_IN_PROG) {
            return status;  // Exit early if any batch is still in progress
        }
        if (status < 0) {
            break;  // Exit loop on first error
        }
    }

    // Now cleanup all batches - either all succeeded or we hit an error
    for (auto* batch : gds_handle->batch_io_list) {
        if (status < 0) {
            batch->cancelBatch();
            batch->destroyBatch();
        }
        delete batch;
    }
    gds_handle->batch_io_list.clear();

    return status;
}

nixl_status_t nixlGdsEngine::releaseReqH(nixlBackendReqH* handle)
{

    nixlGdsBackendReqH *gds_handle = (nixlGdsBackendReqH *) handle;

    delete gds_handle;
    return NIXL_SUCCESS;
}

nixlGdsEngine::~nixlGdsEngine() {
    // Clean up the batch pool
    for (auto* batch : batch_pool) {
        delete batch;  // This will automatically call the nixlGdsIOBatch destructor
    }
    batch_pool.clear();

    if (gds_utils) {
        delete gds_utils;
        gds_utils = nullptr;
    }

    cuFileDriverClose();
}

// Add helper function to create batches that can be called from both prepXfer and postXfer
nixl_status_t nixlGdsEngine::createBatches(const nixl_xfer_op_t &operation,
                                           const nixl_meta_dlist_t &local,
                                           const nixl_meta_dlist_t &remote,
                                           nixlGdsBackendReqH* gds_handle) {
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    // Basic validation
    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        std::cerr << "Error in count or operation selection\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        std::cerr << "Only support I/O between memory (DRAM/VRAM) and file type\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Determine if local is the file segment
    bool is_local_file = (local.getType() == FILE_SEG);

    // Clear existing request list and batches if any
    gds_handle->request_list.clear();
    for (auto* batch : gds_handle->batch_io_list) {
        returnBatchToPool(batch);
    }
    gds_handle->batch_io_list.clear();

    // Create list of all transfer requests
    for (size_t i = 0; i < buf_cnt; i++) {
        void* base_addr;
        size_t total_size;
        size_t base_offset;
        gdsFileHandle fh;

        // Get transfer parameters based on whether local is file or memory
        if (is_local_file) {
            base_addr = (void*)remote[i].addr;
            total_size = remote[i].len;
            base_offset = (size_t)local[i].addr;

            auto it = gds_file_map.find(local[i].devId);
            if (it == gds_file_map.end()) {
                std::cerr << "File handle not found\n";
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        } else {
            base_addr = (void*)local[i].addr;
            total_size = local[i].len;
            base_offset = (size_t)remote[i].addr;

            auto it = gds_file_map.find(remote[i].devId);
            if (it == gds_file_map.end()) {
                std::cerr << "File handle not found\n";
                return NIXL_ERR_NOT_FOUND;
            }
            fh = it->second;
        }

        // Split large transfers into multiple requests
        size_t remaining_size = total_size;
        size_t current_offset = 0;

        while (remaining_size > 0) {
            size_t request_size = std::min(remaining_size,
                                       (size_t)max_request_size);

            GdsTransferRequestH req;
            req.addr = (char*)base_addr + current_offset;
            req.size = request_size;
            req.file_offset = base_offset + current_offset;
            req.fh = fh.cu_fhandle;
            req.op = (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE;

            gds_handle->request_list.push_back(req);

            remaining_size -= request_size;
            current_offset += request_size;
        }
    }

    // Create and prepare batches
    size_t current_req = 0;
    const auto& request_list = gds_handle->request_list;

    while (current_req < request_list.size()) {
        size_t batch_size = std::min(request_list.size() - current_req,
				     (size_t)batch_limit);
        nixlGdsIOBatch* batch_ios = getBatchFromPool(batch_size);

        // Add requests to batch
        for (size_t i = 0; i < batch_size; i++) {
            const auto& req = request_list[current_req + i];
            int rc = batch_ios->addToBatch(req.fh, req.addr, req.size,
                                           req.file_offset, 0, req.op);
            if (rc != 0) {
                returnBatchToPool(batch_ios);
                // Clean up previously created batches
                for (auto* batch : gds_handle->batch_io_list) {
                    returnBatchToPool(batch);
                }
                gds_handle->batch_io_list.clear();
                return NIXL_ERR_BACKEND;
            }
        }

        gds_handle->batch_io_list.push_back(batch_ios);
        current_req += batch_size;
    }

    return NIXL_SUCCESS;
}
