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

#define  GDS_BATCH_LIMIT 128

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
{
    max_reqs            = size;
    io_batch_events     = new CUfileIOEvents_t[size];
    io_batch_params     = new CUfileIOParams_t[size];
    current_status      = NIXL_ERR_NOT_POSTED;
    entries_completed   = 0;
    batch_size          = 0;

    init_err = cuFileBatchIOSetUp(&batch_handle, size);
    if (init_err.err != 0) {
        std::cerr << "Error in creating the batch\n";
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

    // Set default pool size
    pool_size = 8;  // Default value if not specified in params

    // Read custom parameters if available
    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params && custom_params->count("pool_size") > 0) {
        try {
            pool_size = std::stoi((*custom_params)["pool_size"]);
            // Ensure reasonable limits
            if (pool_size < 1) pool_size = 1;
            if (pool_size > 32) pool_size = 32;
        } catch (...) {
            // Keep default if conversion fails
        }
    }

    this->initErr = false;
    if (gds_utils->openGdsDriver() == NIXL_ERR_BACKEND)
        this->initErr = true;

    // Pre-populate the batch pool with full pool size
    for (unsigned int i = 0; i < pool_size; i++) {
        batch_pool.push_back(new nixlGdsIOBatch(GDS_BATCH_LIMIT));
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
           status               = NIXL_SUCCESS;
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
    } else {
        // Unsupported in the backend.
        return NIXL_ERR_BACKEND;
    }
    out = (nixlBackendMD*) md;
    // set value for gds handle here.
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
    // TODO: Determine the batches and prepare most of the handle
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
    // Only keep up to pool_size batches
    if (batch_pool.size() < pool_size) {
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
    void                *addr = NULL;
    size_t              size = 0;
    size_t              offset = 0;
    int                 rc = 0;
    size_t              buf_cnt = local.descCount();
    size_t              file_cnt = remote.descCount();
    nixl_status_t       ret = NIXL_ERR_NOT_POSTED;
    int                 full_batches = 1;
    int                 total_batches = 1;
    int                 remainder = 0;
    int                 curr_buf_cnt = 0;
    gdsFileHandle       fh;
    nixlGdsBackendReqH  *gds_handle;

    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        std::cerr << "Error in count or operation selection\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        std::cerr << "Only support I/O between VRAM to file type\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() == DRAM_SEG) || (local.getType() == DRAM_SEG)) {
        std::cerr << "Backend does not support DRAM to/from files\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    full_batches = buf_cnt / GDS_BATCH_LIMIT;
    remainder = buf_cnt % GDS_BATCH_LIMIT;
    total_batches = full_batches + ((remainder > 0) ? 1 : 0);

    gds_handle = new nixlGdsBackendReqH();
    for (int j = 0; j < total_batches; j++) {
        int req_cnt = (j < full_batches) ? GDS_BATCH_LIMIT : remainder;
        nixlGdsIOBatch *batch_ios = getBatchFromPool(req_cnt);

        for (int i = curr_buf_cnt; i < (curr_buf_cnt + req_cnt); i++) {
            if (local.getType() == VRAM_SEG) {
                addr = (void *)local[i].addr;
                size = local[i].len;
                offset = (size_t)remote[i].addr;

                auto it = gds_file_map.find(remote[i].devId);
                if (it != gds_file_map.end()) {
                    fh = it->second;
                } else {
                    ret = NIXL_ERR_NOT_FOUND;
                    goto err_exit;
                }
            } else if (local.getType() == FILE_SEG) {
                addr = (void *)remote[i].addr;
                size = remote[i].len;
                offset = (size_t)local[i].addr;

                auto it = gds_file_map.find(local[i].devId);
                if (it != gds_file_map.end()) {
                    fh = it->second;
                } else {
                    ret = NIXL_ERR_NOT_FOUND;
                    goto err_exit;
                }
            }
            CUfileOpcode_t op = (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE;
            rc = batch_ios->addToBatch(fh.cu_fhandle, addr, size, offset, 0, op);
            if (rc != 0) {
                returnBatchToPool(batch_ios);
                ret = NIXL_ERR_BACKEND;
                goto err_exit;
            }
        }
        curr_buf_cnt += req_cnt;
        rc = batch_ios->submitBatch(0);
        if (rc != 0) {
            returnBatchToPool(batch_ios);
            ret = NIXL_ERR_BACKEND;
            goto err_exit;
        }
        gds_handle->batch_io_list.push_back(batch_ios);
    }
    handle = gds_handle;
    return ret;

err_exit:
    // Clean up any batches that were already created
    for (auto* batch : gds_handle->batch_io_list) {
        batch->cancelBatch();
        returnBatchToPool(batch);
    }
    delete gds_handle;
    return ret;
}

nixl_status_t nixlGdsEngine::checkXfer(nixlBackendReqH* handle)
{
    nixlGdsBackendReqH *gds_handle = (nixlGdsBackendReqH *)handle;
    nixl_status_t status = NIXL_IN_PROG;

    if (gds_handle->batch_io_list.size() == 0)
        status = NIXL_SUCCESS;

    for (auto it = gds_handle->batch_io_list.begin();
         it != gds_handle->batch_io_list.end();) {
        nixlGdsIOBatch *batch_ios = *it;
        nixl_status_t batch_status = batch_ios->checkStatus();

        if (batch_status == NIXL_IN_PROG) {
            return batch_status;
        } else if (batch_status == NIXL_SUCCESS) {
            returnBatchToPool(batch_ios);
            it = gds_handle->batch_io_list.erase(it);
        } else if (batch_status < 0) {
            // Failure of transfer
            // lets kill every batch
            break;
        } else {
            it++;
        }
    }

    // Cleanup even if one batch fails
    if (status < 0) {
        auto it = gds_handle->batch_io_list.begin();
        while (it != gds_handle->batch_io_list.end()) {
            nixlGdsIOBatch *batch_ios = *it;
            batch_ios->cancelBatch();
            returnBatchToPool(batch_ios);
            it = gds_handle->batch_io_list.erase(it);
        }
    }
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
        batch->destroyBatch();
        delete batch;
    }
    batch_pool.clear();

    cuFileDriverClose();
    delete gds_utils;
}
