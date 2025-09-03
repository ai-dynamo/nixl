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
#include "gds_utils.h"
#include "common/nixl_log.h"

nixl_status_t gdsUtil::registerFileHandle(int fd,
                                          size_t size,
                                          std::string metaInfo,
                                          gdsFileHandle& gds_handle)
{
    CUfileError_t status;
    CUfileDescr_t descr = {};
    CUfileHandle_t handle;

    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "file register error:";
        return NIXL_ERR_BACKEND;
    }

    gds_handle.cu_fhandle = handle;
    gds_handle.fd = fd;
    gds_handle.size = size;
    gds_handle.metadata = metaInfo;

    return NIXL_SUCCESS;
}

nixl_status_t gdsUtil::registerBufHandle(void *ptr,
                                         size_t size,
                                         int flags)
{
    CUfileError_t status;

    status = cuFileBufRegister(ptr, size, flags);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_WARN << "Buffer registration failed - will use compat mode";
    }
    return NIXL_SUCCESS;
}

nixl_status_t gdsUtil::openGdsDriver()
{
    CUfileError_t err;

    err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "Error initializing GPU Direct Storage driver";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void gdsUtil::closeGdsDriver()
{
    cuFileDriverClose();
}

void gdsUtil::deregisterFileHandle(gdsFileHandle& handle)
{
    cuFileHandleDeregister(handle.cu_fhandle);
}

nixl_status_t gdsUtil::deregisterBufHandle(void *ptr)
{
    CUfileError_t status;

    status = cuFileBufDeregister(ptr);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "Error De-Registering Buffer";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
    : max_reqs(size)
{
    CUfileError_t err;

    io_batch_events = new CUfileIOEvents_t[size];
    io_batch_params = new CUfileIOParams_t[size];

    err = cuFileBatchIOSetUp(&batch_handle, size);
    if (err.err != 0) {
        NIXL_ERROR << "Error in setting up Batch";
        init_err = err;
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch()
{
    if (current_status == NIXL_SUCCESS ||
        current_status == NIXL_ERR_NOT_POSTED) {
            delete[] io_batch_events;
            delete[] io_batch_params;
            cuFileBatchIODestroy(batch_handle);
    } else {
            NIXL_ERROR << "Attempting to delete a batch before completion";
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

nixl_status_t nixlGdsIOBatch::cancelBatch()
{
    CUfileError_t   err;

    err = cuFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        NIXL_ERROR << "Error in canceling batch";
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
        NIXL_ERROR << "Error in setting up Batch";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::checkStatus()
{
    CUfileError_t       errBatch;
    unsigned int        nr = batch_size;

    errBatch = cuFileBatchIOGetStatus(batch_handle, nr, &nr,
                                      io_batch_events, NULL);
    if (errBatch.err != 0) {
        NIXL_ERROR << "Error in IO Batch Get Status";
        current_status = NIXL_ERR_BACKEND;
    }

    entries_completed += nr;
    if (entries_completed < (unsigned int)batch_size)
        current_status = NIXL_IN_PROG;
    else if (entries_completed > batch_size)
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
