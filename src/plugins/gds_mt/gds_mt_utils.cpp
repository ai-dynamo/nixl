/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Serapheim Dimitropoulos, WekaIO Ltd.
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
#include <iostream>
#include "common/nixl_log.h"
#include "gds_mt_utils.h"

gdsMtUtil::gdsMtUtil() {
    CUfileError_t err = cuFileDriverOpen();
    if (err.err == CU_FILE_SUCCESS) {
        driver_initialized_ = true;
    } else {
        NIXL_ERROR << "GDS_MT: error initializing GPU Direct Storage driver: error=" << err.err;
        driver_initialized_ = false;
    }
}

gdsMtUtil::~gdsMtUtil() {
    if (driver_initialized_) {
        cuFileDriverClose();
    }
}

gdsMtMemBuf::gdsMtMemBuf(gdsMtUtil& util, void* ptr, size_t sz, int flags)
    : base(ptr), size(sz), util_(util) {
    nixl_status_t status = util_.registerBufHandle(ptr, sz, flags);
    if (status == NIXL_SUCCESS) {
        registered_ = true;
    }
}

gdsMtMemBuf::~gdsMtMemBuf() {
    if (registered_) {
        util_.deregisterBufHandle(base);
    }
}

gdsMtFileHandle::gdsMtFileHandle(gdsMtUtil& util, int file_fd, size_t sz, const std::string& metaInfo)
    : fd(file_fd), size(sz), metadata(metaInfo) {
    
    if (!util.isInitialized()) {
        NIXL_ERROR << "GDS_MT: gdsMtUtil not initialized";
        return;
    }

    CUfileError_t status;
    CUfileDescr_t descr;
    CUfileHandle_t handle;

    descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: file register error: error=" << status.err << ", fd=" << fd;
        return;
    }

    cu_fhandle = handle;
    registered_ = true;
}

gdsMtFileHandle::~gdsMtFileHandle() {
    if (registered_) {
        cuFileHandleDeregister(cu_fhandle);
    }
}

nixl_status_t gdsMtUtil::registerBufHandle(void *ptr,
                                           size_t size,
                                           int flags)
{
    CUfileError_t status;

    status = cuFileBufRegister(ptr, size, flags);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: warning: buffer registration failed - will use compat mode: error=" << status.err;
    }
    return NIXL_SUCCESS;
}

nixl_status_t gdsMtUtil::deregisterBufHandle(void *ptr)
{
    CUfileError_t status;

    status = cuFileBufDeregister(ptr);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: error deregistering buffer: error=" << status.err << " ptr=" << ptr;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}
