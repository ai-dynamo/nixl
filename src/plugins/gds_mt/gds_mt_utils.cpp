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
    : base(ptr), size(sz) {

    // Check if driver is initialized before attempting registration
    if (!util.isInitialized()) {
        NIXL_ERROR << "GDS_MT: gdsMtUtil not initialized";
        return;
    }

    CUfileError_t status = cuFileBufRegister(ptr, sz, flags);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: warning: buffer registration failed - will use compat mode: error=" << status.err;
        // Note: We don't set registered_ = true, but this is not considered a fatal error
    } else {
        registered_ = true;
    }
}

gdsMtMemBuf::~gdsMtMemBuf() {
    if (registered_) {
        CUfileError_t status = cuFileBufDeregister(base);
        if (status.err != CU_FILE_SUCCESS) {
            NIXL_ERROR << "GDS_MT: error deregistering buffer: error=" << status.err << " ptr=" << base;
        }
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
