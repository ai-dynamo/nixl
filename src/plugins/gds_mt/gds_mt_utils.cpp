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
#include <stdexcept>
#include "common/nixl_log.h"
#include "gds_mt_utils.h"

gdsMtUtil::gdsMtUtil() {
    const CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error ("GDS_MT: error initializing GPU Direct Storage driver: error=" +
                                  std::to_string (status.err));
    }
}

gdsMtUtil::~gdsMtUtil() {
    cuFileDriverClose();
}

gdsMtMemBuf::gdsMtMemBuf (void *ptr, size_t sz, int flags) : base_ (ptr) {

    const CUfileError_t status = cuFileBufRegister (ptr, sz, flags);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_WARN << "GDS_MT: warning: buffer registration failed - will use compat mode: error="
                  << status.err;
        // Note: We don't set registered_ = true, but this is not considered a fatal error
    } else {
        registered_ = true;
    }
}

gdsMtMemBuf::~gdsMtMemBuf() {
    if (registered_) {
        const CUfileError_t status = cuFileBufDeregister (base_);
        if (status.err != CU_FILE_SUCCESS) {
            NIXL_WARN << "GDS_MT: warning: deregistering buffer: error=" << status.err
                      << " ptr=" << base_;
        }
    }
}

gdsMtFileHandle::gdsMtFileHandle (int file_fd) : fd (file_fd) {

    CUfileDescr_t descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    const CUfileError_t status = cuFileHandleRegister (&cu_fhandle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error ("GDS_MT: file register error: error=" +
                                  std::to_string (status.err) + ", fd=" + std::to_string (fd));
    }
}

gdsMtFileHandle::~gdsMtFileHandle() {
    cuFileHandleDeregister (cu_fhandle);
}
