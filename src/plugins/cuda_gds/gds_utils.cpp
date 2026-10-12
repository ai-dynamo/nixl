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
#include <stdexcept>
#include <string>

#include "common/nixl_log.h"
#include "gds_utils.h"

gdsFileHandle::gdsFileHandle(nixl::FileFd &&fd) : file_fd(std::move(fd)) {
    CUfileDescr_t descr = {};
    descr.handle.fd = file_fd.fd();
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    const CUfileError_t status = cuFileHandleRegister(&cu_fhandle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        // ~FileFd as the exception unwinds closes the owned fd if any.
        throw std::runtime_error("GDS: file register error: error=" + std::to_string(status.err) +
                                 ", fd=" + std::to_string(file_fd.fd()));
    }
}

gdsFileHandle::~gdsFileHandle() {
    cuFileHandleDeregister(cu_fhandle);
    // ~FileFd closes the fd if path-mode owned it.
}

gdsMemBuf::gdsMemBuf(void *ptr, size_t sz, int flags) : base_(ptr) {
    const CUfileError_t status = cuFileBufRegister(ptr, sz, flags);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_WARN << "GDS: warning: buffer registration failed - will use compat mode: error="
                  << status.err;
        // Not fatal: leave registered_ false so we do not deregister later.
    } else {
        registered_ = true;
    }
}

gdsMemBuf::~gdsMemBuf() {
    if (registered_) {
        const CUfileError_t status = cuFileBufDeregister(base_);
        if (status.err != CU_FILE_SUCCESS) {
            NIXL_WARN << "GDS: warning: deregistering buffer: error=" << status.err
                      << " ptr=" << base_;
        }
    }
}

gdsDriverHandle::gdsDriverHandle() {
    const CUfileError_t status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        throw std::runtime_error("GDS: error initializing GPU Direct Storage driver: error=" +
                                 std::to_string(status.err));
    }
}

gdsDriverHandle::~gdsDriverHandle() {
    cuFileDriverClose();
}
