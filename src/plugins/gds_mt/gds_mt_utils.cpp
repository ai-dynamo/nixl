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

nixl_status_t gdsMtUtil::registerFileHandle(int fd,
                                            size_t size,
                                            std::string metaInfo,
                                            gdsMtFileHandle& gdsMtHandle)
{
    CUfileError_t status;
    CUfileDescr_t descr;
    CUfileHandle_t handle;

    descr = {};
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: file register error: error=" << status.err << ", fd=" << fd;
        return NIXL_ERR_BACKEND;
    }

    gdsMtHandle.cu_fhandle = handle;
    gdsMtHandle.fd = fd;
    gdsMtHandle.size = size;
    gdsMtHandle.metadata = metaInfo;

    return NIXL_SUCCESS;
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

nixl_status_t gdsMtUtil::openGdsMtDriver()
{
    CUfileError_t err;

    err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        NIXL_ERROR << "GDS_MT: error initializing GPU Direct Storage driver: error=" << err.err;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void gdsMtUtil::closeGdsMtDriver()
{
    cuFileDriverClose();
}

void gdsMtUtil::deregisterFileHandle(gdsMtFileHandle& handle)
{
    cuFileHandleDeregister(handle.cu_fhandle);
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
