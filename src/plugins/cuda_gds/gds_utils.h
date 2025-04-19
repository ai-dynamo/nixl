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
#ifndef __GDS_UTILS_H
#define __GDS_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <cufile.h>

class gdsFileHandle {
    public:
        int fd;
        // -1 means inf size file?
        size_t size;
        std::string metadata;
        CUfileHandle_t cu_fhandle;
};

class gdsMemBuf {
    public:
        void *base;
        size_t size;
};

class nixlGdsIOBatch {
    public:
        nixlGdsIOBatch(unsigned int size);
        ~nixlGdsIOBatch();

        nixl_status_t addToBatch(CUfileHandle_t fh, void *buffer,
                                size_t size, size_t file_offset,
                                size_t ptr_offset, CUfileOpcode_t type);
        nixl_status_t submitBatch(int flags);
        nixl_status_t checkStatus();
        nixl_status_t cancelBatch();
        void destroyBatch();
        void reset();

    private:
        CUfileBatchHandle_t batch_handle;
        CUfileIOEvents_t *io_batch_events = nullptr;
        CUfileIOParams_t *io_batch_params = nullptr;
        CUfileError_t init_err = {CU_FILE_SUCCESS};
        unsigned int max_reqs = 0;
        unsigned int batch_size = 0;
        unsigned int entries_completed = 0;
        nixl_status_t current_status = NIXL_ERR_NOT_POSTED;
};

class gdsUtil {
    public:
        gdsUtil() {}
        ~gdsUtil() {}
        nixl_status_t registerFileHandle(int fd, size_t size,
                                       std::string metaInfo,
                                       gdsFileHandle& handle);
        nixl_status_t registerBufHandle(void *ptr, size_t size, int flags);
        void deregisterFileHandle(gdsFileHandle& handle);
        nixl_status_t deregisterBufHandle(void *ptr);
        nixl_status_t openGdsDriver();
        void closeGdsDriver();
};
#endif
