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
