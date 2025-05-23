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
#ifndef __GDS_MT_UTILS_H
#define __GDS_MT_UTILS_H

#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <cufile.h>

class gdsMtFileHandle {
    public:
        gdsMtFileHandle(int fd);
        ~gdsMtFileHandle();

        // Disable copy/move
        gdsMtFileHandle(const gdsMtFileHandle&) = delete;
        gdsMtFileHandle& operator=(const gdsMtFileHandle&) = delete;
        gdsMtFileHandle(gdsMtFileHandle&&) = delete;
        gdsMtFileHandle& operator=(gdsMtFileHandle&&) = delete;

        int fd{-1};
        CUfileHandle_t cu_fhandle{nullptr};
};

class gdsMtMemBuf {
    public:
        gdsMtMemBuf(void* ptr, size_t sz, int flags = 0);
        ~gdsMtMemBuf();

        // Disable copy/move
        gdsMtMemBuf(const gdsMtMemBuf&) = delete;
        gdsMtMemBuf& operator=(const gdsMtMemBuf&) = delete;
        gdsMtMemBuf(gdsMtMemBuf&&) = delete;
        gdsMtMemBuf& operator=(gdsMtMemBuf&&) = delete;

    private:
        void *base_{nullptr};
        bool registered_{false};
};

class gdsMtUtil {
    public:
        gdsMtUtil();
        ~gdsMtUtil();
};
#endif
