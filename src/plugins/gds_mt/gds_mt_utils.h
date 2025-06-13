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

// Forward declaration
class gdsMtUtil;

class gdsMtFileHandle {
    public:
        gdsMtFileHandle(gdsMtUtil& util, int fd, size_t sz, const std::string& metaInfo);
        ~gdsMtFileHandle();

        // Disable copy/move
        gdsMtFileHandle(const gdsMtFileHandle&) = delete;
        gdsMtFileHandle& operator=(const gdsMtFileHandle&) = delete;

        bool isRegistered() const { return registered_; }

        int fd = -1;
        size_t size = 0;
        std::string metadata;
        CUfileHandle_t cu_fhandle = nullptr;

    private:
        bool registered_{false};
};

class gdsMtMemBuf {
    public:
        gdsMtMemBuf(gdsMtUtil& util, void* ptr, size_t sz, int flags = 0);
        ~gdsMtMemBuf();

        // Disable copy/move
        gdsMtMemBuf(const gdsMtMemBuf&) = delete;
        gdsMtMemBuf& operator=(const gdsMtMemBuf&) = delete;

        bool isRegistered() const { return registered_; }

        void *base = nullptr;
        size_t size = 0;

    private:
        bool registered_{false};
};

class gdsMtUtil {
    public:
        gdsMtUtil();
        ~gdsMtUtil();

        bool isInitialized() const { return driver_initialized_; }

    private:
        bool driver_initialized_{false};
};
#endif
