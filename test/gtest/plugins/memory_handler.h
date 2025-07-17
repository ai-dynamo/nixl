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
#ifndef __MEMORY_HANDLER_H
#define __MEMORY_HANDLER_H

#include "backend/backend_aux.h"

class MemoryHandler {
public:
    MemoryHandler(nixl_mem_t memType, int devId) : memType_(memType), devId_(devId) {}

    void
    allocate(size_t len);

    void
    deallocate();

    void
    set(char byte);

    bool
    check(char byte);

    void
    reset();

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0);

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size);

    nixl_mem_t
    getMemType() {
        return memType_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

    int
    getDevId() {
        return devId_;
    }

private:
    nixl_mem_t memType_;
    void *addr_;
    size_t len_;
    int devId_;
    nixlBackendMD *md_;
};

#endif // __MEMORY_HANDLER_H