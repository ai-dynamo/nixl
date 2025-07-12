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
#include "absl/log/check.h"

template<nixl_mem_t MemType> struct MemoryHandler {
    static void *
    allocate(size_t len) {
        CHECK(false) << "Unsupported memory type!";
        return nullptr;
    }

    static void
    deallocate(void *addr) {
        CHECK(false) << "Unsupported memory type!";
    }

    static void
    set(void *addr, char byte, size_t size) {
        CHECK(false) << "Unsupported memory type!";
    }
    
    static void
    populateBlobDesc(nixlBlobDesc *desc, void *addr, size_t len, int devId) {
        CHECK(false) << "Unsupported memory type!";
    }

    static void
    populateMetaDesc(nixlMetaDesc *desc, void *addr, size_t len, int devId, nixlBackendMD *&md) {
        CHECK(false) << "Unsupported memory type!";
    }
};

template<> struct MemoryHandler<DRAM_SEG> {
    static void *
    allocate(size_t len) {
        return new char[len];
    }

    static void
    deallocate(void *addr) {
        delete[] static_cast<char *>(addr);
    }

    static void
    set(void *addr, char byte, size_t size) {
        for (size_t i = 0; i < size; i++) {
            memset(static_cast<char *>(addr) + i, byte + i, 1);
        }
    }
    
    static void
    populateBlobDesc(nixlBlobDesc *desc, void *addr, size_t len, int devId) {
        desc->addr = reinterpret_cast<uintptr_t>(addr);
        desc->len = len;
        desc->devId = devId;
    }

    static void
    populateMetaDesc(nixlMetaDesc *desc, void *addr, size_t len, int devId, nixlBackendMD *&md) {
        desc->addr = reinterpret_cast<uintptr_t>(addr);
        desc->len = len;
        desc->devId = devId;
        desc->metadataP = md;
    }
};

template<> struct MemoryHandler<OBJ_SEG> {
    static void *
    allocate(size_t len) {
        return nullptr;
    }

    static void
    deallocate(void *addr) {}

    static void
    set(void *addr, char byte, size_t size) {}
    
    static void
    populateBlobDesc(nixlBlobDesc *desc, void *addr, size_t len, int devId) {
        desc->addr = 0;
        desc->len = len;
        desc->devId = devId;
        desc->metaInfo = "test-obj-key";
    }

    static void
    populateMetaDesc(nixlMetaDesc *desc, void *addr, size_t len, int devId, nixlBackendMD *&md) {
        desc->addr = 0;
        desc->len = len;
        desc->devId = devId;
        desc->metadataP = md;
    }
};

#endif // __MEMORY_HANDLER_H
