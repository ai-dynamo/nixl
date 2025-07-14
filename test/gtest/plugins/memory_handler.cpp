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

#include "absl/log/check.h"
#include "common/nixl_log.h"
#include "backend/backend_aux.h"
#include "memory_handler.h"

void
MemoryHandler::allocate(size_t len) {
    switch (memType_) {
    case DRAM_SEG:
        addr_ = new char[len];
        break;
    case OBJ_SEG:
        addr_ = nullptr;
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
    len_ = len;
}

void
MemoryHandler::deallocate() {
    switch (memType_) {
    case DRAM_SEG:
        delete[] static_cast<char *>(addr_);
        break;
    case OBJ_SEG:
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
}

void
MemoryHandler::set(char byte) {
    switch (memType_) {
    case DRAM_SEG:
        for (size_t i = 0; i < len_; i++)
            ((char *)addr_)[i] = byte + i;
        break;
    case OBJ_SEG:
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
}

bool
MemoryHandler::check(char byte) {
    switch (memType_) {
    case DRAM_SEG:
        for (size_t i = 0; i < len_; i++) {
            uint8_t expected_byte = (uint8_t)byte + i;
            if (((char *)addr_)[i] != expected_byte) {
                NIXL_ERROR << "Verification failed at index " << i
                           << "! local: " << ((char *)addr_)[i] << ", expected: " << expected_byte;
                return false;
            }
        }
        break;
    case OBJ_SEG:
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
    return true;
}

void
MemoryHandler::reset() {
    switch (memType_) {
    case DRAM_SEG:
        memset(addr_, 0x00, len_);
        break;
    case OBJ_SEG:
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
}

void
MemoryHandler::populateBlobDesc(nixlBlobDesc *desc, int buf_index) {
    switch (memType_) {
    case DRAM_SEG:
        desc->addr = reinterpret_cast<uintptr_t>(addr_);
        break;
    case OBJ_SEG:
        desc->addr = 0;
        desc->metaInfo = "test-obj-key-" + std::to_string(buf_index);
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
    desc->len = len_;
    desc->devId = devId_;
}

void
MemoryHandler::populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
    switch (memType_) {
    case DRAM_SEG:
        desc->addr = reinterpret_cast<uintptr_t>(addr_) + entry_index * entry_size;
        desc->len = entry_size;
        break;
    case OBJ_SEG:
        desc->addr = 0;
        desc->len = len_;
        break;
    default:
        CHECK(false) << "Unsupported memory type!";
        break;
    }
    desc->devId = devId_;
    desc->metadataP = md_;
}