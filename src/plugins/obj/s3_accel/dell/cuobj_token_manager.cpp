/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuobj_token_manager.h"
#include "common/nixl_log.h"
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

CuObjTokenManager::CuObjTokenManager(cuObjProto_t proto) {
    // Pattern B: empty ops struct — we never invoke cuObjPut/cuObjGet.
    // The constructor requires a CUObjIOOps reference, but the callbacks
    // are never called because we only use cuMemObjGetRDMAToken.
    CUObjOps_t ops{};
    std::memset(&ops, 0, sizeof(ops));
    client_ = std::make_unique<cuObjClient>(ops, proto);
}

CuObjTokenManager::~CuObjTokenManager() {
    client_.reset();
}

bool
CuObjTokenManager::isConnected() const {
    return client_ && client_->isConnected();
}

// ---------------------------------------------------------------------------
// Memory registration (1:1 mapping to cuMemObjGetDescriptor)
// ---------------------------------------------------------------------------

cuObjErr_t
CuObjTokenManager::registerMemory(void *ptr, size_t size) {
    if (!ptr || size == 0) {
        NIXL_ERROR << "registerMemory: invalid arguments (ptr=" << ptr << ", size=" << size << ")";
        return CU_OBJ_FAIL;
    }

    if (size >= CUOBJ_MAX_MEMORY_REG_SIZE) {
        NIXL_ERROR << "registerMemory: size " << size << " exceeds cuObject limit ("
                   << CUOBJ_MAX_MEMORY_REG_SIZE << ")";
        return CU_OBJ_FAIL;
    }

    cuObjErr_t rc = client_->cuMemObjGetDescriptor(ptr, size);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed: ptr=0x" << std::hex
                   << reinterpret_cast<uintptr_t>(ptr) << " size=" << std::dec << size;
        return rc;
    }

    NIXL_DEBUG << "registerMemory: ptr=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr)
               << " size=" << std::dec << size;
    return CU_OBJ_SUCCESS;
}

cuObjErr_t
CuObjTokenManager::deregisterMemory(void *ptr) {
    if (!ptr) {
        return CU_OBJ_SUCCESS; // Nothing to do.
    }

    cuObjErr_t rc = client_->cuMemObjPutDescriptor(ptr);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjPutDescriptor failed: ptr=0x" << std::hex
                   << reinterpret_cast<uintptr_t>(ptr);
        return rc;
    }

    NIXL_DEBUG << "deregisterMemory: ptr=0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
    return CU_OBJ_SUCCESS;
}

// ---------------------------------------------------------------------------
// Token generation (Pattern B)
// ---------------------------------------------------------------------------

std::string
CuObjTokenManager::generatePutToken(void *data_ptr, size_t size) {
    return generateToken(data_ptr, size, CUOBJ_PUT);
}

std::string
CuObjTokenManager::generateGetToken(void *data_ptr, size_t size) {
    return generateToken(data_ptr, size, CUOBJ_GET);
}

std::string
CuObjTokenManager::generateToken(void *data_ptr, size_t size, cuObjOpType_t op) {
    if (!data_ptr || size == 0) {
        throw std::runtime_error("generateToken: invalid arguments");
    }

    // Each page is registered at its exact address, so buffer_offset is 0.
    // cuMemObjGetRDMAToken generates a token for [data_ptr, data_ptr+size).
    char *desc_str = nullptr;
    cuObjErr_t rc = client_->cuMemObjGetRDMAToken(data_ptr, size, 0, op, &desc_str);
    if (rc != CU_OBJ_SUCCESS || !desc_str) {
        throw std::runtime_error("cuMemObjGetRDMAToken failed");
    }

    // Copy the token, then free the library-allocated string.
    // After PutRDMAToken the RDMA registration remains valid — only the
    // string allocation is freed (cuObject spec section 1.5.5).
    std::string token(desc_str);
    client_->cuMemObjPutRDMAToken(desc_str);

    return token;
}
