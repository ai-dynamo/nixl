/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuobj_client.h"

#include <exception>

#include "common/nixl_log.h"

namespace nixl_obj_rdma {

SharedCuObjClient::SharedCuObjClient() {
    try {
        // The token-based flow does not use the get/put callbacks, so empty ops
        // suffice (matches the reference SDKs' availability probe).
        client_ = std::make_unique<cuObjClient>(ops_, CUOBJ_PROTO_RDMA_DC_V1);
        connected_ = client_ && client_->isConnected();
        if (connected_) {
            NIXL_INFO << "S3 RDMA fabric connected (cuObject)";
        }
        // A not-connected fabric is not logged here and does not imply an HTTP
        // fallback: instance() returns nullptr and the caller decides (a hard
        // error under accelerated=true).
    }
    catch (const std::exception &e) {
        NIXL_WARN << "cuObjClient init failed: " << e.what();
        connected_ = false;
    }
}

SharedCuObjClient *
SharedCuObjClient::instance() {
    static SharedCuObjClient inst;
    return inst.connected_ ? &inst : nullptr;
}

bool
SharedCuObjClient::registerBuffer(void *ptr, size_t size) {
    const std::lock_guard<std::mutex> lock(mutex_);
    cuObjErr_t rc = client_->cuMemObjGetDescriptor(ptr, size);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed rc=" << rc << " ptr=" << ptr
                   << " size=" << size;
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    std::erase_if(registrations_, [base](const RdmaRegistration &reg) { return reg.base == base; });
    registrations_.push_back(RdmaRegistration{base, size});
    NIXL_DEBUG << "cuMemObjGetDescriptor OK ptr=" << ptr << " size=" << size;
    return true;
}

void
SharedCuObjClient::deregisterBuffer(void *ptr) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (client_->cuMemObjPutDescriptor(ptr) != CU_OBJ_SUCCESS) {
        // Leave the record in place: the descriptor may still be live, and a
        // later token still has to be minted from this base.
        NIXL_WARN << "cuMemObjPutDescriptor failed for ptr " << ptr;
        return;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    std::erase_if(registrations_, [base](const RdmaRegistration &reg) { return reg.base == base; });
}

bool
SharedCuObjClient::isDeviceMemory(const void *ptr) const {
    // No mutex_: this is a static cuObjClient helper that inspects the pointer's
    // memory type via CUDA and touches no cuObjClient instance state, so it needs
    // no serialization against registerBuffer()/getToken().
    return cuObjClient::getMemoryType(ptr) == CUOBJ_MEMORY_CUDA_DEVICE;
}

char *
SharedCuObjClient::getToken(void *ptr, size_t size, size_t offset, cuObjOpType_t op) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto resolved =
        resolveRdmaTokenRange(registrations_, reinterpret_cast<uintptr_t>(ptr), size, offset);
    if (!resolved) {
        NIXL_ERROR << "cuMemObjGetRDMAToken address is not inside a registered buffer ptr=" << ptr
                   << " size=" << size << " offset=" << offset;
        return nullptr;
    }

    char *token = nullptr;
    void *base = reinterpret_cast<void *>(resolved->base);
    cuObjErr_t rc = client_->cuMemObjGetRDMAToken(base, size, resolved->offset, op, &token);
    if (rc != CU_OBJ_SUCCESS || token == nullptr) {
        NIXL_ERROR << "cuMemObjGetRDMAToken failed rc=" << rc << " ptr=" << ptr << " base=" << base
                   << " size=" << size << " offset=" << resolved->offset << " op=" << op
                   << " token=" << static_cast<void *>(token);
        return nullptr;
    }
    return token;
}

void
SharedCuObjClient::putToken(char *token) {
    if (token == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    client_->cuMemObjPutRDMAToken(token);
}

} // namespace nixl_obj_rdma
