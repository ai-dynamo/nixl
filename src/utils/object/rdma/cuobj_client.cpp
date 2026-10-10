/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuobj_client.h"

#include <algorithm>
#include <exception>
#include <iterator>

#include "common/nixl_log.h"
#include "rdma_protocol.h"

namespace nixl_obj_rdma {

// cuObject refuses one registration of CUOBJ_MAX_MEMORY_REG_SIZE or more (4 GiB in
// cuObject 13.2, 4 GiB - 64 KiB in 13.4); larger buffers are registered as
// consecutive pieces of this length.
constexpr size_t registration_piece_size = size_t{2} << 30;

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
    const auto pieces = registrationPieces(
        reinterpret_cast<uintptr_t>(ptr), size, CUOBJ_MAX_MEMORY_REG_SIZE, registration_piece_size);
    for (size_t i = 0; i < pieces.size(); ++i) {
        const auto [start, len] = pieces[i];
        cuObjErr_t rc = client_->cuMemObjGetDescriptor(reinterpret_cast<void *>(start), len);
        if (rc != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "cuMemObjGetDescriptor failed rc=" << rc << " ptr=" << ptr
                       << " size=" << size << " piece=" << i << "/" << pieces.size();
            // Roll back the pieces already registered.
            for (size_t j = 0; j < i; ++j) {
                registrations_.erase(pieces[j].first);
                client_->cuMemObjPutDescriptor(reinterpret_cast<void *>(pieces[j].first));
            }
            return false;
        }
        registrations_[start] = len;
    }
    if (pieces.size() > 1) {
        pieced_buffers_[reinterpret_cast<uintptr_t>(ptr)] = size;
    }
    NIXL_DEBUG << "cuMemObjGetDescriptor OK ptr=" << ptr << " size=" << size
               << " pieces=" << pieces.size();
    return true;
}

void
SharedCuObjClient::deregisterBuffer(void *ptr) {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<uintptr_t, size_t>> pieces{{reinterpret_cast<uintptr_t>(ptr), 0}};
    if (const auto it = pieced_buffers_.find(reinterpret_cast<uintptr_t>(ptr));
        it != pieced_buffers_.end()) {
        pieces = registrationPieces(
            it->first, it->second, CUOBJ_MAX_MEMORY_REG_SIZE, registration_piece_size);
        pieced_buffers_.erase(it);
    }
    for (const auto &[start, len] : pieces) {
        registrations_.erase(start);
        if (client_->cuMemObjPutDescriptor(reinterpret_cast<void *>(start)) != CU_OBJ_SUCCESS) {
            NIXL_WARN << "cuMemObjPutDescriptor failed for ptr " << reinterpret_cast<void *>(start);
        }
    }
}

bool
SharedCuObjClient::isDeviceMemory(const void *ptr) const {
    // No mutex_: this is a static cuObjClient helper that inspects the pointer's
    // memory type via CUDA and touches no cuObjClient instance state, so it needs
    // no serialization against registerBuffer()/getToken().
    return cuObjClient::getMemoryType(ptr) == CUOBJ_MEMORY_CUDA_DEVICE;
}

char *
SharedCuObjClient::getToken(void *ptr, size_t size, cuObjOpType_t op) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    auto contains = [addr, size](const auto &reg) {
        const auto &[base, len] = reg;
        return addr - base <= len && size <= len - (addr - base);
    };
    const auto rit = std::find_if(std::make_reverse_iterator(registrations_.upper_bound(addr)),
                                  registrations_.rend(),
                                  contains);
    if (rit == registrations_.rend()) {
        if (std::any_of(pieced_buffers_.begin(), pieced_buffers_.end(), contains)) {
            NIXL_ERROR << "RDMA range ptr=" << ptr << " size=" << size
                       << " crosses a registration piece boundary: buffers of "
                       << CUOBJ_MAX_MEMORY_REG_SIZE << " bytes or more are registered as "
                       << registration_piece_size << "-byte pieces";
        } else {
            NIXL_ERROR << "No RDMA registration contains ptr=" << ptr << " size=" << size;
        }
        return nullptr;
    }
    const auto base = rit->first;
    const size_t offset = addr - base;

    char *token = nullptr;
    cuObjErr_t rc =
        client_->cuMemObjGetRDMAToken(reinterpret_cast<void *>(base), size, offset, op, &token);
    if (rc != CU_OBJ_SUCCESS || token == nullptr) {
        NIXL_ERROR << "cuMemObjGetRDMAToken failed rc=" << rc
                   << " base=" << reinterpret_cast<void *>(base) << " offset=" << offset
                   << " size=" << size << " op=" << op << " token=" << static_cast<void *>(token);
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
