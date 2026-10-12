/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuobj_client.h"

#include <algorithm>
#include <exception>
#include <iterator>

#include "common/nixl_log.h"

namespace nixl_obj_rdma {

namespace {

    class cuObjClientApi : public cuObjApi {
    public:
        // The token-based flow does not use the get/put callbacks, so empty ops
        // suffice (matches the reference SDKs' availability probe).
        cuObjClientApi() : client_(ops_, CUOBJ_PROTO_RDMA_DC_V1) {}

        [[nodiscard]] bool
        isConnected() {
            return client_.isConnected();
        }

        cuObjErr_t
        cuMemObjGetDescriptor(void *ptr, size_t size) override {
            return client_.cuMemObjGetDescriptor(ptr, size);
        }

        cuObjErr_t
        cuMemObjPutDescriptor(void *ptr) override {
            return client_.cuMemObjPutDescriptor(ptr);
        }

        cuObjErr_t
        cuMemObjGetRDMAToken(void *ptr,
                             size_t size,
                             size_t buffer_offset,
                             cuObjOpType_t operation,
                             char **desc_str_out) override {
            return client_.cuMemObjGetRDMAToken(ptr, size, buffer_offset, operation, desc_str_out);
        }

        cuObjErr_t
        cuMemObjPutRDMAToken(char *desc_str) override {
            return client_.cuMemObjPutRDMAToken(desc_str);
        }

    private:
        CUObjIOOps ops_{};
        cuObjClient client_;
    };

} // namespace

SharedCuObjClient::SharedCuObjClient() {
    try {
        auto client = std::make_unique<cuObjClientApi>();
        connected_ = client->isConnected();
        client_ = std::move(client);
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

SharedCuObjClient::SharedCuObjClient(std::unique_ptr<cuObjApi> api)
    : client_(std::move(api)),
      connected_(client_ != nullptr) {}

SharedCuObjClient *
SharedCuObjClient::instance() {
    static SharedCuObjClient inst;
    return inst.connected_ ? &inst : nullptr;
}

bool
SharedCuObjClient::registerBuffer(void *ptr, size_t size) {
    const std::lock_guard<std::mutex> lock(mutex_);
    // cuObject registrations are process-wide, so agents sharing a buffer share
    // one registration rather than each calling cuMemObjGetDescriptor.
    const auto it = registrations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != registrations_.end()) {
        if (it->second.size != size) {
            NIXL_ERROR << "RDMA buffer ptr=" << ptr << " is already registered with size "
                       << it->second.size << ", cannot register it with size " << size;
            return false;
        }
        ++it->second.refs;
        return true;
    }
    cuObjErr_t rc = client_->cuMemObjGetDescriptor(ptr, size);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed rc=" << rc << " ptr=" << ptr
                   << " size=" << size;
        return false;
    }
    registrations_[reinterpret_cast<uintptr_t>(ptr)] = {size, 1};
    NIXL_DEBUG << "cuMemObjGetDescriptor OK ptr=" << ptr << " size=" << size;
    return true;
}

void
SharedCuObjClient::deregisterBuffer(void *ptr) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = registrations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != registrations_.end()) {
        release(it);
    }
}

void
SharedCuObjClient::release(std::map<uintptr_t, Registration>::iterator it) {
    if (--it->second.refs > 0) {
        return;
    }
    void *ptr = reinterpret_cast<void *>(it->first);
    registrations_.erase(it);
    if (client_->cuMemObjPutDescriptor(ptr) != CU_OBJ_SUCCESS) {
        NIXL_WARN << "cuMemObjPutDescriptor failed for ptr " << ptr;
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
        const auto &[base, registration] = reg;
        const auto len = registration.size;
        return addr - base <= len && size <= len - (addr - base);
    };
    const auto rit = std::find_if(std::make_reverse_iterator(registrations_.upper_bound(addr)),
                                  registrations_.rend(),
                                  contains);
    if (rit == registrations_.rend()) {
        NIXL_ERROR << "No RDMA registration contains ptr=" << ptr << " size=" << size;
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
    ++rit->second.refs;
    tokens_[token] = base;
    return token;
}

void
SharedCuObjClient::putToken(char *token) {
    if (token == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    client_->cuMemObjPutRDMAToken(token);
    const auto it = tokens_.find(token);
    if (it == tokens_.end()) {
        return;
    }
    const auto reg = registrations_.find(it->second);
    if (reg != registrations_.end()) {
        release(reg);
    } else {
        NIXL_ERROR << "No RDMA registration for token base="
                   << reinterpret_cast<void *>(it->second);
    }
    tokens_.erase(it);
}

} // namespace nixl_obj_rdma
