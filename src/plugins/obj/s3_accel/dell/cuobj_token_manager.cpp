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
    // Best-effort cleanup: deregister any regions whose refcount was not
    // driven to zero by the caller (e.g. during error recovery).
    std::lock_guard<std::mutex> lock(regions_mutex_);
    for (auto &region : regions_) {
        if (region.refcount > 0) {
            NIXL_WARN << "CuObjTokenManager destroyed with region still registered: "
                      << "base=0x" << std::hex << region.base
                      << " size=" << std::dec << region.size
                      << " refcount=" << region.refcount;
            if (client_) {
                client_->cuMemObjPutDescriptor(reinterpret_cast<void *>(region.base));
            }
        }
    }
    regions_.clear();
    client_.reset();
}

bool
CuObjTokenManager::isConnected() const {
    return client_ && client_->isConnected();
}

// ---------------------------------------------------------------------------
// Memory registration (pool-aware, de-duplicating)
// ---------------------------------------------------------------------------

cuObjErr_t
CuObjTokenManager::registerMemory(void *ptr, size_t size, size_t pool_hint) {
    if (!ptr || size == 0) {
        NIXL_ERROR << "registerMemory: invalid arguments (ptr="
                   << ptr << ", size=" << size << ")";
        return CU_OBJ_FAIL;
    }

    std::lock_guard<std::mutex> lock(regions_mutex_);

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    // Fast path: check if [addr, addr+size) already falls within a
    // registered region.  This handles the common case where the NIXL
    // core calls registerMem() once per page from a contiguous pool.
    RegisteredRegion *existing = findContainingRegion(addr, size);
    if (existing) {
        existing->refcount++;
        return CU_OBJ_SUCCESS;
    }

    // Slow path: this is a genuinely new region.
    // If a pool_hint was provided, expand the registration to cover the
    // entire pool starting at ptr, capped at the cuObject 4 GiB limit.
    // This way the first page from a pool triggers one cuMemObjGetDescriptor
    // and all subsequent pages are instant hits on the fast path above.
    size_t reg_size = size;
    if (pool_hint > size && pool_hint < CUOBJ_MAX_MEMORY_REG_SIZE) {
        reg_size = pool_hint;
    }
    if (reg_size >= CUOBJ_MAX_MEMORY_REG_SIZE) {
        NIXL_ERROR << "registerMemory: region size " << reg_size
                   << " exceeds cuObject limit (" << CUOBJ_MAX_MEMORY_REG_SIZE << ")";
        return CU_OBJ_FAIL;
    }

    cuObjErr_t rc = client_->cuMemObjGetDescriptor(ptr, reg_size);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuMemObjGetDescriptor failed: ptr=0x" << std::hex << addr
                   << " size=" << std::dec << reg_size;
        return rc;
    }

    regions_.push_back({addr, reg_size, 1});
    NIXL_DEBUG << "registerMemory: registered region base=0x" << std::hex << addr
               << " size=" << std::dec << reg_size;
    return CU_OBJ_SUCCESS;
}

cuObjErr_t
CuObjTokenManager::deregisterMemory(void *ptr, size_t size) {
    if (!ptr || size == 0) {
        return CU_OBJ_SUCCESS; // Nothing to do.
    }

    std::lock_guard<std::mutex> lock(regions_mutex_);

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    RegisteredRegion *region = findContainingRegion(addr, size);
    if (!region) {
        // Not within any registered region — benign for OBJ_SEG metadata
        // that was never registered with cuObject.
        NIXL_DEBUG << "deregisterMemory: ptr=0x" << std::hex << addr
                   << " not within any registered region (no-op)";
        return CU_OBJ_SUCCESS;
    }

    if (region->refcount == 0) {
        // Should not happen — defensive guard.
        NIXL_ERROR << "deregisterMemory: refcount already zero for region base=0x"
                   << std::hex << region->base;
        return CU_OBJ_FAIL;
    }

    region->refcount--;

    if (region->refcount == 0) {
        // Last reference gone — actually deregister with cuObject.
        cuObjErr_t rc = client_->cuMemObjPutDescriptor(
            reinterpret_cast<void *>(region->base));
        if (rc != CU_OBJ_SUCCESS) {
            // Re-increment refcount so the caller can retry.
            region->refcount++;
            NIXL_ERROR << "cuMemObjPutDescriptor failed for region base=0x"
                       << std::hex << region->base;
            return rc;
        }

        NIXL_DEBUG << "deregisterMemory: deregistered region base=0x"
                   << std::hex << region->base;

        // Remove the region from the vector.
        uintptr_t base_to_remove = region->base;
        regions_.erase(
            std::remove_if(regions_.begin(), regions_.end(),
                           [base_to_remove](const RegisteredRegion &r) {
                               return r.base == base_to_remove;
                           }),
            regions_.end());
    }

    return CU_OBJ_SUCCESS;
}

// ---------------------------------------------------------------------------
// Token generation (Pattern B)
// ---------------------------------------------------------------------------

std::string
CuObjTokenManager::generatePutToken(void *data_ptr, size_t size, size_t offset) {
    return generateToken(data_ptr, size, offset, CUOBJ_PUT);
}

std::string
CuObjTokenManager::generateGetToken(void *data_ptr, size_t size, size_t offset) {
    return generateToken(data_ptr, size, offset, CUOBJ_GET);
}

std::string
CuObjTokenManager::generateToken(void *data_ptr, size_t size,
                                 size_t offset, cuObjOpType_t op) {
    if (!data_ptr || size == 0) {
        throw std::runtime_error("generateToken: invalid arguments");
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);

    // Find the registered region that contains this data pointer.
    RegisteredRegion *region = nullptr;
    {
        std::lock_guard<std::mutex> lock(regions_mutex_);
        region = findContainingRegion(addr, size);
    }
    if (!region) {
        throw std::runtime_error(
            "generateToken: data_ptr not within any registered region");
    }

    // Compute the offset relative to the registered pool base.
    // cuMemObjGetRDMAToken wants (pool_base, transfer_size, buffer_offset, op).
    // buffer_offset = (data_ptr - pool_base) + caller's object offset.
    size_t buffer_offset = (addr - region->base) + offset;

    char *desc_str = nullptr;
    cuObjErr_t rc = client_->cuMemObjGetRDMAToken(
        reinterpret_cast<void *>(region->base),
        size, buffer_offset, op, &desc_str);
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

// ---------------------------------------------------------------------------
// Region lookup
// ---------------------------------------------------------------------------

CuObjTokenManager::RegisteredRegion *
CuObjTokenManager::findContainingRegion(uintptr_t addr, size_t size) {
    // Linear scan is fine — typically 1–2 regions (one DRAM pool, maybe one VRAM pool).
    for (auto &region : regions_) {
        if (addr >= region.base &&
            (addr + size) <= (region.base + region.size)) {
            return &region;
        }
    }
    return nullptr;
}
