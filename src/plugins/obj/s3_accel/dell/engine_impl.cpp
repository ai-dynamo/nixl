/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "client.h"
#include "common/nixl_log.h"
#include <algorithm>
#include <memory>

namespace {

/**
 * Backend metadata for DRAM/VRAM memory registered with cuObject.
 *
 * OBJ_SEG metadata is handled by the parent class (nixlObjMetadata in
 * s3/engine_impl.cpp).  This class is only used for DRAM/VRAM segments
 * that need cuObject descriptor registration for RDMA.
 */
class nixlDellMemMetadata : public nixlBackendMD {
public:
    /**
     * @param type  Memory type (DRAM_SEG or VRAM_SEG).
     * @param a     Start address of the registered region.
     * @param l     Length in bytes.
     */
    nixlDellMemMetadata(nixl_mem_t type, uintptr_t a, size_t l)
        : nixlBackendMD(true), memType(type), addr(a), len(l) {}

    ~nixlDellMemMetadata() = default;

    nixl_mem_t memType;   ///< DRAM_SEG or VRAM_SEG.
    uintptr_t  addr;      ///< Start address (for deregisterMemory).
    size_t     len;       ///< Length in bytes.
};

} // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(
        const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params) {
    // Create the token manager (Pattern B — no callbacks).
    tokenMgr_ = std::make_shared<CuObjTokenManager>(CUOBJ_PROTO_RDMA_DC_V1);
    if (!tokenMgr_->isConnected()) {
        NIXL_ERROR << "CuObjTokenManager failed to connect";
        return;
    }

    // Create the Dell RDMA client that uses the token manager.
    s3Client_ = std::make_shared<awsS3DellObsClient>(
        init_params->customParams, tokenMgr_, executor_);
    NIXL_INFO << "Dell ObjectScale engine initialized (Pattern B)";
}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(
        const nixlBackendInitParams *init_params,
        std::shared_ptr<iS3Client> s3_client)
    : S3AccelObjEngineImpl(init_params, s3_client) {
    // Use the injected client if provided (testing), otherwise create one.
    if (s3_client) {
        s3Client_ = s3_client;
    } else {
        tokenMgr_ = std::make_shared<CuObjTokenManager>(CUOBJ_PROTO_RDMA_DC_V1);
        if (!tokenMgr_->isConnected()) {
            NIXL_ERROR << "CuObjTokenManager failed to connect";
            return;
        }
        s3Client_ = std::make_shared<awsS3DellObsClient>(
            init_params->customParams, tokenMgr_, executor_);
    }

    NIXL_INFO << "Dell ObjectScale engine initialized (Pattern B, injected client)";
}

// ---------------------------------------------------------------------------
// Memory registration
// ---------------------------------------------------------------------------

nixl_status_t
S3DellObsObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                    const nixl_mem_t &nixl_mem,
                                    nixlBackendMD *&out) {
    // Check supported memory types.
    auto supported = getSupportedMems();
    if (std::find(supported.begin(), supported.end(), nixl_mem) == supported.end())
        return NIXL_ERR_NOT_SUPPORTED;

    // OBJ_SEG: delegate to the parent for devId → object key mapping.
    if (nixl_mem == OBJ_SEG) {
        return S3AccelObjEngineImpl::registerMem(mem, nixl_mem, out);
    }

    // DRAM_SEG or VRAM_SEG: register with cuObject for RDMA.
    // If tokenMgr_ is null (test environment with injected mock client),
    // skip cuObject registration and return null metadata like the parent.
    if (!tokenMgr_) {
        out = nullptr;
        return NIXL_SUCCESS;
    }

    if (!tokenMgr_->isConnected()) {
        NIXL_ERROR << "CuObjTokenManager is not connected";
        return NIXL_ERR_BACKEND;
    }

    // Register this page at its exact address and size (1:1 mapping).
    // Registration happens at init time, not on the hot path.
    cuObjErr_t rc = tokenMgr_->registerMemory(
        reinterpret_cast<void *>(mem.addr), mem.len);
    if (rc != CU_OBJ_SUCCESS) {
        NIXL_ERROR << "cuObject memory registration failed for addr=0x"
                   << std::hex << mem.addr << " len=" << std::dec << mem.len;
        return NIXL_ERR_BACKEND;
    }

    // Return metadata so deregisterMem can clean up the cuObject registration.
    out = new nixlDellMemMetadata(nixl_mem, mem.addr, mem.len);
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// Memory deregistration
// ---------------------------------------------------------------------------

nixl_status_t
S3DellObsObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    if (!meta) {
        return NIXL_SUCCESS; // Nothing to do.
    }

    // Check if this is Dell-specific DRAM/VRAM metadata.
    // We use dynamic_cast here because the metadata could be either our
    // nixlDellMemMetadata (DRAM/VRAM) or the parent's nixlObjMetadata (OBJ_SEG).
    auto *dell_md = dynamic_cast<nixlDellMemMetadata *>(meta);
    if (!dell_md) {
        // Not our metadata — delegate to the parent for OBJ_SEG cleanup.
        return S3AccelObjEngineImpl::deregisterMem(meta);
    }

    // DRAM/VRAM: deregister from cuObject.
    nixl_status_t result = NIXL_SUCCESS;
    if (tokenMgr_) {
        cuObjErr_t rc = tokenMgr_->deregisterMemory(
            reinterpret_cast<void *>(dell_md->addr));
        if (rc != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "cuObject deregistration failed for addr=0x"
                       << std::hex << dell_md->addr;
            result = NIXL_ERR_BACKEND;
        }
    }

    // Always free the metadata (caller expects it to be consumed).
    delete dell_md;
    return result;
}

// ---------------------------------------------------------------------------
// Client accessor
// ---------------------------------------------------------------------------

iS3Client *
S3DellObsObjEngineImpl::getClient() const {
    return s3Client_.get();
}
