/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
#define NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H

#include "s3_accel/engine_impl.h"
#include "s3_accel/dell/cuobj_token_manager.h"

/**
 * S3 Dell ObjectScale Engine Implementation (Pattern B).
 *
 * Provides RDMA-accelerated S3 object storage operations for Dell ObjectScale.
 * Inherits from S3AccelObjEngineImpl and overrides only what is Dell-specific:
 *
 *   - getSupportedMems() → adds VRAM_SEG to the supported set.
 *   - registerMem()      → registers DRAM/VRAM with cuObject for RDMA.
 *   - deregisterMem()    → deregisters DRAM/VRAM from cuObject.
 *   - getClient()        → returns the Dell RDMA client.
 *
 * The transfer lifecycle (prepXfer, postXfer, checkXfer, releaseReqH) is
 * inherited unchanged from DefaultObjEngineImpl.  The parent's postXfer
 * calls getClient()->putObjectAsync() / getObjectAsync(), which the Dell
 * client overrides to inject RDMA tokens transparently.
 */
class S3DellObsObjEngineImpl : public S3AccelObjEngineImpl {
public:
    /**
     * Construct the Dell engine.
     * Creates a CuObjTokenManager and an awsS3DellObsClient.
     *
     * If the optional backend parameter "rdma_pool_size" is set, the first
     * DRAM/VRAM registration will be expanded to cover the entire pool,
     * so that subsequent page-sized registrations are instant no-ops.
     *
     * @param init_params  Backend initialisation parameters.
     */
    explicit S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * Construct the Dell engine with an injected S3 client (for testing).
     *
     * When a non-null s3_client is provided, it is used as-is (typically a
     * mock).  The CuObjTokenManager is still created but may not be
     * connected in a test environment.
     *
     * @param init_params  Backend initialisation parameters.
     * @param s3_client    Pre-configured S3 client (can be a mock).
     */
    S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                           std::shared_ptr<iS3Client> s3_client);

    /**
     * @return {OBJ_SEG, DRAM_SEG, VRAM_SEG} — the Dell engine supports
     *         GPU-direct transfers in addition to DRAM.
     */
    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

    /**
     * Register memory with the backend for RDMA operations.
     *
     * - OBJ_SEG: delegated to the parent (devId → object key mapping).
     * - DRAM_SEG / VRAM_SEG: registered with cuObject via the token manager.
     *   The token manager de-duplicates: if the address falls within an
     *   already-registered pool region, the call is a refcount increment.
     *
     * @param mem       Memory blob descriptor (addr, len, devId, metaInfo).
     * @param nixl_mem  Memory type.
     * @param out       Output backend metadata handle.
     * @return NIXL_SUCCESS, NIXL_ERR_BACKEND, or NIXL_ERR_NOT_SUPPORTED.
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem,
                const nixl_mem_t &nixl_mem,
                nixlBackendMD *&out) override;

    /**
     * Deregister memory from the backend.
     *
     * - DRAM/VRAM metadata (nixlDellMemMetadata): deregisters from cuObject.
     * - OBJ_SEG metadata: delegated to the parent.
     *
     * @param meta  Backend metadata handle returned by registerMem().
     * @return NIXL_SUCCESS or NIXL_ERR_BACKEND.
     */
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    // prepXfer   — INHERITED from DefaultObjEngineImpl (validates, creates handle).
    // postXfer   — INHERITED from DefaultObjEngineImpl (calls client->putObjectAsync/getObjectAsync).
    // checkXfer  — INHERITED from DefaultObjEngineImpl (polls futures).
    // releaseReqH — INHERITED from DefaultObjEngineImpl (deletes handle).

protected:
    /**
     * @return The Dell RDMA S3 client (or the injected mock).
     */
    iS3Client *
    getClient() const override;

private:
    std::shared_ptr<iS3Client> s3Client_;
    std::shared_ptr<CuObjTokenManager> tokenMgr_;

    /**
     * Optional pool size hint read from the "rdma_pool_size" backend parameter.
     * When non-zero, the first DRAM/VRAM registerMem is expanded to cover
     * [addr, addr+rdmaPoolSize_) so that all subsequent pages within the
     * same contiguous buffer are de-duplicated by the token manager.
     */
    size_t rdmaPoolSize_ = 0;
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
