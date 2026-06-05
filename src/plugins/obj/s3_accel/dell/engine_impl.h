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
 * Inherits from S3AccelObjEngineImpl and overrides Dell-specific behaviour:
 *
 *   - getSupportedMems() → adds VRAM_SEG to the supported set.
 *   - registerMem()      → registers DRAM/VRAM with cuObject for RDMA.
 *   - deregisterMem()    → deregisters DRAM/VRAM from cuObject.
 *   - getClient()        → returns the Dell RDMA client.
 *   - postXfer()         → rejects non-zero PUT offsets, then delegates
 *                           to the parent whose putObjectAsync/getObjectAsync
 *                           the Dell client overrides to inject RDMA tokens.
 *
 * The remaining transfer lifecycle (prepXfer, checkXfer, releaseReqH) is
 * inherited unchanged from DefaultObjEngineImpl.
 */
class S3DellObsObjEngineImpl : public S3AccelObjEngineImpl {
public:
    /**
     * Construct the Dell engine.
     * Creates a CuObjTokenManager and an awsS3DellObsClient.
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
     *   Each page is registered individually at its exact address and size.
     *
     * @param mem       Memory blob descriptor (addr, len, devId, metaInfo).
     * @param nixl_mem  Memory type.
     * @param out       Output backend metadata handle.
     * @return NIXL_SUCCESS, NIXL_ERR_BACKEND, or NIXL_ERR_NOT_SUPPORTED.
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

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

    /**
     * @brief Validate a transfer request and dispatch it.
     *
     * For WRITE operations, rejects non-zero remote offsets before launching
     * any async PUTs.  Dell ObjectScale RDMA PUT does not support partial
     * writes at a non-zero offset; failing early prevents partially-enqueued
     * multi-descriptor requests.
     *
     * @param operation    Transfer direction (NIXL_READ or NIXL_WRITE).
     * @param local        Local descriptor list.
     * @param remote       Remote descriptor list.
     * @param remote_agent Remote agent identifier.
     * @param handle       Backend request handle (from prepXfer).
     * @param opt_args     Optional backend arguments.
     * @return NIXL_ERR_INVALID_PARAM for invalid WRITE offsets; otherwise the
     *         status from S3AccelObjEngineImpl::postXfer().
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    // prepXfer   — INHERITED from DefaultObjEngineImpl (validates, creates handle).
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
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_ENGINE_IMPL_H
