/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H
#define OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H

#include "s3/engine_impl.h"
#include "s3_accel/client.h"

/**
 * Generic, protocol-compliant S3-over-RDMA engine (selected by `accelerated=true`
 * with no `type` / `type=s3`; registered under "s3").
 *
 * It composes the shared RDMA transport through awsS3AccelClient and owns the
 * accel-layer concerns that the plain HTTP base must not carry: it advertises
 * VRAM_SEG, and pins/unpins DRAM and VRAM buffers for RDMA in registerMem/
 * deregisterMem. The base DefaultObjEngineImpl is unchanged.
 */
class S3AccelObjEngineImpl : public DefaultObjEngineImpl {
public:
    explicit S3AccelObjEngineImpl(const nixlBackendInitParams *init_params);
    S3AccelObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iS3Client> s3_client);

    nixl_mem_list_t
    getSupportedMems() const override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

protected:
    iS3Client *
    getClient() const override;

private:
    // True iff the client's generic S3-over-RDMA fast path is fully ready. Gates
    // VRAM advertisement and buffer pinning. False for an injected non-accel
    // client (e.g. a test mock).
    [[nodiscard]] bool
    rdmaReady() const;

    awsS3AccelClient *accelClient_ = nullptr;
};

#endif // OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H
