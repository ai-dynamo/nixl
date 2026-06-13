/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_OBJ_S3_ACCEL_GENERIC_ENGINE_IMPL_H
#define NIXL_SRC_PLUGINS_OBJ_S3_ACCEL_GENERIC_ENGINE_IMPL_H

#include "s3/engine_impl.h"

/**
 * Standard-S3, protocol-compliant S3-over-RDMA engine (the preferred engine).
 *
 * Selected by `accelerated=true` with no `type` (or `type=s3`). It carries
 * no new logic: DefaultObjEngineImpl already advertises VRAM and registers RDMA
 * buffers gated on the client's supportsRdma(), and the standard S3 client
 * enables its RDMA fast path because the params carry accelerated=true with no
 * type. It is preferred because it requires no per-vendor code and speaks only
 * the published, vendor-neutral `x-amz-rdma-*` protocol.
 */
class GenericObjEngineImpl : public DefaultObjEngineImpl {
public:
    explicit GenericObjEngineImpl(const nixlBackendInitParams *init_params)
        : DefaultObjEngineImpl(init_params) {}

    GenericObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iS3Client> s3_client,
                         std::shared_ptr<iS3Client> s3_client_crt)
        : DefaultObjEngineImpl(init_params, std::move(s3_client), std::move(s3_client_crt)) {}
};

#endif // NIXL_SRC_PLUGINS_OBJ_S3_ACCEL_GENERIC_ENGINE_IMPL_H
