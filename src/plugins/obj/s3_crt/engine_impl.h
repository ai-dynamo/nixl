/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_CRT_ENGINE_IMPL_H
#define OBJ_PLUGIN_S3_CRT_ENGINE_IMPL_H

#include "s3/engine_impl.h"

class S3CrtObjEngineImpl : public DefaultObjEngineImpl {
public:
    explicit S3CrtObjEngineImpl(const nixlBackendInitParams *init_params);
    S3CrtObjEngineImpl(const nixlBackendInitParams *init_params,
                       std::shared_ptr<iS3Client> s3_client,
                       std::shared_ptr<iS3Client> s3_client_crt);

    // The CRT client is an HTTP multipart optimization for large objects and
    // cannot do GPU-direct RDMA, so the CRT engine does not advertise VRAM_SEG.
    // GPUDirect transfers require the default (non-CRT) configuration.
    nixl_mem_list_t
    getSupportedMems() const override {
        return {DRAM_SEG, OBJ_SEG};
    }

protected:
    iS3Client *
    getClient() const override;
    iS3Client *
    getClientForSize(size_t data_len) const override;

    std::shared_ptr<iS3Client> s3ClientCrt_;
};

#endif // OBJ_PLUGIN_S3_CRT_ENGINE_IMPL_H
