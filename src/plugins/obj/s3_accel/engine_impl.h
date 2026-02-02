/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H
#define OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H

#include "s3/engine_impl.h"

class S3AccelObjEngineImpl : public DefaultObjEngineImpl {
public:
    explicit S3AccelObjEngineImpl(const nixlBackendInitParams *init_params);
    S3AccelObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iS3Client> s3_client);

    // Inherits getSupportedMems() from DefaultObjEngineImpl returning {OBJ_SEG, DRAM_SEG}.
    // Vendor engines that support GPU-direct transfers should override this to include VRAM_SEG.

protected:
    iS3Client *
    getClient() const override;
};

#endif // OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_H
