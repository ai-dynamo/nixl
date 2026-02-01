/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "s3_accel/client.h"
#include "common/nixl_log.h"

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params)
    : DefaultObjEngineImpl(init_params) {
    s3Client_ = std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Accel client";
}

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iS3Client> s3_client)
    : DefaultObjEngineImpl(init_params) {
    s3Client_ = s3_client ?
        s3_client :
        std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Accel client (injected)";
}

iS3Client *
S3AccelObjEngineImpl::getClient() const {
    return s3Client_.get();
}
