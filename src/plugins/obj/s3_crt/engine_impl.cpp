/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "engine_utils.h"
#include "s3/client.h"
#include "s3_crt/client.h"
#include "common/nixl_log.h"
#include <limits>

S3CrtObjEngineImpl::S3CrtObjEngineImpl(const nixlBackendInitParams *init_params)
    : DefaultObjEngineImpl(init_params) {
    if (crtMinLimit_ == 1) {
        s3Client_.reset();
    }

    s3ClientCrt_ = std::make_shared<awsS3CrtClient>(init_params->customParams, executor_);

    if (s3Client_) {
        NIXL_INFO << "Object storage backend initialized with dual S3 CRT clients";
        NIXL_INFO << "S3 CRT client enabled for objects >= " << crtMinLimit_ << " bytes";
    } else {
        NIXL_INFO << "Object storage backend initialized with S3 CRT client only";
    }
}

S3CrtObjEngineImpl::S3CrtObjEngineImpl(const nixlBackendInitParams *init_params,
                                       std::shared_ptr<iS3Client> s3_client,
                                       std::shared_ptr<iS3Client> s3_client_crt)
    : DefaultObjEngineImpl(init_params, s3_client, s3_client_crt) {
    if (crtMinLimit_ == 1) {
        s3Client_.reset();
    }

    s3ClientCrt_ = s3_client_crt;
    if (!s3ClientCrt_) {
        s3ClientCrt_ = std::make_shared<awsS3CrtClient>(init_params->customParams, executor_);
    }
}

iS3Client *
S3CrtObjEngineImpl::getClient() const {
    return s3ClientCrt_ ? s3ClientCrt_.get() : s3Client_.get();
}

iS3Client *
S3CrtObjEngineImpl::getClientForSize(size_t data_len) const {
    if (!s3ClientCrt_) return s3Client_.get();
    if (!s3Client_ || data_len >= crtMinLimit_) return s3ClientCrt_.get();
    return s3Client_.get();
}
