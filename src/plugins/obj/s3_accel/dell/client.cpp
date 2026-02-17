/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "object/s3/utils.h"
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <absl/strings/str_format.h>
#include <iostream>
#include "common/nixl_log.h"

awsS3DellObsClient::awsS3DellObsClient(nixl_b_params_t *custom_params,
                                       std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor) {
    NIXL_DEBUG << "Initialized Dell ObjectScale Client";
}

void
awsS3DellObsClient::putObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       const std::string &rdma_desc,
                                       put_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat("putObjectRdmaAsync: key=%s, data_ptr=0x%lx, data_len=%zu, offset=%zu, rdma_desc=%s",
                                  std::string(key).c_str(), data_ptr, data_len, offset,
                                  rdma_desc.empty() ? "<empty>" : rdma_desc.c_str());

    if (offset != 0) {
        NIXL_ERROR << "putObjectRdmaAsync: offset is not 0, returning failure";
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    if (!rdma_desc.empty()) {
        request.SetAdditionalCustomHeaderValue("x-rdma-info", rdma_desc);
    }
    request.SetContentLength(0);

    NIXL_DEBUG << "putObjectRdmaAsync: sending PutObjectAsync request";
    s3Client_->PutObjectAsync(
        request,
        [callback](const Aws::S3::S3Client *,
                   const Aws::S3::Model::PutObjectRequest &,
                   const Aws::S3::Model::PutObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (outcome.IsSuccess()) {
                NIXL_DEBUG << "putObjectRdmaAsync: PutObjectAsync completed with success";
            } else {
                const auto &error = outcome.GetError();
                NIXL_ERROR << absl::StrFormat("putObjectRdmaAsync: PutObjectAsync failed - %s: %s (HTTP %d)",
                                              error.GetExceptionName().c_str(),
                                              error.GetMessage().c_str(),
                                              static_cast<int>(error.GetResponseCode()));
            }
            callback(outcome.IsSuccess());
        },
        nullptr);
}

void
awsS3DellObsClient::getObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       const std::string &rdma_desc,
                                       get_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat("getObjectRdmaAsync: key=%s, data_ptr=0x%lx, data_len=%zu, offset=%zu, rdma_desc=%s",
                                  std::string(key).c_str(), data_ptr, data_len, offset,
                                  rdma_desc.empty() ? "<empty>" : rdma_desc.c_str());

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));

    if (!rdma_desc.empty()) {
        request.SetAdditionalCustomHeaderValue("x-rdma-info", rdma_desc);
    }

    NIXL_DEBUG << absl::StrFormat("getObjectRdmaAsync: sending GetObjectAsync request with range bytes=%d-%d",
                                  offset, offset + data_len - 1);
    s3Client_->GetObjectAsync(
        request,
        [callback](const Aws::S3::S3Client *,
                   const Aws::S3::Model::GetObjectRequest &,
                   const Aws::S3::Model::GetObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (outcome.IsSuccess()) {
                NIXL_DEBUG << "getObjectRdmaAsync: GetObjectAsync completed with success";
            } else {
                const auto &error = outcome.GetError();
                NIXL_ERROR << absl::StrFormat("getObjectRdmaAsync: GetObjectAsync failed - %s: %s (HTTP %d)",
                                              error.GetExceptionName().c_str(),
                                              error.GetMessage().c_str(),
                                              static_cast<int>(error.GetResponseCode()));
            }
            callback(outcome.IsSuccess());
        },
        nullptr);
}
