/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "object/s3/utils.h"
#include "object/s3/aws_sdk_init.h"
#include <aws/s3-crt/model/PutObjectRequest.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/HeadObjectRequest.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <absl/strings/str_format.h>
#include <iostream>
#include "common/nixl_log.h"

awsS3CrtClient::awsS3CrtClient(nixl_b_params_t *custom_params,
                               std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3Client(custom_params, executor) {
    // Initialize AWS SDK (thread-safe, only happens once)
    nixl_s3_utils::initAWSSDK();

    // Create S3 CRT client configuration
    Aws::S3Crt::ClientConfiguration config;
    nixl_s3_utils::configureClientCommon(config, custom_params);
    if (executor) config.executor = executor;

    auto credentials_opt = nixl_s3_utils::createAWSCredentials(custom_params);
    bool use_virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);

    if (credentials_opt.has_value())
        s3CrtClient_ = std::make_unique<Aws::S3Crt::S3CrtClient>(
            credentials_opt.value(),
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
    else
        s3CrtClient_ = std::make_unique<Aws::S3Crt::S3CrtClient>(
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
}

void
awsS3CrtClient::setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    throw std::runtime_error("awsS3CrtClient::setExecutor() not supported - "
                             "AWS SDK doesn't allow changing executor after client creation");
}

void
awsS3CrtClient::putObjectAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               put_object_callback_t callback) {
    if (offset != 0) {
        callback(false);
        return;
    }

    Aws::S3Crt::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "PutObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto data_stream =
        Aws::MakeShared<Aws::IOStream>("PutObjectInputStream", preallocated_stream_buf.get());
    request.SetBody(data_stream);

    s3CrtClient_->PutObjectAsync(
        request,
        [callback, preallocated_stream_buf, data_stream](
            const Aws::S3Crt::S3CrtClient *,
            const Aws::S3Crt::Model::PutObjectRequest &,
            const Aws::S3Crt::Model::PutObjectOutcome &outcome,
            const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (!outcome.IsSuccess()) {
                NIXL_ERROR << absl::StrFormat("putObjectAsync (CRT) error: %s",
                                              outcome.GetError().GetMessage());
            }
            callback(outcome.IsSuccess());
        },
        nullptr);
}

void
awsS3CrtClient::getObjectAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               get_object_callback_t callback) {
    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "GetObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto stream_factory = Aws::MakeShared<Aws::IOStreamFactory>(
        "GetObjectStreamFactory", [preallocated_stream_buf]() -> Aws::IOStream * {
            return new Aws::IOStream(preallocated_stream_buf.get());
        });

    Aws::S3Crt::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    request.SetResponseStreamFactory(*stream_factory.get());

    s3CrtClient_->GetObjectAsync(
        request,
        [callback, stream_factory](const Aws::S3Crt::S3CrtClient *,
                                   const Aws::S3Crt::Model::GetObjectRequest &,
                                   const Aws::S3Crt::Model::GetObjectOutcome &outcome,
                                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            callback(outcome.IsSuccess());
        },
        nullptr);
}

bool
awsS3CrtClient::checkObjectExists(std::string_view key) {
    Aws::S3Crt::Model::HeadObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    auto outcome = s3CrtClient_->HeadObject(request);
    if (outcome.IsSuccess())
        return true;
    else if (outcome.GetError().GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND)
        return false;
    else
        throw std::runtime_error("Failed to check if object exists (CRT): " +
                                 outcome.GetError().GetMessage());
}
