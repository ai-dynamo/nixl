/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "object/s3/utils.h"
#include "object/s3/aws_sdk_init.h"
#include "engine_utils.h"
#include <aws/s3-crt/model/PutObjectRequest.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/HeadObjectRequest.h>
#include <aws/s3-crt/S3CrtErrors.h>
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

    // Align the CRT multipart thresholds with crtMinLimit so that every object
    // routed to this client (size >= crtMinLimit) is uploaded via multipart.
    // If crtMinLimit < 5 MiB the CRT SDK clamps partSize to 5 MiB internally
    // (with a warning log) while keeping multipartUploadThreshold at the user
    // value, so MPU still activates at crtMinLimit — but the effective part
    // size will be 5 MiB regardless.
    const size_t crt_min_limit = getCrtMinLimit(custom_params);
    if (crt_min_limit > 0) {
        config.partSize = crt_min_limit;
        config.multipartUploadThreshold = crt_min_limit;
    }

    auto credentials_opt = nixl_s3_utils::createAWSCredentials(custom_params);
    bool use_virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);
    config.useVirtualAddressing = use_virtual_addressing;

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

    // Heap-allocate the request so it outlives this function: the CRT SDK stores
    // a raw pointer to it (userData->originalRequest) and dereferences it in
    // S3CrtRequestHeadersCallback after putObjectAsync() has returned.
    auto request = Aws::MakeShared<Aws::S3Crt::Model::PutObjectRequest>("PutObjectRequest");
    request->WithBucket(bucketName_).WithKey(Aws::String(key));

    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "PutObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto data_stream =
        Aws::MakeShared<Aws::IOStream>("PutObjectInputStream", preallocated_stream_buf.get());
    request->SetBody(data_stream);

    s3CrtClient_->PutObjectAsync(
        *request,
        [callback, preallocated_stream_buf, data_stream, request](
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

    // Heap-allocate the request for the same reason as putObjectAsync: the SDK
    // stores a raw pointer to it (userData->originalRequest) used in callbacks
    // that fire after getObjectAsync() has returned.
    auto request = Aws::MakeShared<Aws::S3Crt::Model::GetObjectRequest>("GetObjectRequest");
    request->WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    request->SetResponseStreamFactory(*stream_factory.get());

    s3CrtClient_->GetObjectAsync(
        *request,
        [callback, stream_factory, request](
            const Aws::S3Crt::S3CrtClient *,
            const Aws::S3Crt::Model::GetObjectRequest &,
            const Aws::S3Crt::Model::GetObjectOutcome &outcome,
            const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (!outcome.IsSuccess())
                NIXL_ERROR << "getObjectAsync (CRT) error: " << outcome.GetError().GetMessage();
            callback(outcome.IsSuccess());
        },
        nullptr);
}

void
awsS3CrtClient::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    Aws::S3Crt::Model::HeadObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    s3CrtClient_->HeadObjectAsync(
        request,
        [callback](const Aws::S3Crt::S3CrtClient *,
                   const Aws::S3Crt::Model::HeadObjectRequest &,
                   const Aws::S3Crt::Model::HeadObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (outcome.IsSuccess()) {
                callback(true);
            } else {
                const auto &error = outcome.GetError();
                auto error_type = error.GetErrorType();
                auto exception_name = error.GetExceptionName();

                // Object-missing indicators: check both error type enum and
                // exception name string, since HeadObject 404 responses lack
                // an XML body and the CRT SDK may report different identifiers.
                bool is_object_missing = error_type == Aws::S3Crt::S3CrtErrors::NO_SUCH_KEY ||
                    error_type == Aws::S3Crt::S3CrtErrors::RESOURCE_NOT_FOUND ||
                    exception_name == "NoSuchKey" || exception_name == "NotFound" ||
                    exception_name == "ResourceNotFound";

                bool is_bucket_error = error_type == Aws::S3Crt::S3CrtErrors::NO_SUCH_BUCKET ||
                    exception_name == "NoSuchBucket";

                if (is_object_missing) {
                    callback(false);
                } else if (is_bucket_error) {
                    NIXL_ERROR << "checkObjectExistsAsync (CRT) bucket/endpoint error: "
                               << error.GetMessage();
                    callback(std::nullopt);
                } else {
                    NIXL_ERROR << "checkObjectExistsAsync (CRT) error: " << error.GetMessage();
                    callback(std::nullopt);
                }
            }
        },
        nullptr);
}
