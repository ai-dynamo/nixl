/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "client.h"
#include "engine_utils.h"
#include "object/s3/utils.h"
#include "object/s3/aws_sdk_init.h"
#include "common/nixl_log.h"
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/S3Errors.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <absl/strings/str_format.h>

awsS3Client::awsS3Client(nixl_b_params_t *custom_params,
                         std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    // RDMA is enabled only when the caller explicitly opts in with
    // accelerated=true (standard S3-over-RDMA, i.e. no type or type=s3).
    : awsS3Client(custom_params, executor, isGenericAccelRequested(custom_params)) {}

awsS3Client::awsS3Client(nixl_b_params_t *custom_params,
                         std::shared_ptr<Aws::Utils::Threading::Executor> executor,
                         bool enable_rdma_fast_path) {
    // Initialize AWS SDK (thread-safe, only happens once)
    nixl_s3_utils::initAWSSDK();

    Aws::Client::ClientConfiguration config;
    nixl_s3_utils::configureClientCommon(config, custom_params);
    if (executor) {
        config.executor = executor;
    }
    executor_ = executor;
    rdma_requested_ = enable_rdma_fast_path;

    auto credentials_opt = nixl_s3_utils::createAWSCredentials(custom_params);
    bool use_virtual_addressing = nixl_s3_utils::getUseVirtualAddressing(custom_params);
    bucketName_ = Aws::String(nixl_s3_utils::getBucketName(custom_params));

    if (credentials_opt.has_value()) {
        s3Client_ = std::make_unique<Aws::S3::S3Client>(
            credentials_opt.value(),
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
    } else {
        s3Client_ = std::make_unique<Aws::S3::S3Client>(
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
    }

#ifdef HAVE_CUOBJ_CLIENT
    // Opportunistically attach the RDMA fast path. When no cuObject fabric is
    // present (or the control plane fails to initialize) rdma_ stays null and
    // every transfer takes the plain HTTP path below. Disabled for the
    // accelerated/vendor clients, which manage RDMA on their own.
    if (enable_rdma_fast_path) {
        rdma_ = nixl_obj_rdma::SharedCuObjClient::instance();
    }
    if (rdma_) {
        rdmaCp_ = std::make_unique<nixl_obj_rdma::S3RdmaControlPlane>(custom_params);
        if (!rdmaCp_->valid()) {
            rdmaCp_.reset();
            rdma_ = nullptr;
        }
    }
#endif

    // Fail fast: an accelerated=true (generic S3-over-RDMA) client whose RDMA
    // fast path is not fully ready (no cuObjClient build, no executor, fabric
    // down, or control plane invalid) can never recover — setExecutor() is
    // unsupported and rdma_/rdmaCp_ are set once here. Surface it at construction
    // instead of hard-failing every transfer later.
    if (rdma_requested_ && !rdmaReady()) {
        throw std::runtime_error(
            "accelerated=true (generic S3-over-RDMA) requested but the "
            "S3-over-RDMA fast path is unavailable (requires a cuObjClient build, "
            "an executor, a reachable RDMA fabric, and a valid control plane)");
    }
}

bool
awsS3Client::rdmaReady() const {
#ifdef HAVE_CUOBJ_CLIENT
    return rdma_ != nullptr && rdmaCp_ != nullptr && executor_ != nullptr;
#else
    return false;
#endif
}

bool
awsS3Client::supportsRdma() const {
    return rdmaReady();
}

void
awsS3Client::setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    throw std::runtime_error("AwsS3Client::setExecutor() not supported - "
                             "AWS SDK doesn't allow changing executor after client creation");
}

void
awsS3Client::putObjectAsync(std::string_view key,
                            uintptr_t data_ptr,
                            size_t data_len,
                            size_t offset,
                            put_object_callback_t callback) {
    if (offset != 0) {
        callback(false);
        return;
    }

    // Under accelerated=true, transfers are RDMA-only: if the fast path is not
    // ready we fail rather than silently write a body-less (0-byte) PUT over
    // HTTP. No auto-fallback today (see header).
    if (rdma_requested_) {
#ifdef HAVE_CUOBJ_CLIENT
        if (rdmaReady()) {
            executor_->Submit([this, k = std::string(key), data_ptr, data_len, callback]() {
                nixl_obj_rdma::S3RdmaClientCtx ctx;
                ctx.bucket = bucketName_.c_str();
                ctx.object = k;
                const ssize_t r = nixl_obj_rdma::rdmaPutWithRetry(
                    *rdma_, *rdmaCp_, ctx, reinterpret_cast<void *>(data_ptr), data_len);
                callback(r >= 0);
                if (r < 0) {
                    NIXL_ERROR << "RDMA PUT failed (accelerated enabled); no HTTP fallback";
                }
            });
            return;
        }
#endif
        NIXL_ERROR << "accelerated=true but RDMA fast path unavailable; failing PUT";
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "PutObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto data_stream =
        Aws::MakeShared<Aws::IOStream>("PutObjectInputStream", preallocated_stream_buf.get());
    request.SetBody(data_stream);

    s3Client_->PutObjectAsync(
        request,
        [callback, preallocated_stream_buf, data_stream](
            const Aws::S3::S3Client *,
            const Aws::S3::Model::PutObjectRequest &,
            const Aws::S3::Model::PutObjectOutcome &outcome,
            const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            callback(outcome.IsSuccess());
        },
        nullptr);
}

void
awsS3Client::getObjectAsync(std::string_view key,
                            uintptr_t data_ptr,
                            size_t data_len,
                            size_t offset,
                            get_object_callback_t callback) {
    // See putObjectAsync: under accelerated=true (generic S3-over-RDMA), GET is
    // RDMA-only with no silent HTTP fallback; fail if the fast path is not ready.
    if (rdma_requested_) {
#ifdef HAVE_CUOBJ_CLIENT
        if (rdmaReady()) {
            executor_->Submit([this, k = std::string(key), data_ptr, data_len, offset, callback]() {
                nixl_obj_rdma::S3RdmaClientCtx ctx;
                ctx.bucket = bucketName_.c_str();
                ctx.object = k;
                const ssize_t r = nixl_obj_rdma::rdmaGetWithRetry(
                    *rdma_, *rdmaCp_, ctx, reinterpret_cast<void *>(data_ptr), data_len, offset);
                callback(r >= 0);
                if (r < 0) {
                    NIXL_ERROR << "RDMA GET failed (accelerated enabled); no HTTP fallback";
                }
            });
            return;
        }
#endif
        NIXL_ERROR << "accelerated=true but RDMA fast path unavailable; failing GET";
        callback(false);
        return;
    }

    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "GetObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto stream_factory = Aws::MakeShared<Aws::IOStreamFactory>(
        "GetObjectStreamFactory", [preallocated_stream_buf]() -> Aws::IOStream * {
            return new Aws::IOStream(preallocated_stream_buf.get());
        });

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    request.SetResponseStreamFactory(*stream_factory.get());

    s3Client_->GetObjectAsync(
        request,
        [callback, stream_factory](const Aws::S3::S3Client *,
                                   const Aws::S3::Model::GetObjectRequest &,
                                   const Aws::S3::Model::GetObjectOutcome &outcome,
                                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            callback(outcome.IsSuccess());
        },
        nullptr);
}

void
awsS3Client::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    Aws::S3::Model::HeadObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    s3Client_->HeadObjectAsync(
        request,
        [callback](const Aws::S3::S3Client *,
                   const Aws::S3::Model::HeadObjectRequest &,
                   const Aws::S3::Model::HeadObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (outcome.IsSuccess()) {
                callback(true);
            } else {
                auto error_type = outcome.GetError().GetErrorType();
                // HeadObject returns HTTP 404 with no body for missing objects,
                // so the SDK cannot parse "NoSuchKey" from XML and instead maps
                // the 404 to RESOURCE_NOT_FOUND.  Accept both that and the
                // explicit NO_SUCH_KEY (which some S3-compatible stores may
                // return via headers or enriched error responses).
                if (error_type == Aws::S3::S3Errors::NO_SUCH_KEY ||
                    error_type == Aws::S3::S3Errors::RESOURCE_NOT_FOUND) {
                    callback(false);
                } else if (error_type == Aws::S3::S3Errors::NO_SUCH_BUCKET) {
                    NIXL_ERROR << "checkObjectExistsAsync bucket/endpoint error: "
                               << outcome.GetError().GetMessage();
                    callback(std::nullopt);
                } else {
                    NIXL_ERROR << "checkObjectExistsAsync error: "
                               << outcome.GetError().GetMessage();
                    callback(std::nullopt);
                }
            }
        },
        nullptr);
}
