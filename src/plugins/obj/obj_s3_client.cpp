/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "obj_s3_client.h"
#include <optional>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectResult.h>
#include <aws/s3/model/GetObjectResult.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/HeadObjectResult.h>
#include <aws/core/http/Scheme.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <absl/strings/str_format.h>
#include "nixl_types.h"

namespace {

Aws::Client::ClientConfiguration
createClientConfiguration(nixl_b_params_t *custom_params) {
    Aws::Client::ClientConfiguration config;

    if (!custom_params) return config;

    auto endpoint_override_it = custom_params->find("endpoint_override");
    if (endpoint_override_it != custom_params->end())
        config.endpointOverride = endpoint_override_it->second;

    auto scheme_it = custom_params->find("scheme");
    if (scheme_it != custom_params->end()) {
        if (scheme_it->second == "http")
            config.scheme = Aws::Http::Scheme::HTTP;
        else if (scheme_it->second == "https")
            config.scheme = Aws::Http::Scheme::HTTPS;
        else
            throw std::runtime_error("Invalid scheme: " + scheme_it->second);
    }

    auto region_it = custom_params->find("region");
    if (region_it != custom_params->end()) config.region = region_it->second;

    auto req_checksum_it = custom_params->find("req_checksum");
    if (req_checksum_it != custom_params->end()) {
        if (req_checksum_it->second == "required")
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED;
        else if (req_checksum_it->second == "supported")
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_SUPPORTED;
        else
            throw std::runtime_error("Invalid value for req_checksum: '" + req_checksum_it->second +
                                     "'. Must be 'required' or 'supported'");
    }

    return config;
}

std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials(nixl_b_params_t *custom_params) {
    if (!custom_params) return std::nullopt;

    std::string access_key, secret_key, session_token;

    auto access_key_it = custom_params->find("access_key");
    if (access_key_it != custom_params->end()) access_key = access_key_it->second;

    auto secret_key_it = custom_params->find("secret_key");
    if (secret_key_it != custom_params->end()) secret_key = secret_key_it->second;

    auto session_token_it = custom_params->find("session_token");
    if (session_token_it != custom_params->end()) session_token = session_token_it->second;

    if (access_key.empty() || secret_key.empty()) return std::nullopt;

    if (session_token.empty()) return Aws::Auth::AWSCredentials(access_key, secret_key);

    return Aws::Auth::AWSCredentials(access_key, secret_key, session_token);
}

bool
getUseVirtualAddressing(nixl_b_params_t *custom_params) {
    if (!custom_params) return false;

    auto virtual_addressing_it = custom_params->find("use_virtual_addressing");
    if (virtual_addressing_it != custom_params->end()) {
        const std::string &value = virtual_addressing_it->second;
        if (value == "true")
            return true;
        else if (value == "false")
            return false;
        else
            throw std::runtime_error("Invalid value for use_virtual_addressing: '" + value +
                                     "'. Must be 'true' or 'false'");
    }

    return false;
}

std::string
getBucketName(nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto bucket_it = custom_params->find("bucket");
        if (bucket_it != custom_params->end() && !bucket_it->second.empty()) {
            return bucket_it->second;
        }
    }

    const char *env_bucket = std::getenv("AWS_DEFAULT_BUCKET");
    if (env_bucket && env_bucket[0] != '\0') return std::string(env_bucket);
    throw std::runtime_error("Bucket name not found. Please provide 'bucket' in custom_params or "
                             "set AWS_DEFAULT_BUCKET environment variable");
}

} // namespace

awsS3Client::awsS3Client(nixl_b_params_t *custom_params,
                         std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsOptions_(
          []() {
              auto *opts = new Aws::SDKOptions();
              Aws::InitAPI(*opts);
              return opts;
          }(),
          [](Aws::SDKOptions *opts) {
              Aws::ShutdownAPI(*opts);
              delete opts;
          }) {
    auto config = ::createClientConfiguration(custom_params);
    if (executor) config.executor = executor;

    auto credentials_opt = ::createAWSCredentials(custom_params);
    bool use_virtual_addressing = ::getUseVirtualAddressing(custom_params);
    bucketName_ = Aws::String(::getBucketName(custom_params));

    if (credentials_opt.has_value())
        s3Client_ = std::make_unique<Aws::S3::S3Client>(
            credentials_opt.value(),
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
    else
        s3Client_ = std::make_unique<Aws::S3::S3Client>(
            config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::RequestDependent,
            use_virtual_addressing);
}

void
awsS3Client::setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    throw std::runtime_error("AwsS3Client::setExecutor() not supported - AWS SDK doesn't allow "
                             "changing executor after client creation");
}

void
awsS3Client::putObjectAsync(std::string_view key,
                            uintptr_t data_ptr,
                            size_t data_len,
                            size_t offset,
                            put_object_callback_t callback) {
    // AWS S3 doesn't support partial put operations with offset
    if (offset != 0) {
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
            const Aws::S3::S3Client *client,
            const Aws::S3::Model::PutObjectRequest &req,
            const Aws::S3::Model::PutObjectOutcome &outcome,
            const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
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
    auto preallocated_stream_buf = Aws::MakeShared<Aws::Utils::Stream::PreallocatedStreamBuf>(
        "GetObjectStreamBuf", reinterpret_cast<unsigned char *>(data_ptr), data_len);
    auto stream_factory = Aws::MakeShared<Aws::IOStreamFactory>(
        "GetObjectStreamFactory", [preallocated_stream_buf]() -> Aws::IOStream * {
            return new Aws::IOStream(preallocated_stream_buf.get()); // AWS SDK owns the stream
        });

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    request.SetResponseStreamFactory(*stream_factory.get());

    s3Client_->GetObjectAsync(
        request,
        [callback,
         stream_factory](const Aws::S3::S3Client *client,
                         const Aws::S3::Model::GetObjectRequest &req,
                         const Aws::S3::Model::GetObjectOutcome &outcome,
                         const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
            callback(outcome.IsSuccess());
        },
        nullptr);
}

bool
awsS3Client::checkObjectExists(std::string_view key) {
    Aws::S3::Model::HeadObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    auto outcome = s3Client_->HeadObject(request);
    if (outcome.IsSuccess())
        return true;
    else if (outcome.GetError().GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND)
        return false;
    else
        throw std::runtime_error("Failed to check if object exists: " +
                                 outcome.GetError().GetMessage());
}
