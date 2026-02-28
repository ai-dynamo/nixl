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
#include "common/nixl_log.h"

/**
 * Implementation of awsS3DellObsClient constructor.
 * Initializes the Dell ObjectScale S3 client with custom parameters.
 *
 * @param custom_params Custom parameters containing S3 configuration
 * @param executor Optional executor for async operations
 */
awsS3DellObsClient::awsS3DellObsClient(nixl_b_params_t *custom_params,
                                       std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor) {
    NIXL_DEBUG << "Initialized Dell ObjectScale Client";
}

/**
 * Asynchronously puts an object to Dell ObjectScale S3 using RDMA acceleration.
 * This method validates parameters, sets up the S3 request with RDMA headers,
 * and initiates the asynchronous put operation.
 *
 * @param key The object key to store
 * @param data_ptr Pointer to the data buffer
 * @param data_len Length of the data to transfer
 * @param offset Offset within the object (must be 0 for Dell ObjectScale)
 * @param rdma_desc RDMA descriptor for acceleration
 * @param callback Callback function invoked on completion with success/failure status
 */
void
awsS3DellObsClient::putObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       std::string_view rdma_desc,
                                       put_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat(
        "putObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc=%s",
        std::string(key).c_str(),
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.empty() ? "<empty>" : std::string(rdma_desc).c_str());

    if (data_len == 0) {
        NIXL_ERROR << "putObjectRdmaAsync: data_len is 0, returning failure";
        callback(false);
        return;
    }

    if (offset != 0) {
        NIXL_ERROR << "putObjectRdmaAsync: offset is not 0, returning failure";
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));

    if (!rdma_desc.empty()) {
        request.SetAdditionalCustomHeaderValue("x-rdma-info", std::string(rdma_desc));
    } else {
        NIXL_ERROR << "putObjectRdmaAsync: rdma_desc is empty, returning failure";
        callback(false);
        return;
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
                NIXL_ERROR << absl::StrFormat(
                    "putObjectRdmaAsync: PutObjectAsync failed - %s: %s (HTTP %d)",
                    error.GetExceptionName().c_str(),
                    error.GetMessage().c_str(),
                    static_cast<int>(error.GetResponseCode()));
            }
            callback(outcome.IsSuccess());
        },
        nullptr);
}

/**
 * Asynchronously gets an object from Dell ObjectScale S3 using RDMA acceleration.
 * This method validates parameters, sets up the S3 request with RDMA headers and range,
 * and initiates the asynchronous get operation.
 *
 * @param key The object key to retrieve
 * @param data_ptr Pointer to the data buffer to fill
 * @param data_len Length of the data to transfer
 * @param offset Offset within the object
 * @param rdma_desc RDMA descriptor for acceleration
 * @param callback Callback function invoked on completion with success/failure status
 */
void
awsS3DellObsClient::getObjectRdmaAsync(std::string_view key,
                                       uintptr_t data_ptr,
                                       size_t data_len,
                                       size_t offset,
                                       std::string_view rdma_desc,
                                       get_object_callback_t callback) {

    NIXL_DEBUG << absl::StrFormat(
        "getObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc=%s",
        std::string(key).c_str(),
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.empty() ? "<empty>" : std::string(rdma_desc).c_str());

    if (data_len == 0) {
        NIXL_ERROR << "getObjectRdmaAsync: data_len is 0, returning failure";
        callback(false);
        return;
    }


    if ((data_len > 0) && (offset > (SIZE_MAX - (data_len - 1)))) {
        NIXL_ERROR << "getObjectRdmaAsync: offset + data_len would overflow, returning failure";
        callback(false);
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "getObjectRdmaAsync: rdma_desc is empty, returning failure";
        callback(false);
        return;
    }

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%zu-%zu", offset, offset + data_len - 1));
    request.SetAdditionalCustomHeaderValue("x-rdma-info", std::string(rdma_desc));

    NIXL_DEBUG << absl::StrFormat(
        "getObjectRdmaAsync: sending GetObjectAsync request with range bytes=%zu-%zu",
        offset,
        offset + data_len - 1);
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
                NIXL_ERROR << absl::StrFormat(
                    "getObjectRdmaAsync: GetObjectAsync failed - %s: %s (HTTP %d)",
                    error.GetExceptionName().c_str(),
                    error.GetMessage().c_str(),
                    static_cast<int>(error.GetResponseCode()));
            }
            callback(outcome.IsSuccess());
        },
        nullptr);
}
