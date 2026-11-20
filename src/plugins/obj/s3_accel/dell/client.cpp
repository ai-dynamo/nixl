/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "object/rdma/cuobj_client.h"
#include "object/rdma/rdmaRetry.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <utility>

namespace {
// An SDK submission that throws after queuing the request must not complete the
// transfer a second time, which would throw from the promise behind the callback.
std::function<void(bool)>
completeOnce(std::function<void(bool)> callback) {
    return [done = std::make_shared<std::atomic_flag>(), callback = std::move(callback)](bool ok) {
        if (!done->test_and_set()) {
            callback(ok);
        }
    };
}

// cuObject mints a token for at most rdma_max_request_size bytes, even from a larger
// registration, and the Dell engine sends each descriptor as one request; splitting a
// larger object is left to the caller.
[[nodiscard]] bool
exceedsRequestLimit(std::string_view op, std::string_view key, size_t data_len) {
    if (data_len <= nixl_obj_rdma::rdma_max_request_size) {
        return false;
    }
    NIXL_ERROR << "Dell RDMA " << op << " of " << data_len
               << " bytes exceeds the cuObject per-request limit of "
               << nixl_obj_rdma::rdma_max_request_size << " bytes, key=" << key;
    return true;
}
} // namespace

awsS3DellObsClient::awsS3DellObsClient(nixl_b_params_t *custom_params,
                                       nixl_obj_rdma::SharedCuObjClient &rdma,
                                       std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, std::move(executor)),
      rdma_(rdma) {}

void
awsS3DellObsClient::putObjectAsync(std::string_view key,
                                   uintptr_t data_ptr,
                                   size_t data_len,
                                   size_t offset,
                                   put_object_callback_t callback) {
    if (!data_ptr || !data_len || offset != 0 || exceedsRequestLimit("PUT", key, data_len)) {
        callback(false);
        return;
    }
    auto token = std::shared_ptr<char>(
        rdma_.getToken(reinterpret_cast<void *>(data_ptr), data_len, CUOBJ_PUT),
        [rdma = &rdma_](char *p) { rdma->putToken(p); });
    if (!token || !*token) {
        callback(false);
        return;
    }
    callback = completeOnce(std::move(callback));
    try {
        Aws::S3::Model::PutObjectRequest request;
        request.WithBucket(bucketName_).WithKey(Aws::String(key));
        request.SetAdditionalCustomHeaderValue("x-rdma-info", token.get());
        request.SetContentLength(0);
        submitPut(request, [token, callback](bool success) { callback(success); });
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Dell RDMA PUT submission failed: " << e.what();
        callback(false);
    }
}

void
awsS3DellObsClient::getObjectAsync(std::string_view key,
                                   uintptr_t data_ptr,
                                   size_t data_len,
                                   size_t offset,
                                   get_object_callback_t callback) {
    if (!data_ptr || !data_len || offset > std::numeric_limits<size_t>::max() - (data_len - 1) ||
        exceedsRequestLimit("GET", key, data_len)) {
        callback(false);
        return;
    }
    auto token = std::shared_ptr<char>(
        rdma_.getToken(reinterpret_cast<void *>(data_ptr), data_len, CUOBJ_GET),
        [rdma = &rdma_](char *p) { rdma->putToken(p); });
    if (!token || !*token) {
        callback(false);
        return;
    }
    callback = completeOnce(std::move(callback));
    try {
        Aws::S3::Model::GetObjectRequest request;
        request.WithBucket(bucketName_)
            .WithKey(Aws::String(key))
            .WithRange(absl::StrFormat("bytes=%zu-%zu", offset, offset + data_len - 1));
        request.SetAdditionalCustomHeaderValue("x-rdma-info", token.get());
        submitGet(request, [token, callback](bool success) { callback(success); });
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Dell RDMA GET submission failed: " << e.what();
        callback(false);
    }
}

void
awsS3DellObsClient::submitPut(const Aws::S3::Model::PutObjectRequest &request,
                              put_object_callback_t callback) {
    s3Client_->PutObjectAsync(
        request,
        [callback](const Aws::S3::S3Client *,
                   const Aws::S3::Model::PutObjectRequest &,
                   const Aws::S3::Model::PutObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (!outcome.IsSuccess()) {
                const auto &error = outcome.GetError();
                NIXL_ERROR << "Dell RDMA request failed: " << error.GetExceptionName() << ": "
                           << error.GetMessage() << " (HTTP "
                           << static_cast<int>(error.GetResponseCode()) << ")";
            }
            callback(outcome.IsSuccess());
        });
}

void
awsS3DellObsClient::submitGet(const Aws::S3::Model::GetObjectRequest &request,
                              get_object_callback_t callback) {
    s3Client_->GetObjectAsync(
        request,
        [callback](const Aws::S3::S3Client *,
                   const Aws::S3::Model::GetObjectRequest &,
                   const Aws::S3::Model::GetObjectOutcome &outcome,
                   const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
            if (!outcome.IsSuccess()) {
                const auto &error = outcome.GetError();
                NIXL_ERROR << "Dell RDMA request failed: " << error.GetExceptionName() << ": "
                           << error.GetMessage() << " (HTTP "
                           << static_cast<int>(error.GetResponseCode()) << ")";
            }
            callback(outcome.IsSuccess());
        });
}
