/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <climits>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

awsS3DellObsClient::awsS3DellObsClient(
        nixl_b_params_t *custom_params,
        std::shared_ptr<CuObjTokenManager> token_mgr,
        std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3AccelClient(custom_params, executor),
      tokenMgr_(std::move(token_mgr)) {
    NIXL_DEBUG << "Initialized Dell ObjectScale client (Pattern B)";
}

// ---------------------------------------------------------------------------
// Per-request fallback helper
// ---------------------------------------------------------------------------

/**
 * Check whether a data pointer refers to CUDA device memory.
 * Used to gate the HTTP fallback: PreallocatedStreamBuf requires
 * CPU-addressable memory, so VRAM pointers must not use the fallback.
 */
static bool
isCudaDeviceMemory(uintptr_t data_ptr) {
    auto mem_type = cuObjClient::getMemoryType(
        reinterpret_cast<const void *>(data_ptr));
    return mem_type == CUOBJ_MEMORY_CUDA_DEVICE;
}

// ---------------------------------------------------------------------------
// PUT
// ---------------------------------------------------------------------------

void
awsS3DellObsClient::putObjectAsync(std::string_view key,
                                    uintptr_t data_ptr,
                                    size_t data_len,
                                    size_t offset,
                                    put_object_callback_t callback) {
    // Dell ObjectScale PUT does not support partial writes with offsets.
    if (data_len == 0) {
        NIXL_ERROR << "putObjectAsync: data_len is 0";
        callback(false);
        return;
    }
    if (offset != 0) {
        NIXL_ERROR << "putObjectAsync: offset must be 0 for Dell ObjectScale PUT";
        callback(false);
        return;
    }

    try {
        // Pattern B: generate RDMA token for the data region.
        std::string token = tokenMgr_->generatePutToken(
            reinterpret_cast<void *>(data_ptr), data_len, 0);

        // Build a standard PutObject request with the RDMA token header.
        // ContentLength is 0 because the server reads data via RDMA_READ,
        // not from the HTTP body.
        Aws::S3::Model::PutObjectRequest request;
        request.WithBucket(bucketName_).WithKey(Aws::String(key));
        request.SetAdditionalCustomHeaderValue("x-rdma-info", token);
        request.SetContentLength(0);

        NIXL_DEBUG << absl::StrFormat(
            "putObjectAsync RDMA: key=%s, data_ptr=%p, data_len=%zu",
            std::string(key).c_str(),
            reinterpret_cast<void *>(data_ptr), data_len);

        s3Client_->PutObjectAsync(
            request,
            [callback](const Aws::S3::S3Client *,
                       const Aws::S3::Model::PutObjectRequest &,
                       const Aws::S3::Model::PutObjectOutcome &outcome,
                       const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
                if (!outcome.IsSuccess()) {
                    const auto &error = outcome.GetError();
                    NIXL_ERROR << absl::StrFormat(
                        "putObjectAsync RDMA failed: %s: %s (HTTP %d)",
                        error.GetExceptionName().c_str(),
                        error.GetMessage().c_str(),
                        static_cast<int>(error.GetResponseCode()));
                }
                callback(outcome.IsSuccess());
            },
            nullptr);
    } catch (const std::exception &e) {
        // RDMA token generation failed.  Attempt per-request fallback to
        // standard HTTP body transfer — but only for CPU-addressable memory.
        if (isCudaDeviceMemory(data_ptr)) {
            NIXL_ERROR << "RDMA put token failed for VRAM, no HTTP fallback: "
                       << e.what();
            callback(false);
            return;
        }
        NIXL_WARN << "RDMA put token failed, falling back to HTTP body: "
                  << e.what();
        awsS3AccelClient::putObjectAsync(key, data_ptr, data_len, offset, callback);
    }
}

// ---------------------------------------------------------------------------
// GET
// ---------------------------------------------------------------------------

void
awsS3DellObsClient::getObjectAsync(std::string_view key,
                                    uintptr_t data_ptr,
                                    size_t data_len,
                                    size_t offset,
                                    get_object_callback_t callback) {
    if (data_len == 0) {
        NIXL_ERROR << "getObjectAsync: data_len is 0";
        callback(false);
        return;
    }
    // Guard against offset + data_len integer overflow.
    if (offset > (SIZE_MAX - (data_len - 1))) {
        NIXL_ERROR << "getObjectAsync: offset + data_len would overflow";
        callback(false);
        return;
    }

    try {
        // Pattern B: generate RDMA token for the receive buffer.
        std::string token = tokenMgr_->generateGetToken(
            reinterpret_cast<void *>(data_ptr), data_len, offset);

        // Build a GetObject request with RDMA token and byte range.
        // The server sends no HTTP body — it writes directly into the
        // client's registered memory via RDMA_WRITE.
        Aws::S3::Model::GetObjectRequest request;
        request.WithBucket(bucketName_)
               .WithKey(Aws::String(key))
               .WithRange(absl::StrFormat("bytes=%zu-%zu",
                                          offset, offset + data_len - 1));
        request.SetAdditionalCustomHeaderValue("x-rdma-info", token);

        NIXL_DEBUG << absl::StrFormat(
            "getObjectAsync RDMA: key=%s, data_ptr=%p, data_len=%zu, offset=%zu",
            std::string(key).c_str(),
            reinterpret_cast<void *>(data_ptr), data_len, offset);

        s3Client_->GetObjectAsync(
            request,
            [callback](const Aws::S3::S3Client *,
                       const Aws::S3::Model::GetObjectRequest &,
                       const Aws::S3::Model::GetObjectOutcome &outcome,
                       const std::shared_ptr<const Aws::Client::AsyncCallerContext> &) {
                if (!outcome.IsSuccess()) {
                    const auto &error = outcome.GetError();
                    NIXL_ERROR << absl::StrFormat(
                        "getObjectAsync RDMA failed: %s: %s (HTTP %d)",
                        error.GetExceptionName().c_str(),
                        error.GetMessage().c_str(),
                        static_cast<int>(error.GetResponseCode()));
                }
                callback(outcome.IsSuccess());
            },
            nullptr);
    } catch (const std::exception &e) {
        // Per-request fallback: DRAM → HTTP body; VRAM → hard fail.
        if (isCudaDeviceMemory(data_ptr)) {
            NIXL_ERROR << "RDMA get token failed for VRAM, no HTTP fallback: "
                       << e.what();
            callback(false);
            return;
        }
        NIXL_WARN << "RDMA get token failed, falling back to HTTP body: "
                  << e.what();
        awsS3AccelClient::getObjectAsync(key, data_ptr, data_len, offset, callback);
    }
}
