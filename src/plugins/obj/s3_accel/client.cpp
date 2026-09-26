/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "object/engine_utils.h"
#include "common/nixl_log.h"

awsS3AccelClient::awsS3AccelClient(nixl_b_params_t *custom_params,
                                   std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3Client(custom_params, executor),
      executor_(std::move(executor)),
      rdma_requested_(isGenericAccelRequested(custom_params)) {
    // Attach the S3-over-RDMA fast path only for the generic accel path. Vendor
    // subclasses (selected by an explicit `type`) leave rdma_requested_ false and
    // manage RDMA on their own, so this stays the plain HTTP base for them.
    if (rdma_requested_) {
        rdma_ = nixl_obj_rdma::SharedCuObjClient::instance();
        if (rdma_) {
            rdmaCp_ = std::make_shared<nixl_obj_rdma::S3RdmaControlPlane>(custom_params);
            if (!rdmaCp_->valid()) {
                rdmaCp_.reset();
                rdma_ = nullptr;
            }
        }
    }

    // Fail fast: a generic accel client whose fast path is not fully ready can
    // never recover (setExecutor is unsupported and rdma_/rdmaCp_ are set once
    // here), so surface it at construction instead of failing every transfer.
    if (rdma_requested_ && !rdmaReady()) {
        throw std::runtime_error("accelerated=true (generic S3-over-RDMA) requested but the "
                                 "fast path is unavailable (requires a reachable RDMA fabric, a "
                                 "valid control plane, and an executor); no HTTP fallback");
    }

    NIXL_DEBUG << "S3 Accelerated client initialized (rdma=" << (rdma_requested_ ? "on" : "off")
               << ")";
}

bool
awsS3AccelClient::rdmaReady() const {
    return rdma_ != nullptr && rdmaCp_ != nullptr && executor_ != nullptr;
}

bool
awsS3AccelClient::supportsRdma() const {
    return rdma_requested_ && rdmaReady();
}

void
awsS3AccelClient::setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    // No-op: the executor is fixed at construction (see ctor); the RDMA fast path
    // is bound to it and cannot be swapped afterward.
    (void)executor;
}

void
awsS3AccelClient::putObjectAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 size_t offset,
                                 put_object_callback_t callback) {
    if (!rdma_requested_) {
        awsS3Client::putObjectAsync(key, data_ptr, data_len, offset, callback);
        return;
    }

    // A single-shot RDMA PUT writes the whole object; there is no offset in the
    // PUT control plane, so a non-zero offset cannot be honored.
    if (offset != 0) {
        NIXL_ERROR << "S3 RDMA put: non-zero offset (" << offset << ") not supported, key=" << key;
        callback(false);
        return;
    }

    // Capture the transfer state by value (bucket, the process-wide cuObject
    // handle, and a shared control plane) rather than `this`, so the task is
    // self-contained and safe even if this client is destroyed while it runs.
    const bool submitted = executor_->Submit([rdma = rdma_,
                                              cp = rdmaCp_,
                                              bucket = std::string(bucketName_.c_str()),
                                              k = std::string(key),
                                              data_ptr,
                                              data_len,
                                              callback]() {
        nixl_obj_rdma::S3RdmaClientCtx ctx;
        ctx.bucket = bucket;
        ctx.object = k;
        const ssize_t r = nixl_obj_rdma::rdmaPutWithRetry(
            *rdma, *cp, ctx, reinterpret_cast<void *>(data_ptr), data_len);
        if (r < 0) {
            NIXL_ERROR << "S3 RDMA put failed (accelerated=true; no HTTP fallback), key=" << k;
        }
        callback(r >= 0);
    });
    // A rejected submission never runs the task, so fire the callback here or the
    // transfer's future would never complete.
    if (!submitted) {
        NIXL_ERROR << "S3 RDMA put: executor rejected task, key=" << key;
        callback(false);
    }
}

void
awsS3AccelClient::getObjectAsync(std::string_view key,
                                 uintptr_t data_ptr,
                                 size_t data_len,
                                 size_t offset,
                                 get_object_callback_t callback) {
    if (!rdma_requested_) {
        awsS3Client::getObjectAsync(key, data_ptr, data_len, offset, callback);
        return;
    }

    // See putObjectAsync: capture the transfer state by value so the task is
    // self-contained and safe even if this client is destroyed while it runs.
    const bool submitted = executor_->Submit([rdma = rdma_,
                                              cp = rdmaCp_,
                                              bucket = std::string(bucketName_.c_str()),
                                              k = std::string(key),
                                              data_ptr,
                                              data_len,
                                              offset,
                                              callback]() {
        nixl_obj_rdma::S3RdmaClientCtx ctx;
        ctx.bucket = bucket;
        ctx.object = k;
        const ssize_t r = nixl_obj_rdma::rdmaGetWithRetry(
            *rdma, *cp, ctx, reinterpret_cast<void *>(data_ptr), data_len, offset);
        if (r < 0) {
            NIXL_ERROR << "S3 RDMA get failed (accelerated=true; no HTTP fallback), key=" << k;
        }
        callback(r >= 0);
    });
    // A rejected submission never runs the task, so fire the callback here or the
    // transfer's future would never complete.
    if (!submitted) {
        NIXL_ERROR << "S3 RDMA get: executor rejected task, key=" << key;
        callback(false);
    }
}
