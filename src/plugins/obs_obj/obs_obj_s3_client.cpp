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

#include "obs_obj_s3_client.h"
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

void
obsObjS3Client::putObjectRdmaAsync(std::string_view key,
                                  uintptr_t data_ptr,
                                  size_t data_len,
                                  size_t offset,
                                  const std::string &rdma_desc,
                                  put_object_callback_t callback) {
    // AWS S3 doesn't support partial put operations with offset
    if (offset != 0) {
        callback(false);
        return;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucketName_).WithKey(Aws::String(key));
    if (!rdma_desc.empty()) {
        request.SetAdditionalCustomHeaderValue("x-rdma-info", rdma_desc);
    }

    request.SetContentLength(0);

    s3Client_->PutObjectAsync(
        request,
        [callback](
            const Aws::S3::S3Client *client,
            const Aws::S3::Model::PutObjectRequest &req,
            const Aws::S3::Model::PutObjectOutcome &outcome,
            const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {

            callback(outcome.IsSuccess());
        },
        nullptr);

}

void
obsObjS3Client::getObjectRdmaAsync(std::string_view key,
                                  uintptr_t data_ptr,
                                  size_t data_len,
                                  size_t offset,
                                  const std::string &rdma_desc,
                                  get_object_callback_t callback) {
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucketName_)
        .WithKey(Aws::String(key))
        .WithRange(absl::StrFormat("bytes=%d-%d", offset, offset + data_len - 1));
    if (!rdma_desc.empty()) {
        request.SetAdditionalCustomHeaderValue("x-rdma-info", rdma_desc);
    }

    s3Client_->GetObjectAsync(
        request,
        [callback]
         (const Aws::S3::S3Client *client,
                         const Aws::S3::Model::GetObjectRequest &req,
                         const Aws::S3::Model::GetObjectOutcome &outcome,
                         const std::shared_ptr<const Aws::Client::AsyncCallerContext> &context) {
            callback(outcome.IsSuccess());
        },
        nullptr);
}
