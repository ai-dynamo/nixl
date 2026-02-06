/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client.h"
#include "common/nixl_log.h"

awsS3AccelClient::awsS3AccelClient(nixl_b_params_t *custom_params,
                                   std::shared_ptr<Aws::Utils::Threading::Executor> executor)
    : awsS3Client(custom_params, executor) {
    // Base class already initializes the S3 client, bucket name, and credentials.
    // Derived vendor clients can add their specific initialization here.
    NIXL_DEBUG << "S3 Accelerated client initialized";
}

void
awsS3AccelClient::putObjectRdmaAsync(std::string_view key,
                                     uintptr_t data_ptr,
                                     size_t data_len,
                                     size_t offset,
                                     const std::string &rdma_desc,
                                     put_object_callback_t callback) {
    throw std::runtime_error("RDMA operations not supported by accelerated S3 client");
}

void
awsS3AccelClient::getObjectRdmaAsync(std::string_view key,
                                     uintptr_t data_ptr,
                                     size_t data_len,
                                     size_t offset,
                                     const std::string &rdma_desc,
                                     get_object_callback_t callback) {
    throw std::runtime_error("RDMA operations not supported by accelerated S3 client");
}
