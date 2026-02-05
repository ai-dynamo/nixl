/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_DELL_CLIENT_H
#define OBJ_PLUGIN_S3_DELL_CLIENT_H

#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include "s3/client.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client for use with Dell Objectscale - Inherits from Vannila S3 client.
 * This client presents Put and GetObject interfaces to enable RDMA for S3-compatible storage using the
 * cuObject API
 */
class awsS3DellObsClient : public awsS3Client {
public:
    /**
     * Constructor that creates an AWS S3 client for use with Dell ObjectScale from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3DellObsClient(nixl_b_params_t *custom_params,
                     std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3DellObsClient() = default;

    void
    putObjectRdmaAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   const std::string &rdma_desc,
                   put_object_callback_t callback);

    void
    getObjectRdmaAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   const std::string &rdma_desc,
                   get_object_callback_t callback);

    // S3 client from base S3
};

#endif // OBJ_PLUGIN_S3_DELL_CLIENT_H
