/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H

#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include "s3_accel/client.h"
#include "rdma_interface.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client for use with Dell Objectscale - Inherits from Accelerated S3 Client.
 * This client presents Put and GetObject interfaces to enable RDMA for S3-compatible storage using
 * the cuObject API
 */
class awsS3DellObsClient : public awsS3AccelClient, public iS3RdmaClient {
public:
    /**
     * Constructor that creates an AWS S3 client for use with Dell ObjectScale from custom
     * parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3DellObsClient(nixl_b_params_t *custom_params,
                       std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3DellObsClient() = default;

    /**
     * Asynchronously puts an object to S3 using RDMA acceleration.
     *
     * @param key The object key to store
     * @param data_ptr Pointer to the data buffer
     * @param data_len Length of the data to transfer
     * @param offset Offset within the object (must be 0 for Dell ObjectScale)
     * @param rdma_desc RDMA descriptor for acceleration
     * @param callback Callback function invoked on completion
     */
    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback);

    /**
     * Asynchronously gets an object from S3 using RDMA acceleration.
     *
     * @param key The object key to retrieve
     * @param data_ptr Pointer to the data buffer to fill
     * @param data_len Length of the data to transfer
     * @param offset Offset within the object
     * @param rdma_desc RDMA descriptor for acceleration
     * @param callback Callback function invoked on completion
     */
    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback);
};

#endif // NIXL_OBJ_PLUGIN_S3_DELL_CLIENT_H
