/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ACCEL_CLIENT_H
#define OBJ_PLUGIN_S3_ACCEL_CLIENT_H

#include "s3/client.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client - Inherits from S3 Vanilla client.
 *
 * This client serves as the base class for vendor-specific accelerated implementations.
 * It inherits all functionality from awsS3Client and can be extended by vendor clients
 * to provide custom S3-compatible storage behavior (e.g., GPU-direct transfers).
 *
 * Vendor implementations should inherit from this class and override specific methods
 * as needed for their storage systems.
 */
class awsS3AccelClient : public awsS3Client {
public:
    /**
     * Constructor that creates an AWS S3 Accelerated client from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3AccelClient(nixl_b_params_t *custom_params,
                     std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3AccelClient() = default;

    // Inherits all methods from awsS3Client:
    // - setExecutor()
    // - putObjectAsync()
    // - getObjectAsync()
    // - checkObjectExists()
    //
    // Vendor clients can override these methods for custom behavior.
};

#endif // OBJ_PLUGIN_S3_ACCEL_CLIENT_H
