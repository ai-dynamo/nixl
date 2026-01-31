/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_ACCEL_CLIENT_H
#define OBJ_PLUGIN_S3_ACCEL_CLIENT_H

#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include "s3/client.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client - Inherits from S3 Vanilla and uses CRT for accelerated transfers.
 * This client uses the same high-performance CRT implementation as S3CrtClient but represents
 * a separate acceleration path from the standard CRT client. This is the base class for
 * vendor-specific accelerated implementations.
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

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override;

    void
    putObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   put_object_callback_t callback) override;

    void
    getObjectAsync(std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override;

    bool
    checkObjectExists(std::string_view key) override;

    // S3 client from base S3
};

#endif // OBJ_PLUGIN_S3_ACCEL_CLIENT_H
