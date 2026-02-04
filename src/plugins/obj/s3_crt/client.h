/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OBJ_PLUGIN_S3_CRT_CLIENT_H
#define OBJ_PLUGIN_S3_CRT_CLIENT_H

#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/Aws.h>
#include "s3/client.h"
#include "nixl_types.h"
#include "obj_backend.h"

/**
 * S3 CRT Object Client - Inherits from S3 Vanilla and uses AWS CRT for high-performance transfers.
 * The S3 CRT (Common Runtime) client uses AWS Common Runtime for improved performance with
 * large objects, providing better throughput and lower CPU utilization.
 * This client overrides the vanilla S3 client methods with CRT implementations.
 */
class awsS3CrtClient : public awsS3Client {
public:
    /**
     * Constructor that creates an AWS S3CrtClient from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3CrtClient(nixl_b_params_t *custom_params,
                   std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3CrtClient() = default;

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

private:
    std::unique_ptr<Aws::S3Crt::S3CrtClient> s3CrtClient_;
};

#endif // OBJ_PLUGIN_S3_CRT_CLIENT_H
