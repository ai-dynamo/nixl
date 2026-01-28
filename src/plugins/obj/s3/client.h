/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef OBJ_PLUGIN_S3_CLIENT_H
#define OBJ_PLUGIN_S3_CLIENT_H

#include <functional>
#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/Aws.h>
#include "nixl_types.h"
#include "obj_backend.h"

/**
 * Concrete implementation of IS3Client using AWS SDK S3Client.
 */
class awsS3Client : public iS3Client {
public:
    /**
     * Constructor that creates an AWS S3Client from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3Client(nixl_b_params_t *custom_params,
                std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

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
    std::unique_ptr<Aws::S3::S3Client> s3Client_;
    Aws::String bucketName_;
};

#endif // OBJ_PLUGIN_S3_CLIENT_H
