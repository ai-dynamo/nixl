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

#ifndef OBJ_PLUGIN_DELL_OBS_CLIENT_H
#define OBJ_PLUGIN_DELL_OBS_CLIENT_H

#include "../client.h"
#include "nixl_types.h"

/**
 * Dell OBS Object Client - Inherits from S3 Accel for Dell Object Storage.
 * This client is specifically configured for Dell's ObjectScale (OBS) platform,
 * leveraging the accelerated S3 CRT implementation with Dell-specific optimizations.
 */
class awsDellOBSClient : public awsS3AccelClient {
public:
    /**
     * Constructor that creates a Dell OBS client from custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsDellOBSClient(nixl_b_params_t *custom_params,
                     std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsDellOBSClient() = default;

    // Inherits all methods from awsS3AccelClient
    // Can override methods here for Dell-specific optimizations if needed
};

#endif // OBJ_PLUGIN_DELL_OBS_CLIENT_H
