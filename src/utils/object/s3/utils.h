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

#ifndef NIXL_SRC_UTILS_OBJECT_S3_UTILS_H
#define NIXL_SRC_UTILS_OBJECT_S3_UTILS_H

#include <optional>
#include <string>
#include <cstdlib>
#include <aws/core/http/Scheme.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include "nixl_types.h"
#include "common/backend.h"

namespace nixl_s3_utils {

/**
 * Create AWS credentials from custom parameters.
 * Returns nullopt if access_key or secret_key are not provided.
 */
std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials(nixl_b_params_t *custom_params);

/**
 * Get use_virtual_addressing setting from custom parameters.
 * Defaults to false if not specified.
 */
bool
getUseVirtualAddressing(nixl_b_params_t *custom_params);

/**
 * Get bucket name from custom parameters or AWS_DEFAULT_BUCKET env var.
 * Throws runtime_error if bucket cannot be determined.
 */
std::string
getBucketName(nixl_b_params_t *custom_params);

/**
 * Template function to configure common client settings.
 * Works with both Aws::Client::ClientConfiguration and Aws::S3Crt::ClientConfiguration
 */
template<typename ConfigType>
void
configureClientCommon(ConfigType &config, nixl_b_params_t *custom_params) {
    if (const auto opt =
            nixl::getBackendParamOptional<std::string>(custom_params, "endpoint_override")) {
        config.endpointOverride = *opt;
    }

    if (const auto opt = nixl::getBackendParamOptional<std::string>(custom_params, "scheme")) {
        if (*opt == "http") {
            config.scheme = Aws::Http::Scheme::HTTP;
        } else if (*opt == "https") {
            config.scheme = Aws::Http::Scheme::HTTPS;
        } else {
            throw std::runtime_error("Invalid scheme: " + *opt);
        }
    }

    if (const auto opt = nixl::getBackendParamOptional<std::string>(custom_params, "region")) {
        config.region = *opt;
    }

    if (const auto opt =
            nixl::getBackendParamOptional<std::string>(custom_params, "req_checksum")) {
        if (*opt == "required") {
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED;
        } else if (*opt == "supported") {
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_SUPPORTED;
        } else {
            throw std::runtime_error("Invalid value for req_checksum: '" + *opt +
                                     "'. Must be 'required' or 'supported'");
        }
    }

    if (const auto opt = nixl::getBackendParamOptional<std::string>(custom_params, "ca_bundle")) {
        config.caFile = *opt;
    }
}

} // namespace nixl_s3_utils

#endif
