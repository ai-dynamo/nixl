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

#ifndef NIXL_S3_UTILS_H
#define NIXL_S3_UTILS_H

#include <optional>
#include <string>
#include <cstdlib>
#include <aws/core/http/Scheme.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include "nixl_types.h"

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
    if (!custom_params) return;

    auto endpoint_override_it = custom_params->find("endpoint_override");
    if (endpoint_override_it != custom_params->end())
        config.endpointOverride = endpoint_override_it->second;

    auto scheme_it = custom_params->find("scheme");
    if (scheme_it != custom_params->end()) {
        if (scheme_it->second == "http")
            config.scheme = Aws::Http::Scheme::HTTP;
        else if (scheme_it->second == "https")
            config.scheme = Aws::Http::Scheme::HTTPS;
        else
            throw std::runtime_error("Invalid scheme: " + scheme_it->second);
    }

    auto region_it = custom_params->find("region");
    if (region_it != custom_params->end()) config.region = region_it->second;

    auto req_checksum_it = custom_params->find("req_checksum");
    if (req_checksum_it != custom_params->end()) {
        if (req_checksum_it->second == "required")
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_REQUIRED;
        else if (req_checksum_it->second == "supported")
            config.checksumConfig.requestChecksumCalculation =
                Aws::Client::RequestChecksumCalculation::WHEN_SUPPORTED;
        else
            throw std::runtime_error("Invalid value for req_checksum: '" + req_checksum_it->second +
                                     "'. Must be 'required' or 'supported'");
    }

    auto ca_bundle_it = custom_params->find("ca_bundle");
    if (ca_bundle_it != custom_params->end()) config.caFile = ca_bundle_it->second;
}

} // namespace nixl_s3_utils

#endif // NIXL_S3_UTILS_H
