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

#include "utils.h"

namespace nixl_s3_utils {

std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials(nixl_b_params_t *custom_params) {
    if (!custom_params) return std::nullopt;

    std::string access_key, secret_key, session_token;

    auto access_key_it = custom_params->find("access_key");
    if (access_key_it != custom_params->end()) access_key = access_key_it->second;

    auto secret_key_it = custom_params->find("secret_key");
    if (secret_key_it != custom_params->end()) secret_key = secret_key_it->second;

    auto session_token_it = custom_params->find("session_token");
    if (session_token_it != custom_params->end()) session_token = session_token_it->second;

    if (access_key.empty() || secret_key.empty()) return std::nullopt;

    if (session_token.empty()) return Aws::Auth::AWSCredentials(access_key, secret_key);

    return Aws::Auth::AWSCredentials(access_key, secret_key, session_token);
}

bool
getUseVirtualAddressing(nixl_b_params_t *custom_params) {
    if (!custom_params) return false;

    auto virtual_addressing_it = custom_params->find("use_virtual_addressing");
    if (virtual_addressing_it != custom_params->end()) {
        const std::string &value = virtual_addressing_it->second;
        if (value == "true")
            return true;
        else if (value == "false")
            return false;
        else
            throw std::runtime_error("Invalid value for use_virtual_addressing: '" + value +
                                     "'. Must be 'true' or 'false'");
    }

    return false;
}

std::string
getBucketName(nixl_b_params_t *custom_params) {
    if (custom_params) {
        auto bucket_it = custom_params->find("bucket");
        if (bucket_it != custom_params->end() && !bucket_it->second.empty()) {
            return bucket_it->second;
        }
    }

    const char *env_bucket = std::getenv("AWS_DEFAULT_BUCKET");
    if (env_bucket && env_bucket[0] != '\0') return std::string(env_bucket);

    throw std::runtime_error("Bucket name not found. Please provide 'bucket' in custom_params or "
                             "set AWS_DEFAULT_BUCKET environment variable");
}

} // namespace nixl_s3_utils
