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

#include "common/backend.h"
#include "common/configuration.h"
#include "utils.h"

namespace nixl_s3_utils {

[[nodiscard]] std::optional<Aws::Auth::AWSCredentials>
createAWSCredentials(nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return std::nullopt;
    }

    const std::string access_key = nixl::getBackendParamDefaulted(custom_params, "access_key", std::string());
    const std::string secret_key = nixl::getBackendParamDefaulted(custom_params, "secret_key", std::string());
    const std::string session_token = nixl::getBackendParamDefaulted(custom_params, "session_token", std::string());

    if (access_key.empty() || secret_key.empty()) {
        return std::nullopt;
    }

    if (session_token.empty()) {
        return Aws::Auth::AWSCredentials(access_key, secret_key);
    }

    return Aws::Auth::AWSCredentials(access_key, secret_key, session_token);
}

[[nodiscard]] bool
getUseVirtualAddressing(nixl_b_params_t *custom_params) {
    return nixl::getBackendParamDefaulted(custom_params, "use_virtual_addressing", false);
}

[[nodiscard]] std::string
getBucketName(nixl_b_params_t *custom_params) {
    const auto str = nixl::getBackendParamDefaulted(custom_params, "bucket", std::string());
    if (!str.empty()) {
        return str;
    }

    return nixl::config::getNonEmptyString("AWS_DEFAULT_BUCKET");
}

} // namespace nixl_s3_utils
