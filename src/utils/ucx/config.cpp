/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "config.h"

#include <stdexcept>

#include <absl/strings/str_format.h>

#include "common/nixl_log.h"

namespace nixl::ucx {
void
config::modify (std::string_view key, std::string_view value) const {
    const char *env_val = std::getenv (absl::StrFormat ("UCX_%s", key.data()).c_str());
    if (env_val) {
        NIXL_DEBUG << "UCX env var has already been set: " << key << "=" << env_val;
    } else {
        modifyAlways (key, value);
    }
}

void
config::modifyAlways (std::string_view key, std::string_view value) const {
    const auto status = ucp_config_modify (config_.get(), key.data(), value.data());
    if (status != UCS_OK) {
        NIXL_DEBUG << "Failed to modify UCX config: " << key << "=" << value << ": "
                   << ucs_status_string(status);
    } else {
        NIXL_DEBUG << "Modified UCX config: " << key << "=" << value;
    }
}

ucp_config_t *
config::readUcpConfig() {
    ucp_config_t *config = nullptr;
    const auto status = ucp_config_read (NULL, NULL, &config);
    if (status != UCS_OK) {
        const auto err_str =
            std::string ("Failed to create UCX config: ") + ucs_status_string (status);
        NIXL_ERROR << err_str;
        throw std::runtime_error (err_str);
    }
    return config;
}
} // namespace nixl::ucx
