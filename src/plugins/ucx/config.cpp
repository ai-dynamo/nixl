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
#include "config.h"

#include <cctype>
#include <stdexcept>
#include <string>

#include "common/configuration.h"
#include "common/hw_info.h"
#include "common/nixl_log.h"
#include "ucx_utils.h"

namespace nixl::ucx {
namespace {

bool
hasInvalidDenyListSyntax(std::string_view tls) {
    bool can_start_deny_list = true;

    for (const char c : tls) {
        if (std::isspace(static_cast<unsigned char>(c)) && can_start_deny_list) {
            continue;
        }

        if (c == '^') {
            if (!can_start_deny_list) {
                return true;
            }

            can_start_deny_list = false;
            continue;
        }

        can_start_deny_list = false;
    }

    return false;
}

bool
isDenyList(std::string_view tls) {
    for (const char c : tls) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }

        return c == '^';
    }

    return false;
}

bool
isCudaSupportToken(std::string_view token) {
    return (token == "cuda") || (token == "cuda_copy");
}

bool
hasCudaSupportToken(std::string_view tls) {
    size_t pos = isDenyList(tls) ? tls.find('^') + 1 : 0;

    while (pos < tls.size()) {
        size_t end = tls.find(',', pos);
        if (end == std::string_view::npos) {
            end = tls.size();
        }

        while ((pos < end) && std::isspace(static_cast<unsigned char>(tls[pos]))) {
            ++pos;
        }

        while ((end > pos) && std::isspace(static_cast<unsigned char>(tls[end - 1]))) {
            --end;
        }

        if (isCudaSupportToken(tls.substr(pos, end - pos))) {
            return true;
        }

        pos = end + 1;
    }

    return false;
}

bool
tlsEnablesCudaSupport(std::string_view tls) {
    const bool has_cuda_support = hasCudaSupportToken(tls);
    return (!isDenyList(tls) && has_cuda_support) || (isDenyList(tls) && !has_cuda_support);
}

} // namespace

void
config::modify(std::string_view key, std::string_view value) const {
    const auto ucx_key = "UCX_" + std::string(key);
    const auto env_val = nixl::config::internal::getenvOptional(ucx_key);

    if (env_val) {
        NIXL_DEBUG << "UCX env var already set: " << ucx_key << "=" << *env_val;
    } else {
        modifyAlways(key, value);
    }
}

void
config::modifyAlways(std::string_view key, std::string_view value) const {
    std::string key_str(key);
    std::string value_str(value);
    const auto status = ucp_config_modify(config_.get(), key_str.c_str(), value_str.c_str());
    if (status != UCS_OK) {
        NIXL_DEBUG << "Failed to modify UCX config: " << key_str << "=" << value_str << ": "
                   << ucs_status_string(status);
    } else {
        NIXL_DEBUG << "Modified UCX config: " << key_str << "=" << value_str;
    }
}

void
config::validateTlsEnvironment() const {
    const auto tls = nixl::config::getValueOptional<std::string>("UCX_TLS");
    if (!tls) {
        return;
    }

    if (hasInvalidDenyListSyntax(*tls)) {
        const std::string error =
            "Invalid UCX_TLS=" + *tls +
            " for NIXL UCX backend: '^' may only appear as the first "
            "non-space character of UCX_TLS.";
        NIXL_ERROR << error;
        throw std::runtime_error(error);
    }
}

void
config::validateTlsCudaSupport(ucp_context_h ctx) const {
    const auto tls = nixl::config::getValueOptional<std::string>("UCX_TLS");
    if (!tls || tlsEnablesCudaSupport(*tls) || nixl::hwInfo::instance().numNvidiaGpus == 0) {
        return;
    }

    const auto supports_cuda = nixlUcpContextSupportsMemoryType(ctx, UCS_MEMORY_TYPE_CUDA);
    if (!supports_cuda || *supports_cuda) {
        return;
    }

    const std::string error =
        "Invalid UCX_TLS=" + *tls +
        " for NIXL UCX backend: NVIDIA GPU(s) are present, "
        "but this setting does not enable CUDA memory support. Add cuda_copy for "
        "basic GPU support, or cuda to also include NVLink support.";
    NIXL_ERROR << error;
    throw std::runtime_error(error);
}

ucp_config_t *
config::readUcpConfig() {
    ucp_config_t *config = nullptr;
    const auto status = ucp_config_read(nullptr, nullptr, &config);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to create UCX config: " +
                                 std::string(ucs_status_string(status)));
    }
    return config;
}
} // namespace nixl::ucx
