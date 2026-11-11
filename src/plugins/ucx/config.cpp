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

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" {
#include <ucs/config/global_opts.h>
}

#include "common/configuration.h"
#include "common/hw_info.h"
#include "common/nixl_log.h"

namespace {

bool
tlsEnablesCudaSupport(std::string_view tls) {
    const bool deny_list = !tls.empty() && tls.front() == '^';
    if (deny_list) {
        tls.remove_prefix(1);
    }

    const std::string tokens = "," + std::string(tls) + ",";
    const bool has_cuda_support =
        std::ranges::any_of(std::array{",all,", ",cuda,", ",cuda_copy,", ",\\cuda_copy,"},
                            [&](auto token) { return tokens.find(token) != std::string::npos; });
    return deny_list ? !has_cuda_support : has_cuda_support;
}

} // namespace

namespace nixl::ucx {

void
config::modify(std::string_view key, std::string_view value) const {
    const auto it = initialValues_.find(std::string(key));
    if (it == initialValues_.end()) {
        modifyUnlessInEnv(key, value);
    } else if (!it->second.isDefault) {
        NIXL_DEBUG << "UCX config already set: UCX_" << key << "=" << it->second.value;
    } else {
        modifyAlways(key, value);
    }
}

void
config::modifyUnlessInEnv(std::string_view key, std::string_view value) const {
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
config::modifyIfEqual(std::string_view key,
                      std::string_view expected,
                      std::string_view value) const {
    const auto it = initialValues_.find(std::string(key));
    if ((it != initialValues_.end()) && (it->second.value != expected)) {
        NIXL_DEBUG << "UCX config already set: UCX_" << key << "=" << it->second.value;
    } else {
        modify(key, value);
    }
}

void
config::validateTlsCudaSupport() const {
    const auto tls = nixl::config::getValueOptional<std::string>("UCX_TLS");
    if (nixl::hwInfo::instance().numNvidiaGpus == 0 || !tls || tlsEnablesCudaSupport(*tls)) {
        return;
    }

    const std::string error = "Invalid UCX_TLS=" + *tls +
        " for NIXL UCX backend: NVIDIA GPU(s) are present, "
        "but this setting does not enable CUDA memory support. Add cuda_copy for "
        "basic GPU support, or cuda to also include NVLink support.";
    NIXL_ERROR << error;
    throw std::runtime_error(error);
}

ucp_config_t *
config::readUcpConfig() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (nixl::hwInfo::instance().numEfaDevices == 0 &&
            !nixl::config::internal::getenvOptional("UCX_MODULES")) {
            const auto mod_status = ucs_global_opts_set_value("MODULES", "^efa");
            if (mod_status != UCS_OK) {
                NIXL_DEBUG << "Failed to set UCX MODULES=^efa: " << ucs_status_string(mod_status);
            } else {
                NIXL_DEBUG << "No EFA devices detected; set UCX MODULES=^efa";
            }
        }
    });

    ucp_config_t *config = nullptr;
    const auto status = ucp_config_read(nullptr, nullptr, &config);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to create UCX config: " +
                                 std::string(ucs_status_string(status)));
    }
    return config;
}

config::dumped_values_t
config::dumpValues(const ucp_config_t *config) {
    char *buf = nullptr;
    size_t size = 0;
    FILE *stream = open_memstream(&buf, &size);
    if (stream == nullptr) {
        NIXL_PDEBUG << "Failed to open a memory stream for the UCX config";
        return {};
    }

    ucp_config_print(config,
                     stream,
                     nullptr,
                     static_cast<ucs_config_print_flags_t>(UCS_CONFIG_PRINT_CONFIG |
                                                           UCS_CONFIG_PRINT_COMMENT_DEFAULT));
    fclose(stream);
    const std::unique_ptr<char, decltype(&free)> buf_guard(buf, &free);

    // The lines are "UCX_<key>=<value>", commented out by "# " for a default value
    constexpr std::string_view default_prefix = "# ";
    constexpr std::string_view ucx_prefix = "UCX_";
    dumped_values_t values;
    std::istringstream dump(std::string(buf, size));
    for (std::string line; std::getline(dump, line);) {
        std::string_view entry = line;
        const bool is_default = entry.starts_with(default_prefix);
        if (is_default) {
            entry.remove_prefix(default_prefix.size());
        }

        const auto pos = entry.find('=');
        if (!entry.starts_with(ucx_prefix) || (pos == std::string_view::npos)) {
            continue;
        }

        values.try_emplace(std::string(entry.substr(ucx_prefix.size(), pos - ucx_prefix.size())),
                           dumpedValue{std::string(entry.substr(pos + 1)), is_default});
    }

    return values;
}
} // namespace nixl::ucx
