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

#ifndef NIXL_SRC_UTILS_UCX_CONFIG_H
#define NIXL_SRC_UTILS_UCX_CONFIG_H

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

extern "C" {
#include <ucp/api/ucp.h>
}

namespace nixl::ucx {
class config {
public:
    // When modify() changes a key
    enum class condition_t {
        // Neither NIXL_UCX_<key> nor UCX_<key> is set in the environment, and the key is not
        // set in a UCX configuration file
        DEFAULT,
        // Always
        ALWAYS
    };

    config() = default;

    [[nodiscard]] ucp_config_t *
    getUcpConfig() const noexcept {
        return config_.get();
    }

    void
    modify(std::string_view key,
           std::string_view value,
           condition_t condition = condition_t::DEFAULT) const;

    // Set the RMA rails under the DEFAULT condition, keeping the rails of the GPU buffers whose
    // NICs UCX assigns
    void
    modifyMaxRmaRails(unsigned rails) const;

    void
    validateTlsCudaSupport() const;

private:
    struct dumpedValue {
        std::string value;
        bool isDefault;
    };

    // By UCX key without the UCX_ prefix
    using dumped_values_t = std::unordered_map<std::string, dumpedValue>;

    [[nodiscard]] static ucp_config_t *
    readUcpConfig();

    [[nodiscard]] static dumped_values_t
    dumpValues(const ucp_config_t &config);

    // The variable name and value that modify() keeps under the condition, if any
    [[nodiscard]] std::optional<std::pair<std::string, std::string>>
    keptValue(std::string_view key, condition_t condition) const;

    const std::unique_ptr<ucp_config_t, void (*)(ucp_config_t *)> config_{readUcpConfig(),
                                                                          &ucp_config_release};
    // The values of config_ as read, before any modification: initialized after config_
    const dumped_values_t initialValues_{dumpValues(*config_)};
};
} // namespace nixl::ucx

#endif
