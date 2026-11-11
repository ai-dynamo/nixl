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
#include <string>
#include <string_view>
#include <unordered_map>

extern "C" {
#include <ucp/api/ucp.h>
}

namespace nixl::ucx {
class config {
public:
    config() = default;

    [[nodiscard]] ucp_config_t *
    getUcpConfig() const noexcept {
        return config_.get();
    }

    // Modify the config if it was not set via environment variable or UCX configuration file when
    // it was read. The UCX config dump tells it only for the UCP keys, so for another key only the
    // environment variable is checked. Some UCX versions dump a configuration file value as a
    // default one.
    void
    modify(std::string_view key, std::string_view value) const;

    // Modify the config if it is not set via environment variable, also over a value from a UCX
    // configuration file
    void
    modifyUnlessInEnv(std::string_view key, std::string_view value) const;

    // Modify the config always
    void
    modifyAlways(std::string_view key, std::string_view value) const;

    // Modify the config as modify() does, if the key had the value expected when the config was
    // read. With the value that the key has when nobody sets it, this keeps a configuration file
    // value which UCX dumps as a default one.
    void
    modifyIfEqual(std::string_view key, std::string_view expected, std::string_view value) const;

    void
    validateTlsCudaSupport() const;

private:
    struct dumpedValue {
        std::string value;
        bool isDefault; ///< UCX comments the key out in its config dump as a default one
    };

    // By UCX key without the UCX_ prefix
    using dumped_values_t = std::unordered_map<std::string, dumpedValue>;

    [[nodiscard]] static ucp_config_t *
    readUcpConfig();

    [[nodiscard]] static dumped_values_t
    dumpValues(const ucp_config_t *config);

    const std::unique_ptr<ucp_config_t, void (*)(ucp_config_t *)> config_{readUcpConfig(),
                                                                          &ucp_config_release};
    // The values of config_ as read, before any modification: initialized after config_
    const dumped_values_t initialValues_{dumpValues(config_.get())};
};
} // namespace nixl::ucx

#endif
