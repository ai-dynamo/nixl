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
#include <string_view>

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

    // Modify the config if it is not already set via environment variable
    void
    modify(std::string_view key, std::string_view value) const;

    // Modify the config always
    void
    modifyAlways(std::string_view key, std::string_view value) const;

private:
    [[nodiscard]] static ucp_config_t *
    readUcpConfig();

    const std::unique_ptr<ucp_config_t, void (*)(ucp_config_t *)> config_{readUcpConfig(),
                                                                          &ucp_config_release};
};
} // namespace nixl::ucx

#endif
