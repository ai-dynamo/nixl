/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "proxy_config.h"

#include <algorithm>
#include <array>
#include <string>

#include "backend_aux.h"
#include "common/backend.h"
#include "nixl_log.h"

namespace {

constexpr std::array kKnownProxyParams = {
    nixl::kProxyEnabledParam,
    nixl::kProxyChannelCountParam,
    nixl::kProxyThreadCountParam,
    nixl::kProxyMaxPeersParam,
    nixl::kProxyRingDepthParam,
    nixl::kProxyDrainTimeoutParam,
};

[[nodiscard]] bool
isPowerOfTwo(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] bool
isProxyParamKey(const std::string &key) noexcept {
    return key.starts_with("device_proxy") || key.starts_with("proxy_");
}

} // namespace

namespace nixl {

nixl_status_t
parseProxyConfig(const nixlBackendInitParams &init_params, proxyConfig &config) noexcept {
    const nixl_b_params_t *params = init_params.customParams;

    std::string tuning_key;
    if (params != nullptr) {
        for (const auto &[key, value] : *params) {
            if (!isProxyParamKey(key)) {
                continue;
            }
            if (std::find(kKnownProxyParams.begin(), kKnownProxyParams.end(), key) ==
                kKnownProxyParams.end()) {
                NIXL_ERROR << "Unknown device proxy backend parameter '" << key << "'";
                return NIXL_ERR_INVALID_PARAM;
            }
            if (key != kProxyEnabledParam && tuning_key.empty()) {
                tuning_key = key;
            }
        }
    }

    proxyConfig parsed;
    try {
        parsed.enabled =
            getBackendParamDefaulted<bool>(params, std::string(kProxyEnabledParam), false);
        const auto channel_count =
            getBackendParamOptional<uint32_t>(params, std::string(kProxyChannelCountParam));
        const auto thread_count =
            getBackendParamOptional<uint32_t>(params, std::string(kProxyThreadCountParam));
        const auto max_peers =
            getBackendParamOptional<uint32_t>(params, std::string(kProxyMaxPeersParam));
        const auto ring_depth =
            getBackendParamOptional<uint32_t>(params, std::string(kProxyRingDepthParam));
        const auto drain_timeout_ms =
            getBackendParamOptional<uint32_t>(params, std::string(kProxyDrainTimeoutParam));

        parsed.channelCount = channel_count.value_or(kDefaultProxyChannelCount);
        parsed.threadCount = thread_count.value_or(parsed.channelCount);
        parsed.maxPeers = max_peers.value_or(kDefaultProxyMaxPeers);
        parsed.ringDepth = ring_depth.value_or(kDefaultProxyRingDepth);
        parsed.drainTimeoutMs = drain_timeout_ms.value_or(kDefaultProxyDrainTimeoutMs);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to parse device proxy backend parameters: " << e.what();
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!parsed.enabled) {
        if (!tuning_key.empty()) {
            NIXL_ERROR << "Device proxy parameter '" << tuning_key << "' given without "
                       << kProxyEnabledParam << "=true";
            return NIXL_ERR_INVALID_PARAM;
        }
        config = parsed;
        return NIXL_SUCCESS;
    }

    if (parsed.channelCount == 0 || parsed.threadCount == 0 || parsed.maxPeers == 0) {
        NIXL_ERROR << "Device proxy counts must be positive: " << kProxyChannelCountParam << "="
                   << parsed.channelCount << " " << kProxyThreadCountParam << "="
                   << parsed.threadCount << " " << kProxyMaxPeersParam << "=" << parsed.maxPeers;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!isPowerOfTwo(parsed.ringDepth)) {
        NIXL_ERROR << "Device proxy " << kProxyRingDepthParam
                   << " must be a non-zero power of two: " << parsed.ringDepth;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (init_params.enableProgTh) {
        NIXL_ERROR << "Device proxy progress threads own the backend workers; the backend "
                      "progress thread (enableProgTh) is not allowed with "
                   << kProxyEnabledParam << "=true";
        return NIXL_ERR_NOT_ALLOWED;
    }

    if (!parsed.evenThreadSplit()) {
        NIXL_ERROR << "Device proxy " << kProxyThreadCountParam << "=" << parsed.threadCount
                   << " must divide " << kProxyChannelCountParam << "=" << parsed.channelCount
                   << ", or be a multiple of it that divides the " << parsed.ringCount()
                   << " ring(s) (" << kProxyMaxPeersParam << "=" << parsed.maxPeers << ")";
        return NIXL_ERR_INVALID_PARAM;
    }

    config = parsed;
    return NIXL_SUCCESS;
}

} // namespace nixl
