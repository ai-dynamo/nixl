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
    kProxyEnabledParam,
    kProxyChannelCountParam,
    kProxyThreadCountParam,
    kProxyMaxPeersParam,
    kProxyPthrDelayParam,
    kProxyRingDepthParam,
};

bool
isPowerOfTwo(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool
isProxyParamKey(const std::string &key) {
    // Prefix-match both families so near-miss spellings (device_proxy_enable,
    // proxy_channel_cnt) are rejected as unknown instead of silently ignored.
    return key.rfind("device_proxy", 0) == 0 || key.rfind("proxy_", 0) == 0;
}

} // namespace

nixl_status_t
nixlParseProxyConfig(const nixlBackendInitParams &init_params, nixlProxyConfig &config) noexcept {
    config = nixlProxyConfig{};
    const nixl_b_params_t *params = init_params.customParams;

    bool has_tuning_params = false;
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
            if (key != kProxyEnabledParam) {
                has_tuning_params = true;
            }
        }
    }

    try {
        config.enabled =
            nixl::getBackendParamDefaulted<bool>(params, std::string(kProxyEnabledParam), false);
        const auto channel_count =
            nixl::getBackendParamOptional<uint32_t>(params, std::string(kProxyChannelCountParam));
        const auto thread_count =
            nixl::getBackendParamOptional<uint32_t>(params, std::string(kProxyThreadCountParam));
        const auto max_peers =
            nixl::getBackendParamOptional<uint32_t>(params, std::string(kProxyMaxPeersParam));
        const auto pthr_delay =
            nixl::getBackendParamOptional<uint64_t>(params, std::string(kProxyPthrDelayParam));
        const auto ring_depth =
            nixl::getBackendParamOptional<uint32_t>(params, std::string(kProxyRingDepthParam));

        config.channel_count = channel_count.value_or(kDefaultProxyChannelCount);
        config.thread_count = thread_count.value_or(config.channel_count);
        config.max_peers = max_peers.value_or(kDefaultProxyMaxPeers);
        config.pthr_delay_us = pthr_delay.value_or(0);
        config.ring_depth = ring_depth.value_or(kDefaultProxyRingDepth);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to parse device proxy backend parameters: " << e.what();
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!config.enabled) {
        if (has_tuning_params) {
            NIXL_ERROR << "Device proxy parameters given without " << kProxyEnabledParam
                       << "=true";
            return NIXL_ERR_INVALID_PARAM;
        }
        return NIXL_SUCCESS;
    }

    if (config.channel_count == 0 || config.thread_count == 0 || config.max_peers == 0) {
        NIXL_ERROR << "Device proxy counts must be positive: " << kProxyChannelCountParam << "="
                   << config.channel_count << " " << kProxyThreadCountParam << "="
                   << config.thread_count << " " << kProxyMaxPeersParam << "=" << config.max_peers;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!isPowerOfTwo(config.ring_depth)) {
        NIXL_ERROR << "Device proxy " << kProxyRingDepthParam
                   << " must be a non-zero power of two: " << config.ring_depth;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (config.thread_count > config.channel_count) {
        NIXL_INFO << "Device proxy " << kProxyThreadCountParam << "=" << config.thread_count
                  << " exceeds " << kProxyChannelCountParam << "=" << config.channel_count
                  << "; only " << config.effectiveThreadCount()
                  << " thread(s) will be started (channels are striped across threads)";
    }

    if (init_params.enableProgTh) {
        NIXL_ERROR << "Device proxy progress threads own the backend workers; the backend "
                      "progress thread (enableProgTh) is not allowed with "
                   << kProxyEnabledParam << "=true";
        return NIXL_ERR_NOT_ALLOWED;
    }

    return NIXL_SUCCESS;
}
