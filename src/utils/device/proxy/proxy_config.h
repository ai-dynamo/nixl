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
#ifndef NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONFIG_H
#define NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONFIG_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <nixl_types.h>

class nixlBackendInitParams;

/** Backend-param keys controlling the device proxy. */
inline constexpr std::string_view kProxyEnabledParam = "device_proxy";
inline constexpr std::string_view kProxyChannelCountParam = "proxy_channel_count";
inline constexpr std::string_view kProxyThreadCountParam = "proxy_thread_count";
inline constexpr std::string_view kProxyMaxPeersParam = "proxy_max_peers";
inline constexpr std::string_view kProxyPthrDelayParam = "proxy_pthr_delay_us";
inline constexpr std::string_view kProxyRingDepthParam = "proxy_ring_depth";

inline constexpr uint32_t kDefaultProxyChannelCount = 4;
/**
 * Deliberately small: max_peers sizes real resources (one UCX worker and one
 * ring per channel x peer slot, one endpoint per worker per remote). Deployments
 * should pass their actual peer capacity.
 */
inline constexpr uint32_t kDefaultProxyMaxPeers = 8;
/** Work-ring slots per (channel, peer); the GPU masks indices, so a power of two. */
inline constexpr uint32_t kDefaultProxyRingDepth = 256;

struct nixlProxyConfig {
    bool enabled = false;
    uint32_t channel_count = kDefaultProxyChannelCount;
    /** Number of proxy CPU progress threads; defaults to channel_count. */
    uint32_t thread_count = kDefaultProxyChannelCount;
    uint32_t max_peers = kDefaultProxyMaxPeers;
    /** Proxy thread poll delay; 0 = busy-poll. */
    uint64_t pthr_delay_us = 0;
    /** Work-ring depth per (channel, peer) slot; power of two. */
    uint32_t ring_depth = kDefaultProxyRingDepth;

    /**
     * Backend workers the topology requires: the proxy drives one per
     * (channel, peer) slot, so this is the single source of that count.
     */
    [[nodiscard]] size_t
    ucxWorkerCount() const noexcept {
        return static_cast<size_t>(channel_count) * max_peers;
    }

    /**
     * Proxy threads actually started. Channels are striped across threads, so
     * a thread beyond channel_count would own nothing.
     */
    [[nodiscard]] uint32_t
    effectiveThreadCount() const noexcept {
        return std::min(thread_count, channel_count);
    }
};

/**
 * Parse and strictly validate the proxy backend params.
 *
 * Fails (NIXL_ERR_INVALID_PARAM) on malformed values, zero counts, a ring
 * depth that is not a power of two, unknown proxy_*-prefixed keys, or proxy_*
 * keys given without device_proxy=true.
 * Fails (NIXL_ERR_NOT_ALLOWED) when device_proxy=true is combined with the
 * shared progress thread (enableProgTh).
 * With device_proxy absent/false and no other proxy keys, succeeds with
 * config.enabled == false.
 */
[[nodiscard]] nixl_status_t
nixlParseProxyConfig(const nixlBackendInitParams &init_params, nixlProxyConfig &config) noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONFIG_H
