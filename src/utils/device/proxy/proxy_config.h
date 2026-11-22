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

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <nixl_types.h>

class nixlBackendInitParams;

namespace nixl {

/** Backend-param keys controlling the device proxy. */
inline constexpr std::string_view kProxyEnabledParam = "device_proxy";
inline constexpr std::string_view kProxyChannelCountParam = "proxy_channel_count";
inline constexpr std::string_view kProxyThreadCountParam = "proxy_thread_count";
inline constexpr std::string_view kProxyMaxPeersParam = "proxy_max_peers";
inline constexpr std::string_view kProxyRingDepthParam = "proxy_ring_depth";
inline constexpr std::string_view kProxyDrainTimeoutParam = "proxy_drain_timeout_ms";

inline constexpr uint32_t kDefaultProxyChannelCount = 4;
/** Deliberately small: each peer costs a ring per channel; pass the real peer capacity. */
inline constexpr uint32_t kDefaultProxyMaxPeers = 8;
/** Work-ring slots per (channel, peer); the GPU masks indices, so a power of two. */
inline constexpr uint32_t kDefaultProxyRingDepth = 256;
/** A drain that takes longer aborts the process; 0 waits forever. */
inline constexpr uint32_t kDefaultProxyDrainTimeoutMs = 60000;

/** Peers first, first + stride, ... of a channel. */
struct proxyPeerSlice {
    uint32_t first = 0;
    uint32_t stride = 1;
};

/** One proxy thread's rings: `peers` of each of the channelCount channels from firstChannel. */
struct proxyThreadShare {
    uint32_t firstChannel;
    uint32_t channelCount;
    proxyPeerSlice peers;
};

struct proxyConfig {
    bool enabled = false;
    uint32_t channelCount = kDefaultProxyChannelCount;
    /** Number of proxy CPU progress threads; defaults to channelCount. */
    uint32_t threadCount = kDefaultProxyChannelCount;
    uint32_t maxPeers = kDefaultProxyMaxPeers;
    /** Work-ring depth per (channel, peer) slot; power of two. */
    uint32_t ringDepth = kDefaultProxyRingDepth;
    /** Drain deadline in milliseconds; 0 disables it. */
    uint32_t drainTimeoutMs = kDefaultProxyDrainTimeoutMs;

    [[nodiscard]] size_t
    ringCount() const noexcept {
        return static_cast<size_t>(channelCount) * maxPeers;
    }

    /** Threads split the rings evenly: whole channels, or each channel's peers. */
    [[nodiscard]] bool
    evenThreadSplit() const noexcept {
        if (threadCount == 0 || channelCount == 0) {
            return false;
        }
        if (threadCount <= channelCount) {
            return channelCount % threadCount == 0;
        }
        return threadCount % channelCount == 0 && maxPeers % (threadCount / channelCount) == 0;
    }

    /** Thread `thread`'s rings: a block of whole channels, or a stride of one channel's peers;
     *  requires evenThreadSplit(). */
    [[nodiscard]] proxyThreadShare
    threadShare(uint32_t thread) const noexcept {
        if (threadCount <= channelCount) {
            const uint32_t channels = channelCount / threadCount;
            return {thread * channels, channels, {}};
        }
        const uint32_t per_channel = threadCount / channelCount;
        return {thread / per_channel, 1, {thread % per_channel, per_channel}};
    }
};

/**
 * @brief Parse the device proxy backend parameters.
 * @param[out] config Parsed configuration; unchanged on failure.
 * @retval NIXL_ERR_INVALID_PARAM Unknown proxy key, invalid value, or tuning without
 *         device_proxy=true.
 * @retval NIXL_ERR_NOT_ALLOWED device_proxy=true with the backend progress thread.
 */
[[nodiscard]] nixl_status_t
parseProxyConfig(const nixlBackendInitParams &init_params, proxyConfig &config) noexcept;

} // namespace nixl

#endif // NIXL_SRC_UTILS_DEVICE_PROXY_PROXY_CONFIG_H
