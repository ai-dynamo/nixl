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

#include <gtest/gtest.h>

#include <vector>

#include "backend_aux.h"
#include "device/proxy/proxy_config.h"

namespace {

nixl_status_t
parse(nixl_b_params_t params, nixl::proxyConfig &config, bool enable_prog_th = false) {
    nixlBackendInitParams init_params;
    init_params.localAgent = "test-agent";
    init_params.type = "UCX";
    init_params.customParams = &params;
    init_params.enableProgTh = enable_prog_th;
    return nixl::parseProxyConfig(init_params, config);
}

TEST(ProxyConfigTest, AcceptsParameters) {
    nixl::proxyConfig config;
    EXPECT_EQ(parse({}, config), NIXL_SUCCESS);
    EXPECT_FALSE(config.enabled);

    ASSERT_EQ(parse({{"device_proxy", "true"}}, config), NIXL_SUCCESS);
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.channel_count, nixl::kDefaultProxyChannelCount);
    EXPECT_EQ(config.thread_count, nixl::kDefaultProxyChannelCount);
    EXPECT_EQ(config.max_peers, nixl::kDefaultProxyMaxPeers);
    EXPECT_EQ(config.ring_depth, nixl::kDefaultProxyRingDepth);
    EXPECT_EQ(config.drain_timeout_ms, nixl::kDefaultProxyDrainTimeoutMs);

    ASSERT_EQ(parse({{"device_proxy", "true"},
                     {"proxy_channel_count", "3"},
                     {"proxy_thread_count", "15"},
                     {"proxy_max_peers", "5"},
                     {"proxy_ring_depth", "512"},
                     {"proxy_drain_timeout_ms", "0"}},
                    config),
              NIXL_SUCCESS);
    EXPECT_EQ(config.channel_count, 3u);
    EXPECT_EQ(config.thread_count, 15u);
    EXPECT_EQ(config.max_peers, 5u);
    EXPECT_EQ(config.ring_depth, 512u);
    EXPECT_EQ(config.drain_timeout_ms, 0u);
    EXPECT_EQ(config.ringCount(), 15u);
}

TEST(ProxyConfigTest, RejectsParameters) {
    struct Row {
        const char *name;
        nixl_b_params_t params;
        nixl_status_t expected;
        bool enable_prog_th = false;
    };

    const std::vector<Row> rows = {
        {"malformed value",
         {{"device_proxy", "true"}, {"proxy_channel_count", "four"}},
         NIXL_ERR_INVALID_PARAM},
        {"zero count",
         {{"device_proxy", "true"}, {"proxy_max_peers", "0"}},
         NIXL_ERR_INVALID_PARAM},
        {"ring depth not a power of two",
         {{"device_proxy", "true"}, {"proxy_ring_depth", "100"}},
         NIXL_ERR_INVALID_PARAM},
        {"unknown proxy key",
         {{"device_proxy", "true"}, {"proxy_worker_count", "2"}},
         NIXL_ERR_INVALID_PARAM},
        {"tuning without enable", {{"proxy_channel_count", "2"}}, NIXL_ERR_INVALID_PARAM},
        {"threads not dividing channels",
         {{"device_proxy", "true"}, {"proxy_channel_count", "4"}, {"proxy_thread_count", "3"}},
         NIXL_ERR_INVALID_PARAM},
        {"threads not a multiple of channels",
         {{"device_proxy", "true"}, {"proxy_channel_count", "4"}, {"proxy_thread_count", "6"}},
         NIXL_ERR_INVALID_PARAM},
        {"threads per channel not dividing peers",
         {{"device_proxy", "true"},
          {"proxy_channel_count", "4"},
          {"proxy_thread_count", "12"},
          {"proxy_max_peers", "8"}},
         NIXL_ERR_INVALID_PARAM},
        {"more threads than rings",
         {{"device_proxy", "true"},
          {"proxy_channel_count", "4"},
          {"proxy_thread_count", "64"},
          {"proxy_max_peers", "8"}},
         NIXL_ERR_INVALID_PARAM},
        {"progress thread conflict", {{"device_proxy", "true"}}, NIXL_ERR_NOT_ALLOWED, true},
    };
    for (const auto &row : rows) {
        nixl::proxyConfig config;
        config.channel_count = 7;
        EXPECT_EQ(parse(row.params, config, row.enable_prog_th), row.expected) << row.name;
        EXPECT_EQ(config.channel_count, 7u) << row.name << ": config written on failure";
    }
}

TEST(ProxyConfigTest, EvenThreadSplits) {
    nixl::proxyConfig config;
    config.channel_count = 4;
    config.max_peers = 8;
    std::vector<uint32_t> accepted;
    for (uint32_t threads = 1; threads <= 64; ++threads) {
        config.thread_count = threads;
        if (config.evenThreadSplit()) {
            accepted.push_back(threads);
        }
    }
    EXPECT_EQ(accepted, (std::vector<uint32_t>{1, 2, 4, 8, 16, 32}));
}

TEST(ProxyConfigTest, ThreadShares) {
    for (uint32_t channels : {1u, 2u, 3u, 4u, 8u}) {
        for (uint32_t peers : {1u, 2u, 3u, 4u, 8u}) {
            for (uint32_t threads = 1; threads <= channels * peers; ++threads) {
                nixl::proxyConfig config;
                config.channel_count = channels;
                config.max_peers = peers;
                config.thread_count = threads;
                if (!config.evenThreadSplit()) {
                    continue;
                }
                std::vector<uint32_t> owners(config.ringCount(), 0);
                for (uint32_t thread = 0; thread < threads; ++thread) {
                    const nixl::proxyThreadShare share = config.threadShare(thread);
                    ASSERT_LE(share.first_channel + share.channel_count, channels);
                    size_t owned = 0;
                    for (uint32_t channel = share.first_channel;
                         channel < share.first_channel + share.channel_count;
                         ++channel) {
                        for (uint32_t peer = share.peers.first; peer < peers;
                             peer += share.peers.stride) {
                            ++owners[channel * peers + peer];
                            ++owned;
                        }
                    }
                    EXPECT_EQ(owned, config.ringCount() / threads)
                        << channels << " channels, " << peers << " peers, " << threads
                        << " threads";
                }
                EXPECT_EQ(owners, std::vector<uint32_t>(config.ringCount(), 1))
                    << channels << " channels, " << peers << " peers, " << threads << " threads";
            }
        }
    }
}

} // namespace
