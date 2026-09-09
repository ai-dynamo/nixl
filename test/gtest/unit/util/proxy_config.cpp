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

    ASSERT_EQ(parse({{"device_proxy", "true"},
                     {"proxy_channel_count", "3"},
                     {"proxy_thread_count", "8"},
                     {"proxy_max_peers", "5"},
                     {"proxy_ring_depth", "512"}},
                    config),
              NIXL_SUCCESS);
    EXPECT_EQ(config.channel_count, 3u);
    EXPECT_EQ(config.thread_count, 8u);
    EXPECT_EQ(config.max_peers, 5u);
    EXPECT_EQ(config.ring_depth, 512u);
    EXPECT_EQ(config.ringCount(), 15u);
    EXPECT_EQ(config.effectiveThreadCount(), 3u);
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
        {"progress thread conflict", {{"device_proxy", "true"}}, NIXL_ERR_NOT_ALLOWED, true},
    };
    for (const auto &row : rows) {
        nixl::proxyConfig config;
        config.channel_count = 7;
        EXPECT_EQ(parse(row.params, config, row.enable_prog_th), row.expected) << row.name;
        EXPECT_EQ(config.channel_count, 7u) << row.name << ": config written on failure";
    }
}

} // namespace
