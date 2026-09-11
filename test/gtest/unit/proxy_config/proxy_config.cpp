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

#include "backend_aux.h"
#include "device/proxy/proxy_config.h"

namespace {

nixlBackendInitParams
makeInitParams(nixl_b_params_t *params, bool enable_prog_th = false) {
    nixlBackendInitParams init_params;
    init_params.localAgent = "test-agent";
    init_params.type = "UCX";
    init_params.customParams = params;
    init_params.enableProgTh = enable_prog_th;
    return init_params;
}

TEST(ProxyConfigTest, DefaultsDisabled) {
    nixlProxyConfig config;

    nixl_b_params_t empty;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&empty), config), NIXL_SUCCESS);
    EXPECT_FALSE(config.enabled);

    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(nullptr), config), NIXL_SUCCESS);
    EXPECT_FALSE(config.enabled);

    nixl_b_params_t off{{"device_proxy", "false"}};
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&off), config), NIXL_SUCCESS);
    EXPECT_FALSE(config.enabled);
}

TEST(ProxyConfigTest, EnabledDefaults) {
    nixl_b_params_t params{{"device_proxy", "true"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.channel_count, 4u);
    EXPECT_EQ(config.thread_count, 4u);
    EXPECT_EQ(config.max_peers, 8u);
    EXPECT_EQ(config.pthr_delay_us, 0u);
    EXPECT_EQ(config.ring_depth, kDefaultProxyRingDepth);
}

TEST(ProxyConfigTest, ExplicitValues) {
    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "8"},
                           {"proxy_thread_count", "2"},
                           {"proxy_max_peers", "32"},
                           {"proxy_pthr_delay_us", "50"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.channel_count, 8u);
    EXPECT_EQ(config.thread_count, 2u);
    EXPECT_EQ(config.max_peers, 32u);
    EXPECT_EQ(config.pthr_delay_us, 50u);
}

TEST(ProxyConfigTest, ThreadCountFollowsChannelCount) {
    nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_channel_count", "6"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_EQ(config.channel_count, 6u);
    EXPECT_EQ(config.thread_count, 6u);
}

TEST(ProxyConfigTest, MalformedValues) {
    nixlProxyConfig config;
    {
        nixl_b_params_t params{{"device_proxy", "maybe"}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
    }
    {
        nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_channel_count", "four"}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
    }
    {
        nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_pthr_delay_us", ""}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
    }
}

TEST(ProxyConfigTest, ZeroCounts) {
    nixlProxyConfig config;
    for (const char *key : {"proxy_channel_count", "proxy_thread_count", "proxy_max_peers"}) {
        nixl_b_params_t params{{"device_proxy", "true"}, {key, "0"}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM)
            << key;
    }
}

TEST(ProxyConfigTest, UnknownProxyKeyRejected) {
    // The pre-rename spelling must fail loudly, not be silently ignored.
    nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_worker_count", "2"}};
    nixlProxyConfig config;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
}

TEST(ProxyConfigTest, EnableKeyTypoRejected) {
    // Near-miss spellings of the enable key are unknown keys, not no-ops.
    nixl_b_params_t params{{"device_proxy_enable", "true"}};
    nixlProxyConfig config;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
}

TEST(ProxyConfigTest, TuningWithoutEnableRejected) {
    nixlProxyConfig config;
    {
        nixl_b_params_t params{{"proxy_channel_count", "2"}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
    }
    {
        nixl_b_params_t params{{"device_proxy", "false"}, {"proxy_max_peers", "16"}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
    }
}

TEST(ProxyConfigTest, ProgressThreadConflict) {
    nixl_b_params_t params{{"device_proxy", "true"}};
    nixlProxyConfig config;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params, true), config), NIXL_ERR_NOT_ALLOWED);

    nixl_b_params_t off_params;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&off_params, true), config), NIXL_SUCCESS);
    EXPECT_FALSE(config.enabled);
}

TEST(ProxyConfigTest, TopologyAccessors) {
    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "3"},
                           {"proxy_thread_count", "2"},
                           {"proxy_max_peers", "5"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_EQ(config.ucxWorkerCount(), 15u);
    EXPECT_EQ(config.effectiveThreadCount(), 2u);
}

TEST(ProxyConfigTest, EffectiveThreadCountClampsToChannelCount) {
    nixl_b_params_t params{{"device_proxy", "true"},
                           {"proxy_channel_count", "2"},
                           {"proxy_thread_count", "8"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    // The parsed value is preserved; only the started-thread count is clamped.
    EXPECT_EQ(config.thread_count, 8u);
    EXPECT_EQ(config.effectiveThreadCount(), 2u);
}

TEST(ProxyConfigTest, RingDepthOverride) {
    nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_ring_depth", "1024"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_EQ(config.ring_depth, 1024u);
}

TEST(ProxyConfigTest, RingDepthMustBeNonZeroPowerOfTwo) {
    nixlProxyConfig config;
    for (const char *depth : {"0", "3", "100", "255"}) {
        nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_ring_depth", depth}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM)
            << depth;
    }
    for (const char *depth : {"1", "2", "512"}) {
        nixl_b_params_t params{{"device_proxy", "true"}, {"proxy_ring_depth", depth}};
        EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS) << depth;
    }
}

TEST(ProxyConfigTest, RingDepthWithoutEnableRejected) {
    nixl_b_params_t params{{"proxy_ring_depth", "512"}};
    nixlProxyConfig config;
    EXPECT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_ERR_INVALID_PARAM);
}

TEST(ProxyConfigTest, NonProxyParamsIgnored) {
    nixl_b_params_t params{{"device_proxy", "true"}, {"num_workers", "16"}, {"num_threads", "3"}};
    nixlProxyConfig config;
    ASSERT_EQ(nixlParseProxyConfig(makeInitParams(&params), config), NIXL_SUCCESS);
    EXPECT_TRUE(config.enabled);
}

} // namespace
