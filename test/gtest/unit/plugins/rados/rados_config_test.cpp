/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rados_config.h"

namespace gtest::rados {

namespace {

    nixl_b_params_t
    validBase() {
        return {{"conf", "fixture.conf"}, {"pool", "qa-pool"}, {"name", "client.qa"}};
    }

} // namespace

TEST(radosConfigTest, ParsesRequiredParametersAndCephOptions) {
    auto params = validBase();
    params["ceph.zeta"] = "3";
    params["ceph.admin_socket"] = "";
    params["ceph.alpha"] = "1";
    params["ceph.log_conf"] = "file:///fixture/log.conf";
    const auto config = nixl::rados::parseCustomParams(params);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->conf, "fixture.conf");
    EXPECT_EQ(config->pool, "qa-pool");
    EXPECT_EQ(config->name, "client.qa");
    EXPECT_EQ(config->ceph_options,
              nixl::rados::ceph_options_t({{"admin_socket", ""},
                                           {"alpha", "1"},
                                           {"log_conf", "file:///fixture/log.conf"},
                                           {"zeta", "3"}}));
}

TEST(radosConfigTest, RejectsInvalidParameterTables) {
    for (const char *key : {"conf", "pool", "name"}) {
        auto absent = validBase();
        absent.erase(key);
        EXPECT_FALSE(nixl::rados::parseCustomParams(absent).has_value())
            << "absent " << key << " must be rejected";
        auto empty = validBase();
        empty[key] = "";
        EXPECT_FALSE(nixl::rados::parseCustomParams(empty).has_value())
            << "empty " << key << " must be rejected";
    }
    // The reserved-key match is exact, so near-miss names stay unknown keys.
    for (const char *key : {"rbd_cache",
                            "conf_file",
                            "pool_name",
                            "entity",
                            "rados_conf",
                            "ceph.",
                            "status-wait-us",
                            "STATUS_WAIT_US",
                            "status_wait",
                            "xstatus_wait_us"}) {
        auto params = validBase();
        params[key] = "1";
        EXPECT_FALSE(nixl::rados::parseCustomParams(params).has_value())
            << "unprefixed " << key << " must be rejected";
    }
}

TEST(radosConfigTest, StatusWaitUsStrictDecimalParseTable) {
    const auto absent = nixl::rados::parseCustomParams(validBase());
    ASSERT_TRUE(absent.has_value());
    EXPECT_EQ(absent->status_wait_us, 0U);

    // Digits only, leading zeros valid, the 1,000,000-us ceiling accepted.
    const std::vector<std::pair<std::string, uint64_t>> accepted = {
        {"0", 0},
        {"00000", 0},
        {"1", 1},
        {"10", 10},
        {"000010", 10},
        {"999999", 999999},
        {"1000000", 1000000},
        {"000000000000000000001", 1},
    };
    for (const auto &[text, expected] : accepted) {
        auto params = validBase();
        params["status_wait_us"] = text;
        const auto config = nixl::rados::parseCustomParams(params);
        ASSERT_TRUE(config.has_value()) << "value '" << text << "' must parse";
        EXPECT_EQ(config->status_wait_us, expected) << "value '" << text << "'";
        EXPECT_TRUE(config->ceph_options.empty()) << "reserved key must not reach Ceph options";
    }

    // Malformed spellings and above-ceiling/overflowing values are failures,
    // not truncations.
    for (const char *text : {"",
                             "-1",
                             "+1",
                             " 1",
                             "1 ",
                             "\t1",
                             "1.0",
                             "1,000",
                             "1e3",
                             "0x10",
                             "abc",
                             "1_0",
                             "1000001",
                             "99999999999999999999999999",
                             "18446744073709551615",
                             "18446744073709551616"}) {
        auto params = validBase();
        params["status_wait_us"] = text;
        EXPECT_FALSE(nixl::rados::parseCustomParams(params).has_value())
            << "value '" << text << "' must be rejected";
    }
}

TEST(radosConfigTest, KeepsCephStatusWaitUsAsNormalCephOption) {
    // The exact reserved-key match leaves `ceph.status_wait_us` a Ceph option.
    auto params = validBase();
    params["ceph.status_wait_us"] = "4242";
    auto config = nixl::rados::parseCustomParams(params);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->status_wait_us, 0U);
    EXPECT_EQ(config->ceph_options, nixl::rados::ceph_options_t({{"status_wait_us", "4242"}}));

    // Both spellings coexist: engine budget and Ceph option are distinct.
    params["status_wait_us"] = "500";
    config = nixl::rados::parseCustomParams(params);
    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->status_wait_us, 500U);
    EXPECT_EQ(config->ceph_options, nixl::rados::ceph_options_t({{"status_wait_us", "4242"}}));
}

TEST(radosConfigTest, ParsesOptionalNamespace) {
    const auto absent = nixl::rados::parseCustomParams(validBase());
    ASSERT_TRUE(absent.has_value());
    EXPECT_FALSE(absent->ns.has_value());

    auto params = validBase();
    params["namespace"] = "qa-images";
    const auto config = nixl::rados::parseCustomParams(params);
    ASSERT_TRUE(config.has_value());
    ASSERT_TRUE(config->ns.has_value());
    EXPECT_EQ(*config->ns, "qa-images");
    EXPECT_TRUE(config->ceph_options.empty()) << "reserved key must not reach Ceph options";
}

TEST(radosConfigTest, RejectsEmptyNulAndSentinelNamespace) {
    auto empty = validBase();
    empty["namespace"] = "";
    EXPECT_FALSE(nixl::rados::parseCustomParams(empty).has_value());

    auto nul = validBase();
    nul["namespace"] = std::string("qa\0ns", 5);
    EXPECT_FALSE(nixl::rados::parseCustomParams(nul).has_value());

    auto only_nul = validBase();
    only_nul["namespace"] = std::string("\0", 1);
    EXPECT_FALSE(nixl::rados::parseCustomParams(only_nul).has_value());

    // Exact-match rejection of LIBRADOS_ALL_NSPACES; longer names merely
    // starting with 0x01 stay valid namespaces.
    auto sentinel = validBase();
    sentinel["namespace"] = std::string("\x01", 1);
    EXPECT_FALSE(nixl::rados::parseCustomParams(sentinel).has_value());
}

} // namespace gtest::rados
