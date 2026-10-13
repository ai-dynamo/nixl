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
    for (const char *key : {"rbd_cache",
                            "conf_file",
                            "pool_name",
                            "entity",
                            "rados_conf",
                            "ceph."}) {
        auto params = validBase();
        params[key] = "1";
        EXPECT_FALSE(nixl::rados::parseCustomParams(params).has_value())
            << "unprefixed " << key << " must be rejected";
    }
}

} // namespace gtest::rados
