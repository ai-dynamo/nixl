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

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "nixl.h"
#include "ucx_utils.h"

namespace {

using device_list_t = std::vector<std::string>;

TEST(UcxParams, SplitDeviceList) {
    EXPECT_EQ(ucx_split_device_list(""), device_list_t{});
    EXPECT_EQ(ucx_split_device_list("mlx5_0"), device_list_t{"mlx5_0"});
    EXPECT_EQ(ucx_split_device_list("mlx5_0,mlx5_1"), (device_list_t{"mlx5_0", "mlx5_1"}));
    EXPECT_EQ(ucx_split_device_list("mlx5_0, mlx5_1"), (device_list_t{"mlx5_0", "mlx5_1"}));
    EXPECT_EQ(ucx_split_device_list(" mlx5_0 ,, mlx5_1:2 ,"),
              (device_list_t{"mlx5_0", "mlx5_1:2"}));
}

TEST(UcxParams, DeviceListToNetDevices) {
    EXPECT_EQ(ucx_device_list_to_net_devices({}), "");
    EXPECT_EQ(ucx_device_list_to_net_devices({"mlx5_0"}), "mlx5_0:1");
    EXPECT_EQ(ucx_device_list_to_net_devices({"mlx5_0", "mlx5_1:2"}), "mlx5_0:1,mlx5_1:2");
}

TEST(UcxParams, NetDevicesDecision) {
    const std::optional<std::string> no_env;
    const std::optional<std::string> env = "mlx5_2:1";
    const device_list_t ib{"mlx5_0", "mlx5_1:2"};

    struct Case {
        const char *name;
        device_list_t devs;
        std::string net_devices;
        std::string engine_config;
        std::optional<std::string> env;
        std::optional<std::string> value;
        bool override_env;
        bool env_wins;
    };

    const std::vector<Case> cases = {
        {"nothing set", {}, "", "", no_env, std::nullopt, false, false},
        {"nothing set, env set", {}, "", "", env, std::nullopt, false, false},
        {"ucx_net_devices", {}, "^tun0,docker0", "", no_env, "^tun0,docker0", false, false},
        {"ucx_net_devices, env set", {}, "^tun0,docker0", "", env, "^tun0,docker0", false, true},
        {"device_list", ib, "", "", no_env, "mlx5_0:1,mlx5_1:2", true, false},
        {"device_list, env set", ib, "", "", env, "mlx5_0:1,mlx5_1:2", true, false},
        {"ucx_net_devices, unrelated engine_config",
         {},
         "eth0",
         "RNDV_THRESH=8k",
         no_env,
         "eth0",
         false,
         false},
        {"engine_config NET_DEVICES alone",
         {},
         "",
         "NET_DEVICES=^tun0",
         no_env,
         std::nullopt,
         false,
         false},
    };

    for (const auto &c : cases) {
        const auto d = ucx_net_devices_decide(c.devs, c.net_devices, c.engine_config, c.env);
        EXPECT_EQ(d.value, c.value) << c.name;
        EXPECT_EQ(d.overrideEnv, c.override_env) << c.name;
        EXPECT_EQ(d.envWins, c.env_wins) << c.name;
    }

    // Conflicts.
    EXPECT_THROW((void)ucx_net_devices_decide(ib, "^tun0", "", no_env), std::invalid_argument);
    EXPECT_THROW(
        (void)ucx_net_devices_decide({}, "^tun0", "RNDV_THRESH=8k,NET_DEVICES=eth0", no_env),
        std::invalid_argument);
    EXPECT_THROW((void)ucx_net_devices_decide(ib, "", "NET_DEVICES=eth0", no_env),
                 std::invalid_argument);
}

// Creates a UCX backend in a fresh agent with the plugin defaults plus `overrides`. Returns
// std::nullopt when the UCX plugin is not built.
[[nodiscard]] std::optional<nixl_status_t>
createUcxBackend(const std::string &agent_name, const nixl_b_params_t &overrides) {
    gtest::ScopedEnv env;
    env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");

    nixlAgentConfig cfg;
    nixlAgent agent(agent_name, cfg);

    std::vector<nixl_backend_t> plugins;
    EXPECT_EQ(agent.getAvailPlugins(plugins), NIXL_SUCCESS);
    if (std::find(plugins.begin(), plugins.end(), "UCX") == plugins.end()) {
        return std::nullopt;
    }

    nixl_mem_list_t mems;
    nixl_b_params_t params;
    EXPECT_EQ(agent.getPluginParams("UCX", mems, params), NIXL_SUCCESS);
    // The device parameters are advertised under the names the backend reads.
    EXPECT_EQ(params.count(std::string(nixl_ucx_device_list_param_name)), 1u);
    EXPECT_EQ(params.count(std::string(nixl_ucx_net_devices_param_name)), 1u);
    for (const auto &[key, value] : overrides) {
        params[key] = value;
    }

    nixlBackendH *backend = nullptr;
    const auto status = agent.createBackend("UCX", params, backend);
    EXPECT_EQ(status == NIXL_SUCCESS, backend != nullptr);
    return status;
}

TEST(UcxParams, NetDevicesListCreatesBackend) {
    // Smoke test: a backend accepts a multi-entry exclusion list. What reaches UCX is covered by
    // NetDevicesDecision.
    const auto status = createUcxBackend(
        "ucx_net_devices_list",
        {{std::string(nixl_ucx_net_devices_param_name), "^nixl_test_dev0,nixl_test_dev1"}});
    if (!status) {
        GTEST_SKIP() << "UCX plugin not available";
    }
    EXPECT_EQ(*status, NIXL_SUCCESS);
}

TEST(UcxParams, DeviceListAndNetDevicesAreExclusive) {
    const gtest::LogIgnoreGuard lig_invalid(
        "Failed to create engine: device_list and ucx_net_devices are mutually exclusive");
    const gtest::LogIgnoreGuard lig_backend(
        "backend (creation failed|initialization error) for 'UCX'");
    const auto status =
        createUcxBackend("ucx_net_devices_exclusive",
                         {{std::string(nixl_ucx_device_list_param_name), "mlx5_0"},
                          {std::string(nixl_ucx_net_devices_param_name), "^nixl_test_dev0"}});
    if (!status) {
        GTEST_SKIP() << "UCX plugin not available";
    }
    EXPECT_NE(*status, NIXL_SUCCESS);
    EXPECT_EQ(lig_invalid.getIgnoredCount(), 1u);
}

} // namespace
