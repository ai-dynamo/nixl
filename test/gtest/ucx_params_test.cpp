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
#include <string>
#include <vector>

#include "common.h"
#include "nixl.h"
#include "ucx_utils.h"

namespace {

using device_list_t = std::vector<std::string>;

TEST(UcxParams, SplitDeviceList) {
    EXPECT_EQ(ucxSplitDeviceList(""), device_list_t{});
    EXPECT_EQ(ucxSplitDeviceList(" , ,"), device_list_t{});
    EXPECT_EQ(ucxSplitDeviceList("mlx5_0"), device_list_t{"mlx5_0"});
    EXPECT_EQ(ucxSplitDeviceList("mlx5_0,mlx5_1"), (device_list_t{"mlx5_0", "mlx5_1"}));
    EXPECT_EQ(ucxSplitDeviceList("mlx5_0, mlx5_1"), (device_list_t{"mlx5_0", "mlx5_1"}));
    EXPECT_EQ(ucxSplitDeviceList(" mlx5_0 ,, mlx5_1:2 ,"), (device_list_t{"mlx5_0", "mlx5_1:2"}));
}

TEST(UcxParams, DeviceListToNetDevices) {
    EXPECT_EQ(ucxDeviceListToNetDevices({}), "");
    EXPECT_EQ(ucxDeviceListToNetDevices({"mlx5_0"}), "mlx5_0:1");
    EXPECT_EQ(ucxDeviceListToNetDevices({"mlx5_0", "mlx5_1:2"}), "mlx5_0:1,mlx5_1:2");
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
    // The device list is advertised under the name the backend reads.
    EXPECT_EQ(params.count(std::string(nixl_ucx_device_list_param_name)), 1u);
    EXPECT_EQ(params.count("ucx_devices"), 0u);
    for (const auto &[key, value] : overrides) {
        params[key] = value;
    }

    nixlBackendH *backend = nullptr;
    const auto status = agent.createBackend("UCX", params, backend);
    EXPECT_EQ(status == NIXL_SUCCESS, backend != nullptr);
    return status;
}

TEST(UcxParams, AdvertisedDefaultsCreateBackend) {
    // getPluginParams advertises device_list with an empty default; passing it back unchanged
    // means no device restriction.
    const auto status = createUcxBackend("ucx_params_defaults", {});
    if (!status) {
        GTEST_SKIP() << "UCX plugin not available";
    }
    EXPECT_EQ(*status, NIXL_SUCCESS);
}

TEST(UcxParams, DeviceListWithoutDevicesFails) {
    const gtest::LogIgnoreGuard lig_invalid(
        "Failed to create engine: device_list=\",,,\" names no devices");
    const gtest::LogIgnoreGuard lig_backend(
        "backend (creation failed|initialization error) for 'UCX'");
    const auto status = createUcxBackend("ucx_params_empty_device_list",
                                         {{std::string(nixl_ucx_device_list_param_name), ",,,"}});
    if (!status) {
        GTEST_SKIP() << "UCX plugin not available";
    }
    EXPECT_NE(*status, NIXL_SUCCESS);
    EXPECT_EQ(lig_invalid.getIgnoredCount(), 1u);
}

} // namespace
