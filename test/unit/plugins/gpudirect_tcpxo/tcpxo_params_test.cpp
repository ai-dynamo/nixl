/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#include <cstdlib>

#include <optional>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "backend_aux.h"
#include "common/nixl_log.h"
#include "gtest/gtest.h"
#include "nixl_types.h"

#include "params.h"
#include "tcpxo_backend.h"
#include "tcpxo_common.h"

namespace tcpxo {
namespace {

    // Saves the current value of `name`, sets it to `value` (or unsets it if `std::nullopt`),
    // and returns a cleanup object that restores the original state on scope exit.
    [[nodiscard]] auto
    ScopedEnvVar(const char *name, std::optional<const char *> value = std::nullopt) {
        std::optional<std::string> prev_val;
        if (const char *curr = std::getenv(name)) {
            prev_val = curr;
        }

        if (value.has_value()) {
            ::setenv(name, *value, 1);
        } else {
            ::unsetenv(name);
        }

        return absl::MakeCleanup([name, prev_val = std::move(prev_val)]() {
            if (prev_val.has_value()) {
                ::setenv(name, prev_val->c_str(), 1);
            } else {
                ::unsetenv(name);
            }
        });
    }

    TEST(TcpxoParamsTest, DefaultParams) {
        auto cleanup_flows = ScopedEnvVar(kFastrakNumFlowsPerDxsConnectionParamName);
        auto cleanup_timeout = ScopedEnvVar(kFastrakDataTransferTimeoutParamName);

        nixlBackendInitParams init;
        nixl_b_params_t custom_params;
        init.localAgent = "TestAgent";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);
        const auto unset_params = GetUnsetParams();

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value,
                  unset_params.fastrak_num_flows_per_dxs_connection.default_value);
        EXPECT_EQ(engine.params().fastrak_data_transfer_timeout_ms.value,
                  unset_params.fastrak_data_transfer_timeout_ms.default_value);
    }

    TEST(TcpxoParamsTest, EnvVars) {
        auto cleanup_flows = ScopedEnvVar(kFastrakNumFlowsPerDxsConnectionParamName, "4");
        auto cleanup_timeout = ScopedEnvVar(kFastrakDataTransferTimeoutParamName, "5000");

        nixlBackendInitParams init;
        nixl_b_params_t custom_params;
        init.localAgent = "TestAgentEnv";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value, 4);
        EXPECT_EQ(engine.params().fastrak_data_transfer_timeout_ms.value, 5000);
    }

    TEST(TcpxoParamsTest, InitParams) {
        nixlBackendInitParams init;
        nixl_b_params_t custom_params;
        custom_params[kFastrakNumFlowsPerDxsConnectionParamName] = "6";
        custom_params[kFastrakDataTransferTimeoutParamName] = "10000";
        init.localAgent = "TestAgentInit";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value, 6);
        EXPECT_EQ(engine.params().fastrak_data_transfer_timeout_ms.value, 10000);
    }

    TEST(TcpxoParamsTest, Override) {
        // init_params should override env vars
        auto cleanup_flows = ScopedEnvVar(kFastrakNumFlowsPerDxsConnectionParamName, "4");

        nixlBackendInitParams init;
        nixl_b_params_t custom_params;
        custom_params[kFastrakNumFlowsPerDxsConnectionParamName] = "7";
        init.localAgent = "TestAgentOverride";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value, 7);
    }

    TEST(TcpxoParamsTest, RangeChecking) {
        nixlBackendInitParams init;
        nixl_b_params_t custom_params;

        // Test max value clamping (max is kFastrakMaxNumFlowsPerDxsConn = 8)
        custom_params[kFastrakNumFlowsPerDxsConnectionParamName] = "100";

        // Test min value clamping (min is 1)
        custom_params[kFastrakDataTransferSlownessParamName] = "0";

        init.localAgent = "TestAgentRange";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value,
                  kFastrakMaxNumFlowsPerDxsConn);
        EXPECT_EQ(engine.params().fastrak_data_transfer_slowness_ms.value, 1);
    }

    TEST(TcpxoParamsTest, InvalidValues) {
        nixlBackendInitParams init;
        nixl_b_params_t custom_params;

        custom_params[kFastrakNumFlowsPerDxsConnectionParamName] = "not_a_number";

        init.localAgent = "TestAgentInvalid";
        init.customParams = &custom_params;
        init.enableProgTh = false;
        init.pthrDelay = 0;

        nixlTcpxoEngine engine(&init);
        const auto unset_params = GetUnsetParams();

        EXPECT_EQ(engine.params().fastrak_num_flows_per_dxs_connection.value,
                  unset_params.fastrak_num_flows_per_dxs_connection.default_value);
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
