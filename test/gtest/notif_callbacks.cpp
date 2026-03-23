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

#include "nixl.h"
#include "gtest/gtest.h"

#include <chrono>
#include <future>

namespace {

const std::string backend = "UCX";

const std::string name1 = "Agent";
const std::string name2 = "Guard";

} // namespace

TEST(NotifCallbacks, DefaultWithProgressThread) {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;

    nixlAgent agent1(name1, cfg);
    nixlBackendH *backend1 = nullptr;
    nixl_mem_list_t memories1;
    nixl_b_params_t params1;
    {
        const auto status = agent1.getPluginParams(backend, memories1, params1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent1.createBackend(backend, params1, backend1);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend1, nullptr);
    }
    const std::string message1 = "notification_message_1";
    std::promise<bool> promise;
    cfg.notifCallbacks.setDefaultCallback([&](std::string&& remote, std::string&& message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });

    nixlAgent agent2(name2, cfg);
    nixlBackendH *backend2 = nullptr;
    nixl_b_params_t params2;
    nixl_mem_list_t memories2;
    {
        const auto status = agent2.getPluginParams(backend, memories2, params2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent2.createBackend(backend, params2, backend2);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend2, nullptr);
    }
    std::string metadata2;
    {
        const auto status = agent2.getLocalMD(metadata2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        std::string name;
        const auto status = agent1.loadRemoteMD(metadata2, name);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(name, name2);
    }
    {
        const auto status = agent1.genNotif(name2, message1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}

TEST(NotifCallbacks, PrefixFixedSizeWithProgressThread) {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;

    nixlAgent agent1(name1, cfg);
    nixlBackendH *backend1 = nullptr;
    nixl_mem_list_t memories1;
    nixl_b_params_t params1;
    {
        const auto status = agent1.getPluginParams(backend, memories1, params1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent1.createBackend(backend, params1, backend1);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend1, nullptr);
    }
    const std::string message1 = "notification_message_1";
    std::promise<bool> promise;
    cfg.notifCallbacks.addCallback("notif", [&](std::string&& remote, std::string&& message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.addCallback("aaaaa", [](std::string&& remote, std::string&& message) {
        ADD_FAILURE();
    });
    cfg.notifCallbacks.addCallback("zzzzz", [](std::string&& remote, std::string&& message) {
        ADD_FAILURE();
    });

    nixlAgent agent2(name2, cfg);
    nixlBackendH *backend2 = nullptr;
    nixl_b_params_t params2;
    nixl_mem_list_t memories2;
    {
        const auto status = agent2.getPluginParams(backend, memories2, params2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent2.createBackend(backend, params2, backend2);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend2, nullptr);
    }
    std::string metadata2;
    {
        const auto status = agent2.getLocalMD(metadata2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        std::string name;
        const auto status = agent1.loadRemoteMD(metadata2, name);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(name, name2);
    }
    {
        const auto status = agent1.genNotif(name2, message1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}

TEST(NotifCallbacks, PrefixLinearScanWithProgressThread) {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;

    nixlAgent agent1(name1, cfg);
    nixlBackendH *backend1 = nullptr;
    nixl_mem_list_t memories1;
    nixl_b_params_t params1;
    {
        const auto status = agent1.getPluginParams(backend, memories1, params1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent1.createBackend(backend, params1, backend1);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend1, nullptr);
    }
    const std::string message1 = "notification_message_1";
    std::promise<bool> promise;
    cfg.notifCallbacks.addCallback("aaa", [](std::string&& remote, std::string&& message) {
        ADD_FAILURE();
    });
    cfg.notifCallbacks.addCallback("notif", [&](std::string&& remote, std::string&& message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.addCallback("zzzzzzz", [](std::string&& remote, std::string&& message) {
        ADD_FAILURE();
    });

    nixlAgent agent2(name2, cfg);
    nixlBackendH *backend2 = nullptr;
    nixl_b_params_t params2;
    nixl_mem_list_t memories2;
    {
        const auto status = agent2.getPluginParams(backend, memories2, params2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        const auto status = agent2.createBackend(backend, params2, backend2);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_NE(backend2, nullptr);
    }
    std::string metadata2;
    {
        const auto status = agent2.getLocalMD(metadata2);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    {
        std::string name;
        const auto status = agent1.loadRemoteMD(metadata2, name);
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(name, name2);
    }
    {
        const auto status = agent1.genNotif(name2, message1);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}
