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

const std::string ucx = "UCX";

const std::string name1 = "Agent";
const std::string name2 = "Guard";

struct testAgent {
    testAgent(const std::string &name, const nixlAgentConfig &cfg) : agent(name, cfg) {}

    void
    createBackend() {
        {
            const auto status = agent.getPluginParams(ucx, memories, params);
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
        {
            const auto status = agent.createBackend(ucx, params, backend);
            ASSERT_EQ(status, NIXL_SUCCESS);
            ASSERT_NE(backend, nullptr);
        }
    }

    void
    loadMetadataFrom(testAgent &src, const std::string &src_name) {
        std::string md;
        {
            const auto status = src.agent.getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);
        }
        {
            std::string name;
            const auto status = agent.loadRemoteMD(md, name);
            ASSERT_EQ(status, NIXL_SUCCESS);
            ASSERT_EQ(name, src_name);
        }
    }

    nixlAgent agent;
    nixlBackendH *backend = nullptr;
    nixl_mem_list_t memories;
    nixl_b_params_t params;
};

struct agentPair {
    explicit agentPair(const nixlAgentConfig &cfg) : agent1(name1, cfg), agent2(name2, cfg) {
        agent1.createBackend();
        agent2.createBackend();
        agent1.loadMetadataFrom(agent2, name2);
    }

    void
    genNotif(const std::string &msg) {
        const auto status = agent1.agent.genNotif(name2, msg);
        ASSERT_EQ(status, NIXL_SUCCESS);
    }

    testAgent agent1;
    testAgent agent2;
};

const std::string prefix1 = "notif";
const std::string message1 = prefix1 + "ication_message_1";

const nixl_notif_callback_t dummy_callback([](std::string &&, std::string &&) {});

} // namespace

TEST(NotifCallbacks, AddCallbackFailures) {
    nixlNotifCallbacks cbs;
    EXPECT_THROW(cbs.addCallback("", dummy_callback), std::runtime_error);
    EXPECT_THROW(cbs.addCallback("foo", nixl_notif_callback_t()), std::runtime_error);
    cbs.addCallback("foo", dummy_callback);
    EXPECT_THROW(cbs.addCallback("f", dummy_callback), std::runtime_error);
    EXPECT_THROW(cbs.addCallback("fo", dummy_callback), std::runtime_error);
    EXPECT_THROW(cbs.addCallback("foo", dummy_callback), std::runtime_error);
    EXPECT_THROW(cbs.addCallback("foooo", dummy_callback), std::runtime_error);
}

TEST(NotifCallbacks, DefaultWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.setDefaultCallback([&](std::string &&remote, std::string &&message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}

TEST(NotifCallbacks, PrefixBinarySearchWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.addCallback(prefix1, [&](std::string &&remote, std::string &&message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.addCallback(
        "aaaaa", [](std::string &&remote, std::string &&message) { ADD_FAILURE(); });
    cfg.notifCallbacks.addCallback(
        "zzzzz", [](std::string &&remote, std::string &&message) { ADD_FAILURE(); });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}

TEST(NotifCallbacks, PrefixLinearScanWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.addCallback(
        "aaa", [](std::string &&remote, std::string &&message) { ADD_FAILURE(); });
    cfg.notifCallbacks.addCallback(prefix1, [&](std::string &&remote, std::string &&message) {
        EXPECT_EQ(remote, name1);
        EXPECT_EQ(message, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.addCallback(
        "zzzzzzz", [](std::string &&remote, std::string &&message) { ADD_FAILURE(); });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
}
