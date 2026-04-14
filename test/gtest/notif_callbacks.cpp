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
#include "backend/notif_callbacks.h"
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
            EXPECT_EQ(status, NIXL_SUCCESS);
        }
        {
            const auto status = agent.createBackend(ucx, params, backend);
            EXPECT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend, nullptr);
        }
    }

    void
    loadMetadataFrom(testAgent &src, const std::string &src_name) {
        std::string md;
        {
            const auto status = src.agent.getLocalMD(md);
            EXPECT_EQ(status, NIXL_SUCCESS);
        }
        {
            std::string name;
            const auto status = agent.loadRemoteMD(md, name);
            EXPECT_EQ(status, NIXL_SUCCESS);
            EXPECT_EQ(name, src_name);
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
        EXPECT_EQ(status, NIXL_SUCCESS);
    }

    [[nodiscard]] nixl_notifs_t
    getNotifs() {
        nixl_notifs_t result;
        const auto status = agent2.agent.getNotifs(result);
        EXPECT_EQ(status, NIXL_SUCCESS);
        return result;
    }

    testAgent agent1;
    testAgent agent2;
};

const std::string prefix1 = "notif";
const std::string message1 = prefix1 + "ication_message_1";

const nixl_notif_callback_t dummy_callback([](nixlNotifCallbackArgs &&) {});

} // namespace

TEST(NotifCallbacks, AddCallbackFailures) {
    {
        std::map<std::string, nixl_notif_callback_t> cbs;
        cbs.try_emplace("foo", nixl_notif_callback_t());
        EXPECT_EQ(cbs.size(), 1);
        nixlNotifCallbacks ncbs;
        EXPECT_THROW(ncbs.assign(cbs), std::runtime_error);
    }
    {
        std::map<std::string, nixl_notif_callback_t> cbs;
        cbs.try_emplace("xyz", dummy_callback);
        cbs.try_emplace("x", dummy_callback);
        EXPECT_EQ(cbs.size(), 2);
        nixlNotifCallbacks ncbs;
        EXPECT_THROW(ncbs.assign(cbs), std::runtime_error);
    }
    {
        std::map<std::string, nixl_notif_callback_t> cbs;
        cbs.try_emplace("xyz", dummy_callback);
        cbs.try_emplace("xy", dummy_callback);
        EXPECT_EQ(cbs.size(), 2);
        nixlNotifCallbacks ncbs;
        EXPECT_THROW(ncbs.assign(cbs), std::runtime_error);
    }
    {
        std::map<std::string, nixl_notif_callback_t> cbs;
        cbs.try_emplace("xyz", dummy_callback);
        cbs.try_emplace("xyzx", dummy_callback);
        EXPECT_EQ(cbs.size(), 2);
        nixlNotifCallbacks ncbs;
        EXPECT_THROW(ncbs.assign(cbs), std::runtime_error);
    }
}

TEST(NotifCallbacks, DefaultWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.try_emplace("", [&](nixlNotifCallbackArgs &&args) {
        EXPECT_EQ(args.remoteAgent, name1);
        EXPECT_EQ(args.notifMessage, message1);
        promise.set_value(true);
    });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
    EXPECT_TRUE(agents.getNotifs().empty());
}

TEST(NotifCallbacks, PrefixBinarySearchWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.try_emplace(prefix1, [&](nixlNotifCallbackArgs &&args) {
        EXPECT_EQ(args.remoteAgent, name1);
        EXPECT_EQ(args.notifMessage, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.try_emplace(
        "aaaaa", [](nixlNotifCallbackArgs &&) { ADD_FAILURE(); });
    cfg.notifCallbacks.try_emplace(
        "zzzzz", [](nixlNotifCallbackArgs &&) { ADD_FAILURE(); });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
    EXPECT_TRUE(agents.getNotifs().empty());
}

TEST(NotifCallbacks, PrefixLinearScanWithProgressThread) {
    std::promise<bool> promise;

    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.notifCallbacks.try_emplace(
        "aaa", [](nixlNotifCallbackArgs &&) { ADD_FAILURE(); });
    cfg.notifCallbacks.try_emplace(prefix1, [&](nixlNotifCallbackArgs &&args) {
        EXPECT_EQ(args.remoteAgent, name1);
        EXPECT_EQ(args.notifMessage, message1);
        promise.set_value(true);
    });
    cfg.notifCallbacks.try_emplace(
        "zzzzzzz", [](nixlNotifCallbackArgs &&) { ADD_FAILURE(); });

    agentPair agents(cfg);
    agents.genNotif(message1);
    const auto future = promise.get_future();
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(5000)), std::future_status::ready);
    EXPECT_TRUE(agents.getNotifs().empty());
}
