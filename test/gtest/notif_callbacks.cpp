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
#include "notif_dispatcher.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common.h"

namespace {

using namespace std::chrono_literals;

template<typename functionType> class functionCallback final : public nixlNotifCallback {
public:
    explicit functionCallback(functionType function) : function_(std::move(function)) {}

    void
    received(std::string_view remote_agent, std::string_view message) override {
        function_(remote_agent, message);
    }

private:
    functionType function_;
};

template<typename functionType>
std::shared_ptr<nixlNotifCallback>
makeCallback(functionType function) {
    return std::make_shared<functionCallback<functionType>>(std::move(function));
}

// Shared by all registrations that use it, which is allowed.
const auto ignore = makeCallback([](std::string_view, std::string_view) {});

TEST(NotifCallbacks, ValidatesAndReusesPrefixes) {
    nixlAgentConfig cfg;
    nixlAgent agent("notification-receiver", cfg);

    EXPECT_EQ(agent.registerNotifCallback("prefix", nullptr), NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(agent.registerNotifCallback("prefix", ignore), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback("pre", ignore), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("prefix:", ignore), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("prefix", ignore), NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(agent.registerNotifCallback("other", ignore), NIXL_SUCCESS);

    // Deregistration needs the exact prefix.
    EXPECT_EQ(agent.deregisterNotifCallback("pre"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.deregisterNotifCallback(""), NIXL_ERR_NOT_FOUND);
    ASSERT_EQ(agent.deregisterNotifCallback("prefix"), NIXL_SUCCESS);
    EXPECT_EQ(agent.deregisterNotifCallback("prefix"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.registerNotifCallback("prefix:", ignore), NIXL_SUCCESS);
}

TEST(NotifCallbacks, MatchingCallbackOwnsNotificationEvenWhenItThrows) {
    nixlNotifDispatcher dispatcher;
    const std::string_view prefix("bin\0", 4);
    const std::string_view payload("bin\0ary", 7);
    int binary_calls = 0;
    const auto binary = makeCallback([&](std::string_view sender, std::string_view message) {
        EXPECT_EQ(sender, "remote");
        EXPECT_EQ(message, payload);
        ++binary_calls;
    });
    const auto failing = makeCallback(
        [](std::string_view, std::string_view) { throw std::runtime_error("callback failure"); });
    const auto odd = makeCallback([](std::string_view, std::string_view) { throw 7; });
    ASSERT_EQ(dispatcher.add(prefix, binary), NIXL_SUCCESS);
    ASSERT_EQ(dispatcher.add("bad", failing), NIXL_SUCCESS);
    ASSERT_EQ(dispatcher.add("odd", odd), NIXL_SUCCESS);

    EXPECT_TRUE(dispatcher.dispatch("remote", payload));
    EXPECT_EQ(binary_calls, 1);
    EXPECT_FALSE(dispatcher.dispatch("remote", "unmatched"));

    // A failing callback still owns the notification, which is dropped with an error.
    const gtest::LogIgnoreGuard failure(
        "Notification callback failed on a notification from 'remote'");
    EXPECT_TRUE(dispatcher.dispatch("remote", "bad-notification"));
    EXPECT_TRUE(dispatcher.dispatch("remote", "odd-notification"));
    EXPECT_EQ(failure.getIgnoredCount(), 2u);
}

TEST(NotifCallbacks, DefaultRequiresDeregistrationBeforeReplacement) {
    nixlNotifDispatcher dispatcher;
    int first_calls = 0;
    int second_calls = 0;
    int prefix_calls = 0;
    const auto first = makeCallback([&](std::string_view, std::string_view) { ++first_calls; });
    const auto second = makeCallback([&](std::string_view, std::string_view) { ++second_calls; });
    const auto prefixed = makeCallback([&](std::string_view, std::string_view) { ++prefix_calls; });

    ASSERT_EQ(dispatcher.add("", first), NIXL_SUCCESS);
    EXPECT_EQ(dispatcher.add("", second), NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(dispatcher.add("kv:", prefixed), NIXL_SUCCESS);
    EXPECT_TRUE(dispatcher.dispatch("remote", "unmatched"));
    EXPECT_TRUE(dispatcher.dispatch("remote", "kv:matched"));

    ASSERT_EQ(dispatcher.remove(""), NIXL_SUCCESS);
    EXPECT_FALSE(dispatcher.dispatch("remote", "unmatched"));
    ASSERT_EQ(dispatcher.add("", second), NIXL_SUCCESS);
    EXPECT_TRUE(dispatcher.dispatch("remote", "unmatched"));
    EXPECT_EQ(first_calls, 1);
    EXPECT_EQ(second_calls, 1);
    EXPECT_EQ(prefix_calls, 1);
}

TEST(NotifCallbacks, ConcurrentRegistrationAndDispatchAreSafe) {
    nixlNotifDispatcher dispatcher;
    std::atomic<bool> stop{false};
    auto receiver = std::async(std::launch::async, [&] {
        while (!stop.load()) {
            (void)dispatcher.dispatch("remote", "p0:message");
            (void)dispatcher.dispatch("remote", "unmatched");
        }
    });

    std::vector<std::future<void>> registrars;
    for (int thread = 0; thread < 4; ++thread) {
        registrars.push_back(std::async(std::launch::async, [&, thread] {
            const std::string prefix = "p" + std::to_string(thread) + ":";
            for (int iteration = 0; iteration < 500; ++iteration) {
                EXPECT_EQ(dispatcher.add(prefix, ignore), NIXL_SUCCESS);
                // Threads compete for the single default, and each removes only its own.
                const bool has_default = dispatcher.add("", ignore) == NIXL_SUCCESS;
                EXPECT_EQ(dispatcher.remove(prefix), NIXL_SUCCESS);
                if (has_default) {
                    EXPECT_EQ(dispatcher.remove(""), NIXL_SUCCESS);
                }
            }
        }));
    }
    for (auto &registrar : registrars) {
        registrar.get();
    }
    stop.store(true);
    receiver.get();
    EXPECT_FALSE(dispatcher.hasRegistrations());
    EXPECT_FALSE(dispatcher.dispatch("remote", "p0:message"));
}

TEST(NotifCallbacks, AgentDestructionReleasesCallbacks) {
    std::weak_ptr<nixlNotifCallback> registered;
    {
        nixlAgentConfig cfg;
        nixlAgent agent("notification-draining", cfg);
        auto callback = makeCallback([](std::string_view, std::string_view) {});
        registered = callback;
        ASSERT_EQ(agent.registerNotifCallback("cleanup", std::move(callback)), NIXL_SUCCESS);
        EXPECT_FALSE(registered.expired());
    }
    EXPECT_TRUE(registered.expired());
}

TEST(NotifCallbacks, DeregistrationWaitsForRunningCallback) {
    nixlNotifDispatcher dispatcher;
    std::promise<void> entered, unblock, removing;
    const auto unblock_signal = unblock.get_future();
    const auto blocking = makeCallback([&](std::string_view, std::string_view) {
        entered.set_value();
        unblock_signal.wait();
    });
    ASSERT_EQ(dispatcher.add("prefix", blocking), NIXL_SUCCESS);
    auto receiving =
        std::async(std::launch::async, [&] { return dispatcher.dispatch("remote", "prefix"); });
    const auto entry_status = entered.get_future().wait_for(5s);
    auto removal = std::async(std::launch::async, [&] {
        removing.set_value();
        return dispatcher.remove("prefix");
    });
    removing.get_future().wait();
    const auto removal_status = removal.wait_for(20ms);
    unblock.set_value();

    EXPECT_EQ(removal.get(), NIXL_SUCCESS);
    EXPECT_EQ(entry_status, std::future_status::ready);
    EXPECT_EQ(removal_status, std::future_status::timeout);
    EXPECT_TRUE(receiving.get());
    EXPECT_FALSE(dispatcher.dispatch("remote", "prefix"));
}

#ifdef HAVE_UCX_BACKEND
TEST(NotifCallbacks, UcxRoutesMatchesAndFallsBackAfterDeregistration) {
    gtest::ScopedEnv env;
    env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");
    const gtest::LogIgnoreGuard ib_warning(
        "IB device\\(s\\) were detected, but accelerated IB support was not found");
    const gtest::LogIgnoreGuard efa_warning(
        "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
    const gtest::LogIgnoreGuard mock_plugin_probe(
        "Plugin file does not exist: .*/test/gtest/mocks/libplugin_UCX.so");

    std::promise<std::string> delivered;
    auto delivery = delivered.get_future();
    const std::string name = "dynamic-callback-ucx";
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    nixlAgent agent(name, cfg);
    nixlBackendH *backend = nullptr;
    ASSERT_EQ(agent.createBackend("UCX", {}, backend), NIXL_SUCCESS);
    const auto receive_fallback = [&] {
        nixl_notifs_t pending;
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (pending.empty() && std::chrono::steady_clock::now() < deadline) {
            EXPECT_EQ(agent.getNotifs(pending), NIXL_SUCCESS);
            if (pending.empty()) {
                std::this_thread::sleep_for(1ms);
            }
        }
        return pending[name];
    };

    const auto deliver = makeCallback([&](std::string_view sender, std::string_view message) {
        EXPECT_EQ(sender, name);
        delivered.set_value(std::string(message));
    });
    ASSERT_EQ(agent.registerNotifCallback("prefix:", deliver), NIXL_SUCCESS);
    ASSERT_EQ(agent.genNotif(name, "prefix:callback"), NIXL_SUCCESS);
    ASSERT_EQ(delivery.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(delivery.get(), "prefix:callback");

    ASSERT_EQ(agent.genNotif(name, "unmatched"), NIXL_SUCCESS);
    EXPECT_EQ(receive_fallback(), std::vector<std::string>{"unmatched"});

    ASSERT_EQ(agent.deregisterNotifCallback("prefix:"), NIXL_SUCCESS);
    ASSERT_EQ(agent.genNotif(name, "prefix:legacy"), NIXL_SUCCESS);
    EXPECT_EQ(receive_fallback(), std::vector<std::string>{"prefix:legacy"});
}
#endif

} // namespace
