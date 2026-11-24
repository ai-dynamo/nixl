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
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_UCX_BACKEND
#include "common.h"
#endif

namespace {

using namespace std::chrono_literals;

TEST(NotifCallbacks, ValidatesAndReusesPrefixes) {
    nixlAgentConfig cfg;
    nixlAgent receiving("notification-receiver", cfg);
    nixlAgent other("notification-other", cfg);
    nixlNotifRegistration first, second;
    const auto consume = [](std::string_view, std::string_view) { return true; };

    EXPECT_EQ(receiving.registerNotifCallback("prefix", {}, first), NIXL_ERR_INVALID_PARAM);
    ASSERT_EQ(receiving.registerNotifCallback("prefix", consume, first), NIXL_SUCCESS);
    EXPECT_EQ(receiving.registerNotifCallback("pre", consume, second), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(receiving.registerNotifCallback("prefix:", consume, second), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(receiving.registerNotifCallback("other", consume, first), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(other.deregisterNotifCallback(first), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(receiving.registerNotifCallback("prefix", consume, second), NIXL_ERR_INVALID_PARAM);

    ASSERT_EQ(receiving.deregisterNotifCallback(first), NIXL_SUCCESS);
    EXPECT_EQ(receiving.deregisterNotifCallback(first), NIXL_SUCCESS);
    {
        nixlNotifRegistration automatic;
        ASSERT_EQ(receiving.registerNotifCallback("prefix", consume, automatic), NIXL_SUCCESS);
    }
    EXPECT_EQ(receiving.registerNotifCallback("prefix", consume, second), NIXL_SUCCESS);
}

TEST(NotifCallbacks, DeclinedAndThrowingCallbacksLeavePayloadAvailable) {
    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    nixlNotifRegistration binary, throwing, fallback;
    const std::string_view prefix("bin\0", 4);
    const std::string_view payload("bin\0ary", 7);
    ASSERT_EQ(dispatcher->add(
                  prefix,
                  [&](std::string_view sender, std::string_view message) {
                      EXPECT_EQ(sender, "remote");
                      EXPECT_EQ(message, payload);
                      return false;
                  },
                  binary),
              NIXL_SUCCESS);
    ASSERT_EQ(dispatcher->add(
                  "bad", [](std::string_view, std::string_view) -> bool { throw 7; }, throwing),
              NIXL_SUCCESS);
    ASSERT_EQ(dispatcher->add(
                  "", [](std::string_view, std::string_view) { return true; }, fallback),
              NIXL_SUCCESS);

    EXPECT_FALSE(dispatcher->dispatch("remote", payload));
    EXPECT_FALSE(dispatcher->dispatch("remote", "bad-notification"));
    EXPECT_TRUE(dispatcher->dispatch("remote", "unmatched"));
    fallback.reset();
    EXPECT_FALSE(dispatcher->dispatch("remote", "unmatched"));
}

TEST(NotifCallbacks, DefaultRequiresDeregistrationBeforeReplacement) {
    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    int first_calls = 0;
    int second_calls = 0;
    const auto first_callback = [&](std::string_view, std::string_view) {
        ++first_calls;
        return true;
    };
    const auto second_callback = [&](std::string_view, std::string_view) {
        ++second_calls;
        return true;
    };

    nixlNotifRegistration first, second;
    EXPECT_FALSE(first);
    ASSERT_EQ(dispatcher->add("", first_callback, first), NIXL_SUCCESS);
    EXPECT_TRUE(first);
    EXPECT_EQ(dispatcher->add("", second_callback, second), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(dispatcher->add("", second_callback, first), NIXL_ERR_INVALID_PARAM);
    EXPECT_FALSE(second);
    EXPECT_TRUE(dispatcher->dispatch("remote", "unmatched"));

    ASSERT_EQ(dispatcher->remove(first), NIXL_SUCCESS);
    EXPECT_FALSE(first);
    ASSERT_EQ(dispatcher->add("", second_callback, second), NIXL_SUCCESS);
    EXPECT_TRUE(dispatcher->dispatch("remote", "unmatched"));
    EXPECT_EQ(first_calls, 1);
    EXPECT_EQ(second_calls, 1);
}

TEST(NotifCallbacks, RemovalRejectsForeignTokensAndShutdownTokensAreReusable) {
    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    const auto other = std::make_shared<nixlNotifDispatcher>();
    const auto consume = [](std::string_view, std::string_view) { return true; };
    nixlNotifRegistration empty, token, reusable;

    EXPECT_EQ(dispatcher->remove(empty), NIXL_SUCCESS);
    ASSERT_EQ(dispatcher->add("prefix", consume, token), NIXL_SUCCESS);
    EXPECT_EQ(other->remove(token), NIXL_ERR_INVALID_PARAM);
    EXPECT_TRUE(dispatcher->dispatch("remote", "prefix"));
    ASSERT_EQ(dispatcher->add("another", consume, reusable), NIXL_SUCCESS);

    nixlNotifRegistration moved(std::move(token));
    EXPECT_FALSE(token);
    EXPECT_TRUE(moved);

    dispatcher->shutdown();
    EXPECT_FALSE(dispatcher->dispatch("remote", "prefix"));
    EXPECT_FALSE(moved);
    EXPECT_FALSE(reusable);
    EXPECT_EQ(other->remove(moved), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(dispatcher->add("closed", consume, empty), NIXL_ERR_NOT_ALLOWED);
    EXPECT_FALSE(empty);

    // The old dispatcher is still alive. Resetting its first token cannot remove the
    // new dispatcher's first registration, even though they have the same numeric ID.
    ASSERT_EQ(other->add("prefix", consume, reusable), NIXL_SUCCESS);
    EXPECT_TRUE(reusable);
    moved.reset();
    EXPECT_TRUE(other->dispatch("remote", "prefix"));
    EXPECT_EQ(other->remove(reusable), NIXL_SUCCESS);
    EXPECT_FALSE(reusable);
}

TEST(NotifCallbacks, ResetDestroysCallbackWithoutHoldingDispatcher) {
    struct captureProbe {
        std::weak_ptr<nixlNotifDispatcher> dispatcher;
        long *result = nullptr;

        ~captureProbe() {
            *result = dispatcher.use_count();
        }
    };

    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    long owners = -1;
    nixlNotifRegistration token;
    {
        const auto probe = std::make_shared<captureProbe>();
        probe->dispatcher = dispatcher;
        probe->result = &owners;
        ASSERT_EQ(
            dispatcher->add(
                "prefix", [probe](std::string_view, std::string_view) { return true; }, token),
            NIXL_SUCCESS);
    }
    token.reset();
    // Only this test owns the dispatcher while the callback's captures are destroyed.
    EXPECT_EQ(owners, 1);
}

TEST(NotifCallbacks, ConcurrentRegistrationAndDispatchAreSafe) {
    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    const auto consume = [](std::string_view, std::string_view) { return true; };
    std::atomic<bool> stop{false};
    auto receiver = std::async(std::launch::async, [&] {
        while (!stop.load()) {
            (void)dispatcher->dispatch("remote", "p0:message");
            (void)dispatcher->dispatch("remote", "unmatched");
        }
    });

    std::vector<std::future<void>> registrars;
    for (int thread = 0; thread < 4; ++thread) {
        registrars.push_back(std::async(std::launch::async, [&, thread] {
            const std::string prefix = "p" + std::to_string(thread) + ":";
            for (int iteration = 0; iteration < 500; ++iteration) {
                nixlNotifRegistration token, fallback;
                EXPECT_EQ(dispatcher->add(prefix, consume, token), NIXL_SUCCESS);
                // Threads compete for the single default, so either result is valid.
                const auto status = dispatcher->add("", consume, fallback);
                EXPECT_TRUE(status == NIXL_SUCCESS || status == NIXL_ERR_INVALID_PARAM);
            }
        }));
    }
    for (auto &registrar : registrars) {
        registrar.get();
    }
    stop.store(true);
    receiver.get();
    EXPECT_FALSE(dispatcher->dispatch("remote", "p0:message"));
    EXPECT_FALSE(dispatcher->dispatch("remote", "unmatched"));
}

TEST(NotifCallbacks, AgentDestructionReleasesCallbackBeforeToken) {
    nixlNotifRegistration token;
    const auto resource = std::make_shared<int>(1);
    const auto baseline = resource.use_count();
    {
        nixlAgentConfig cfg;
        nixlAgent agent("notification-draining", cfg);
        ASSERT_EQ(agent.registerNotifCallback(
                      "cleanup",
                      [resource](std::string_view, std::string_view) { return *resource != 0; },
                      token),
                  NIXL_SUCCESS);
        EXPECT_GT(resource.use_count(), baseline);
        EXPECT_TRUE(token);
    }
    EXPECT_EQ(resource.use_count(), baseline);
    EXPECT_FALSE(token);
    token.reset();
}

TEST(NotifCallbacks, DeregistrationWaitsForRunningCallback) {
    const auto dispatcher = std::make_shared<nixlNotifDispatcher>();
    nixlNotifRegistration token;
    std::promise<void> entered, unblock, removing;
    const auto unblock_signal = unblock.get_future();
    ASSERT_EQ(dispatcher->add(
                  "prefix",
                  [&](std::string_view, std::string_view) {
                      entered.set_value();
                      unblock_signal.wait();
                      return true;
                  },
                  token),
              NIXL_SUCCESS);
    auto receiving =
        std::async(std::launch::async, [&] { return dispatcher->dispatch("remote", "prefix"); });
    const auto entry_status = entered.get_future().wait_for(5s);
    auto removal = std::async(std::launch::async, [&] {
        removing.set_value();
        token.reset();
    });
    removing.get_future().wait();
    const auto removal_status = removal.wait_for(20ms);
    unblock.set_value();
    removal.get();

    EXPECT_EQ(entry_status, std::future_status::ready);
    EXPECT_EQ(removal_status, std::future_status::timeout);
    EXPECT_TRUE(receiving.get());
    EXPECT_FALSE(dispatcher->dispatch("remote", "prefix"));
}

#ifdef HAVE_UCX_BACKEND
TEST(NotifCallbacks, UcxRoutesDynamicRegistrationsAndFallsBackAfterRemoval) {
    gtest::ScopedEnv env;
    env.addVar("NIXL_PLUGIN_DIR", std::string(BUILD_DIR) + "/src/plugins/ucx");
    const gtest::LogIgnoreGuard ib_warning(
        "IB device\\(s\\) were detected, but accelerated IB support was not found");
    const gtest::LogIgnoreGuard efa_warning(
        "Amazon EFA\\(s\\) were detected, but the UCX backend was configured");
    const gtest::LogIgnoreGuard mock_plugin_probe(
        "Plugin file does not exist: .*/test/gtest/mocks/libplugin_UCX.so");

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

    std::promise<std::string> delivered;
    auto delivery = delivered.get_future();
    {
        nixlNotifRegistration token;
        ASSERT_EQ(agent.registerNotifCallback(
                      "prefix:",
                      [&](std::string_view sender, std::string_view message) {
                          EXPECT_EQ(sender, name);
                          if (message == "prefix:declined") {
                              return false;
                          }
                          delivered.set_value(std::string(message));
                          return true;
                      },
                      token),
                  NIXL_SUCCESS);
        ASSERT_EQ(agent.genNotif(name, "prefix:callback"), NIXL_SUCCESS);
        ASSERT_EQ(delivery.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(delivery.get(), "prefix:callback");

        ASSERT_EQ(agent.genNotif(name, "prefix:declined"), NIXL_SUCCESS);
        EXPECT_EQ(receive_fallback(), std::vector<std::string>{"prefix:declined"});
    }

    ASSERT_EQ(agent.genNotif(name, "prefix:legacy"), NIXL_SUCCESS);
    EXPECT_EQ(receive_fallback(), std::vector<std::string>{"prefix:legacy"});
}
#endif

} // namespace
