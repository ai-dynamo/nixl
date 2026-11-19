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
#include "common.h"
#include "mocks/gmock_engine.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;

const std::string ucx = "UCX";
const std::string sender_name = "Sender";
const std::string receiver_name = "Receiver";
const std::string prefix = "notif";
const std::string message = prefix + "ication_message_1";
const std::string unsupported_log = "notification callback";

constexpr std::chrono::milliseconds wait_time = 10'000ms;
constexpr std::chrono::milliseconds quiet_time = 200ms;
constexpr std::chrono::milliseconds poll_interval = 1ms;

const nixl_notif_callback_t noop_callback([](nixlNotifCallbackArgs &&) {});

[[nodiscard]] std::string
sequenceMessage(const std::string &head, size_t i) {
    return head + std::to_string(i);
}

struct testAgent {
    testAgent(const std::string &name, const nixlAgentConfig &cfg) : agent(name, cfg) {}

    void
    createBackend() {
        nixl_mem_list_t memories;
        nixl_b_params_t params;
        EXPECT_EQ(agent.getPluginParams(ucx, memories, params), NIXL_SUCCESS);
        EXPECT_EQ(agent.createBackend(ucx, params, backend), NIXL_SUCCESS);
        EXPECT_NE(backend, nullptr);
    }

    void
    loadMetadataFrom(testAgent &src, const std::string &src_name) {
        std::string md;
        EXPECT_EQ(src.agent.getLocalMD(md), NIXL_SUCCESS);
        std::string name;
        EXPECT_EQ(agent.loadRemoteMD(md, name), NIXL_SUCCESS);
        EXPECT_EQ(name, src_name);
    }

    nixlAgent agent;
    nixlBackendH *backend = nullptr;
};

struct recordedCall {
    std::string remoteAgent;
    std::string notifMessage;
    std::thread::id thread;
};

class callbackRecorder {
public:
    [[nodiscard]] nixl_notif_callback_t
    callback() {
        return [this](nixlNotifCallbackArgs &&args) { record(std::move(args)); };
    }

    void
    record(nixlNotifCallbackArgs &&args) {
        const std::lock_guard lock(mutex_);
        calls_.push_back(
            {std::move(args.remoteAgent), std::move(args.notifMessage), std::this_thread::get_id()});
    }

    [[nodiscard]] size_t
    count() const {
        const std::lock_guard lock(mutex_);
        return calls_.size();
    }

    [[nodiscard]] std::vector<recordedCall>
    calls() const {
        const std::lock_guard lock(mutex_);
        return calls_;
    }

    [[nodiscard]] std::vector<std::string>
    messages() const {
        const std::lock_guard lock(mutex_);
        std::vector<std::string> result;
        for (const auto &call : calls_) {
            result.push_back(call.notifMessage);
        }
        return result;
    }

private:
    mutable std::mutex mutex_;
    std::vector<recordedCall> calls_;
};

class ucxPair {
public:
    explicit ucxPair(const nixlAgentConfig &receiver_cfg)
        : sender_(std::make_unique<testAgent>(sender_name, senderConfig())),
          receiver_(std::make_unique<testAgent>(receiver_name, receiver_cfg)) {
        sender_->createBackend();
        receiver_->createBackend();
        sender_->loadMetadataFrom(*receiver_, receiver_name);
    }

    void
    send(const nixl_blob_t &msg) {
        EXPECT_EQ(trySend(msg), NIXL_SUCCESS);
    }

    nixl_status_t
    trySend(const nixl_blob_t &msg) {
        return sender_->agent.genNotif(receiver_name, msg);
    }

    [[nodiscard]] nixlAgent &
    receiver() noexcept {
        return receiver_->agent;
    }

    void
    destroyReceiver() {
        receiver_.reset();
    }

    void
    poll() {
        nixl_notifs_t notifs;
        EXPECT_EQ(receiver_->agent.getNotifs(notifs), NIXL_SUCCESS);
        for (auto &[agent, msgs] : notifs) {
            for (auto &msg : msgs) {
                inbox_.emplace_back(agent, std::move(msg));
            }
        }
    }

    [[nodiscard]] bool
    pollUntil(const std::function<bool()> &done) {
        const auto deadline = std::chrono::steady_clock::now() + wait_time;
        while (true) {
            poll();
            if (done()) {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(poll_interval);
        }
    }

    [[nodiscard]] bool
    inboxHas(const std::string &msg) const {
        return std::any_of(inbox_.begin(), inbox_.end(), [&](const auto &entry) {
            return entry.second == msg;
        });
    }

    [[nodiscard]] bool
    sendSentinel(const std::string &sentinel) {
        send(sentinel);
        return pollUntil([&] { return inboxHas(sentinel); });
    }

    [[nodiscard]] const std::vector<std::pair<std::string, std::string>> &
    inbox() const noexcept {
        return inbox_;
    }

    [[nodiscard]] std::vector<std::string>
    inboxMessages() const {
        std::vector<std::string> result;
        for (const auto &entry : inbox_) {
            result.push_back(entry.second);
        }
        return result;
    }

private:
    [[nodiscard]] static nixlAgentConfig
    senderConfig() {
        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        return cfg;
    }

    std::unique_ptr<testAgent> sender_;
    std::unique_ptr<testAgent> receiver_;
    std::vector<std::pair<std::string, std::string>> inbox_;
};

[[nodiscard]] nixlAgentConfig
receiverConfig(bool use_prog_thread) {
    nixlAgentConfig cfg;
    cfg.useProgThread = use_prog_thread;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    return cfg;
}

void
expectEachOnce(const std::vector<std::string> &callback_msgs,
               const std::vector<std::string> &inbox_msgs,
               size_t count) {
    std::multiset<std::string> seen(callback_msgs.begin(), callback_msgs.end());
    seen.insert(inbox_msgs.begin(), inbox_msgs.end());
    EXPECT_EQ(seen.size(), count);
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(seen.count(sequenceMessage(prefix, i)), 1) << "message " << i;
    }
}

struct mockCaps {
    bool remote = true;
    bool notif = true;
    bool callback = true;
};

class notifAgentApiTest : public testing::Test {
protected:
    [[nodiscard]] static nixlAgentConfig
    config() {
        nixlAgentConfig cfg;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        return cfg;
    }

    [[nodiscard]] nixl_status_t
    createMock(nixlAgent &agent, const mockCaps &caps) {
        ON_CALL(gmock_, supportsRemote()).WillByDefault(testing::Return(caps.remote));
        ON_CALL(gmock_, supportsNotif()).WillByDefault(testing::Return(caps.notif));
        ON_CALL(gmock_, supportsNotifCallback()).WillByDefault(testing::Return(caps.callback));
        nixl_b_params_t params;
        gmock_.SetToParams(params);
        nixlBackendH *backend = nullptr;
        return agent.createBackend(gtest::GetMockBackendName(), params, backend);
    }

    testing::NiceMock<mocks::GMockBackendEngine> gmock_;
};

class notifDeliveryTest : public testing::TestWithParam<bool> {
protected:
    [[nodiscard]] nixlAgentConfig
    config() const {
        return receiverConfig(GetParam());
    }
};

void
runConcurrently(size_t threads,
                const std::function<nixl_status_t(size_t)> &call,
                std::vector<nixl_status_t> &results) {
    results.assign(threads, NIXL_ERR_UNKNOWN);
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([&, i] {
            while (!go) {
                std::this_thread::yield();
            }
            results[i] = call(i);
        });
    }
    go = true;
    for (auto &worker : workers) {
        worker.join();
    }
}

[[nodiscard]] size_t
countEqual(const std::vector<nixl_status_t> &results, nixl_status_t status) {
    return std::count(results.begin(), results.end(), status);
}

} // namespace

TEST_F(notifAgentApiTest, RegisterSuccess) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(createMock(agent, {}), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, RegisterWithoutBackends) {
    nixlAgent agent("agent", config());
    EXPECT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, RegisterInvalidParam) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback("xyz", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback("", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("abc", nixl_notif_callback_t()), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("xyz", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("x", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.registerNotifCallback("xyzx", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST_F(notifAgentApiTest, RejectedRegisterChangesNothing) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback("xyz", noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.registerNotifCallback("x", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(agent.deregisterNotifCallback("x"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.deregisterNotifCallback("xyz"), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, DeregisterSuccess) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(agent.deregisterNotifCallback(prefix), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, DeregisterNotFound) {
    nixlAgent agent("agent", config());
    EXPECT_EQ(agent.deregisterNotifCallback(prefix), NIXL_ERR_NOT_FOUND);
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(agent.deregisterNotifCallback(""), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.deregisterNotifCallback("not"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.deregisterNotifCallback("notify"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent.deregisterNotifCallback(prefix), NIXL_SUCCESS);
    EXPECT_EQ(agent.deregisterNotifCallback(prefix), NIXL_ERR_NOT_FOUND);
}

TEST_F(notifAgentApiTest, DeregisterRemovesOnlyNamedPrefix) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback("aa", noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.registerNotifCallback("ab", noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterNotifCallback("aa"), NIXL_SUCCESS);
    EXPECT_EQ(agent.deregisterNotifCallback("ab"), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, RegisterAgainAfterDeregister) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterNotifCallback(prefix), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterNotifCallback(prefix), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback("not", noop_callback), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, RegistrationsArePerAgent) {
    nixlAgent agent1("agent1", config());
    nixlAgent agent2("agent2", config());
    EXPECT_EQ(agent1.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(agent2.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent1.registerNotifCallback("only1", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(agent2.deregisterNotifCallback("only1"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(agent1.deregisterNotifCallback("only1"), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, RegisterNotSupported) {
    const gtest::LogIgnoreGuard lig(unsupported_log);
    nixlAgent agent("agent", config());
    ASSERT_EQ(createMock(agent, {.callback = false}), NIXL_SUCCESS);
    EXPECT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(agent.deregisterNotifCallback(prefix), NIXL_ERR_NOT_FOUND);
}

TEST_F(notifAgentApiTest, CreateBackendFailsAfterRegister) {
    const gtest::LogIgnoreGuard lig(unsupported_log);
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(createMock(agent, {.callback = false}), NIXL_ERR_NOT_SUPPORTED);
}

TEST_F(notifAgentApiTest, CreateBackendAfterAllDeregistered) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    ASSERT_EQ(agent.deregisterNotifCallback(prefix), NIXL_SUCCESS);
    EXPECT_EQ(createMock(agent, {.callback = false}), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, LocalOnlyBackendUnaffected) {
    nixlAgent agent1("agent1", config());
    ASSERT_EQ(agent1.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(createMock(agent1, {.remote = false, .notif = false, .callback = false}),
              NIXL_SUCCESS);

    nixlAgent agent2("agent2", config());
    ASSERT_EQ(createMock(agent2, {.remote = false, .notif = false, .callback = false}),
              NIXL_SUCCESS);
    EXPECT_EQ(agent2.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
}

TEST_F(notifAgentApiTest, ConcurrentRegisterSamePrefixOneWinner) {
    nixlAgent agent("agent", config());
    std::vector<nixl_status_t> results;
    runConcurrently(
        8, [&](size_t) { return agent.registerNotifCallback(prefix, noop_callback); }, results);
    EXPECT_EQ(countEqual(results, NIXL_SUCCESS), 1);
    EXPECT_EQ(countEqual(results, NIXL_ERR_INVALID_PARAM), 7);
}

TEST_F(notifAgentApiTest, ConcurrentRegisterNestedPrefixesOneWinner) {
    nixlAgent agent("agent", config());
    std::vector<nixl_status_t> results;
    runConcurrently(
        8,
        [&](size_t i) {
            return agent.registerNotifCallback(std::string(i + 1, 'p'), noop_callback);
        },
        results);
    EXPECT_EQ(countEqual(results, NIXL_SUCCESS), 1);
    EXPECT_EQ(countEqual(results, NIXL_ERR_INVALID_PARAM), 7);
}

TEST_F(notifAgentApiTest, ConcurrentRegisterDisjointPrefixesAllSucceed) {
    nixlAgent agent("agent", config());
    std::vector<nixl_status_t> results;
    runConcurrently(
        8,
        [&](size_t i) {
            return agent.registerNotifCallback("p" + std::to_string(i) + "_", noop_callback);
        },
        results);
    EXPECT_EQ(countEqual(results, NIXL_SUCCESS), 8);
}

TEST_F(notifAgentApiTest, ConcurrentDeregisterOneWinner) {
    nixlAgent agent("agent", config());
    ASSERT_EQ(agent.registerNotifCallback(prefix, noop_callback), NIXL_SUCCESS);
    std::vector<nixl_status_t> results;
    runConcurrently(8, [&](size_t) { return agent.deregisterNotifCallback(prefix); }, results);
    EXPECT_EQ(countEqual(results, NIXL_SUCCESS), 1);
    EXPECT_EQ(countEqual(results, NIXL_ERR_NOT_FOUND), 7);
}

TEST_P(notifDeliveryTest, PlainNotifReachesGetNotifs) {
    ucxPair pair(config());
    pair.send(message);
    EXPECT_TRUE(pair.pollUntil([&] { return pair.inboxHas(message); }));
    ASSERT_EQ(pair.inbox().size(), 1);
    EXPECT_EQ(pair.inbox()[0].first, sender_name);
}

TEST_P(notifDeliveryTest, PrefixEqualLength) {
    callbackRecorder recorder;
    callbackRecorder others;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().registerNotifCallback("aaaaa", others.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().registerNotifCallback("zzzzz", others.callback()), NIXL_SUCCESS);
    pair.send(message);
    EXPECT_TRUE(pair.pollUntil([&] { return recorder.count() == 1; }));
    EXPECT_EQ(recorder.messages(), std::vector<std::string>{message});
    EXPECT_EQ(others.count(), 0);
    EXPECT_TRUE(pair.inbox().empty());
}

TEST_P(notifDeliveryTest, PrefixMixedLength) {
    callbackRecorder recorder;
    callbackRecorder others;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback("aaa", others.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().registerNotifCallback("zzzzzzz", others.callback()), NIXL_SUCCESS);
    pair.send(message);
    EXPECT_TRUE(pair.pollUntil([&] { return recorder.count() == 1; }));
    EXPECT_EQ(recorder.messages(), std::vector<std::string>{message});
    EXPECT_EQ(others.count(), 0);
    EXPECT_TRUE(pair.inbox().empty());
}

TEST_P(notifDeliveryTest, UnmatchedToGetNotifs) {
    callbackRecorder recorder;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    pair.send("other_message");
    EXPECT_TRUE(pair.pollUntil([&] { return pair.inboxHas("other_message"); }));
    EXPECT_EQ(recorder.count(), 0);
}

TEST_P(notifDeliveryTest, ShortMessageToGetNotifs) {
    callbackRecorder recorder;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    pair.send("not");
    EXPECT_TRUE(pair.pollUntil([&] { return pair.inboxHas("not"); }));
    EXPECT_EQ(recorder.count(), 0);
}

TEST_P(notifDeliveryTest, BytesIntact) {
    callbackRecorder recorder;
    const std::string binary_prefix{'\0', '\xff', 'p'};
    std::string payload = binary_prefix;
    for (int i = 0; i < 4096; ++i) {
        payload.push_back(static_cast<char>(i % 256));
    }
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(binary_prefix, recorder.callback()),
              NIXL_SUCCESS);
    pair.send(payload);
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() == 1; }));
    const auto calls = recorder.calls();
    EXPECT_EQ(calls[0].remoteAgent, sender_name);
    EXPECT_EQ(calls[0].notifMessage, payload);
}

TEST_P(notifDeliveryTest, OrderPreserved) {
    constexpr size_t count = 1000;
    callbackRecorder recorder;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    std::vector<std::string> expected;
    for (size_t i = 0; i < count; ++i) {
        expected.push_back(sequenceMessage(prefix, i));
        pair.send(expected.back());
    }
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() == count; }));
    EXPECT_EQ(recorder.messages(), expected);
}

TEST_P(notifDeliveryTest, OrderPreservedAcrossCallbackAndGetNotifs) {
    constexpr size_t count = 1000;
    callbackRecorder recorder;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    std::vector<std::string> expected_callback;
    std::vector<std::string> expected_inbox;
    for (size_t i = 0; i < count; ++i) {
        auto &expected = (i % 2 == 0) ? expected_callback : expected_inbox;
        expected.push_back(sequenceMessage((i % 2 == 0) ? prefix : "plain", i));
        pair.send(expected.back());
    }
    ASSERT_TRUE(pair.pollUntil([&] {
        return recorder.count() == expected_callback.size() &&
            pair.inbox().size() == expected_inbox.size();
    }));
    EXPECT_EQ(recorder.messages(), expected_callback);
    EXPECT_EQ(pair.inboxMessages(), expected_inbox);
}

TEST_P(notifDeliveryTest, CallbackThread) {
    callbackRecorder recorder;
    ucxPair pair(config());
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    pair.send(message);
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() == 1; }));
    const bool on_test_thread = recorder.calls()[0].thread == std::this_thread::get_id();
    EXPECT_EQ(on_test_thread, !GetParam());
}

INSTANTIATE_TEST_SUITE_P(ProgressThread,
                         notifDeliveryTest,
                         testing::Bool(),
                         [](const testing::TestParamInfo<bool> &info) {
                             return info.param ? "WithProgressThread" : "WithoutProgressThread";
                         });

TEST(NotifDynamic, BeforeRegisterToGetNotifs) {
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    pair.send(prefix + "_before");
    ASSERT_TRUE(pair.pollUntil([&] { return pair.inboxHas(prefix + "_before"); }));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    EXPECT_EQ(recorder.count(), 0);
}

TEST(NotifDynamic, DeregisterToGetNotifs) {
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().deregisterNotifCallback(prefix), NIXL_SUCCESS);
    pair.send(prefix + "_after");
    EXPECT_TRUE(pair.pollUntil([&] { return pair.inboxHas(prefix + "_after"); }));
    EXPECT_EQ(recorder.count(), 0);
}

TEST(NotifDynamic, ShorterPrefixTakesOverAfterDeregister) {
    callbackRecorder long_cb;
    callbackRecorder short_cb;
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback("notify", long_cb.callback()), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().deregisterNotifCallback("notify"), NIXL_SUCCESS);
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, short_cb.callback()), NIXL_SUCCESS);
    pair.send("notify_1");
    EXPECT_TRUE(pair.pollUntil([&] { return short_cb.count() == 1; }));
    EXPECT_EQ(long_cb.count(), 0);
}

TEST(NotifDynamic, CallbackCopiedAtRegistration) {
    callbackRecorder first;
    callbackRecorder second;
    ucxPair pair(receiverConfig(true));
    nixl_notif_callback_t callback = first.callback();
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, callback), NIXL_SUCCESS);
    callback = second.callback();
    pair.send(prefix + "_1");
    EXPECT_TRUE(pair.pollUntil([&] { return first.count() == 1; }));
    EXPECT_EQ(second.count(), 0);
}

TEST(NotifDynamic, RegisterDuringStreamExactlyOnce) {
    constexpr size_t count = 2000;
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    std::thread sender([&] {
        for (size_t i = 0; i < count; ++i) {
            pair.send(sequenceMessage(prefix, i));
        }
    });
    std::this_thread::sleep_for(poll_interval);
    EXPECT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    sender.join();
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() + pair.inbox().size() == count; }));
    expectEachOnce(recorder.messages(), pair.inboxMessages(), count);
}

TEST(NotifDynamic, DeregisterDuringStreamExactlyOnce) {
    constexpr size_t count = 2000;
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    std::thread sender([&] {
        for (size_t i = 0; i < count; ++i) {
            pair.send(sequenceMessage(prefix, i));
        }
    });
    std::this_thread::sleep_for(poll_interval);
    EXPECT_EQ(pair.receiver().deregisterNotifCallback(prefix), NIXL_SUCCESS);
    sender.join();
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() + pair.inbox().size() == count; }));
    expectEachOnce(recorder.messages(), pair.inboxMessages(), count);
}

TEST(NotifDynamic, ChurnNoLossNoDuplicate) {
    constexpr size_t count = 5000;
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    std::atomic<bool> done{false};
    std::thread sender([&] {
        for (size_t i = 0; i < count; ++i) {
            pair.send(sequenceMessage(prefix, i));
        }
        done = true;
    });
    while (!done) {
        if (pair.receiver().registerNotifCallback(prefix, recorder.callback()) != NIXL_SUCCESS ||
            pair.receiver().deregisterNotifCallback(prefix) != NIXL_SUCCESS) {
            ADD_FAILURE() << "register or deregister failed";
            break;
        }
    }
    sender.join();
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() + pair.inbox().size() == count; }));
    expectEachOnce(recorder.messages(), pair.inboxMessages(), count);
}

TEST(NotifDynamic, ChurnOfOtherPrefixDoesNotDisturbRegistered) {
    constexpr size_t count = 2000;
    callbackRecorder recorder;
    callbackRecorder other;
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix, recorder.callback()), NIXL_SUCCESS);
    std::atomic<bool> done{false};
    std::thread sender([&] {
        for (size_t i = 0; i < count; ++i) {
            pair.send(sequenceMessage(prefix, i));
        }
        done = true;
    });
    while (!done) {
        if (pair.receiver().registerNotifCallback("other", other.callback()) != NIXL_SUCCESS ||
            pair.receiver().deregisterNotifCallback("other") != NIXL_SUCCESS) {
            ADD_FAILURE() << "register or deregister failed";
            break;
        }
    }
    sender.join();
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() == count; }));
    EXPECT_TRUE(pair.inbox().empty());
    EXPECT_EQ(other.count(), 0);
}

TEST(NotifStress, SlowCallbackNoLoss) {
    constexpr size_t count = 50;
    callbackRecorder recorder;
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix,
                                                    [&](nixlNotifCallbackArgs &&args) {
                                                        if (recorder.count() == 0) {
                                                            std::this_thread::sleep_for(100ms);
                                                        }
                                                        recorder.record(std::move(args));
                                                    }),
              NIXL_SUCCESS);
    std::vector<std::string> expected;
    for (size_t i = 0; i < count; ++i) {
        expected.push_back(sequenceMessage(prefix, i));
        pair.send(expected.back());
    }
    ASSERT_TRUE(pair.pollUntil([&] { return recorder.count() == count; }));
    EXPECT_EQ(recorder.messages(), expected);
}

TEST(NotifStress, DeregisterWaitsForRunningCallback) {
    std::promise<void> entered;
    std::promise<void> release;
    auto release_future = release.get_future().share();
    std::atomic<size_t> calls{0};
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix,
                                                    [&](nixlNotifCallbackArgs &&) {
                                                        if (calls++ == 0) {
                                                            entered.set_value();
                                                            release_future.wait();
                                                        }
                                                    }),
              NIXL_SUCCESS);
    pair.send(sequenceMessage(prefix, 0));
    if (entered.get_future().wait_for(wait_time) != std::future_status::ready) {
        release.set_value();
        FAIL() << "the callback did not start";
    }
    auto deregistered = std::async(std::launch::async, [&] {
        return pair.receiver().deregisterNotifCallback(prefix);
    });
    EXPECT_EQ(deregistered.wait_for(quiet_time), std::future_status::timeout);
    release.set_value();
    ASSERT_EQ(deregistered.wait_for(wait_time), std::future_status::ready);
    EXPECT_EQ(deregistered.get(), NIXL_SUCCESS);
    EXPECT_TRUE(pair.sendSentinel(sequenceMessage(prefix, 1)));
    EXPECT_EQ(calls, 1);
}

TEST(NotifStress, DestroyWithMessagesInFlight) {
    const gtest::LogIgnoreGuard lig(".*");
    std::atomic<size_t> calls{0};
    std::atomic<bool> stop{false};
    ucxPair pair(receiverConfig(true));
    ASSERT_EQ(pair.receiver().registerNotifCallback(prefix,
                                                    [&](nixlNotifCallbackArgs &&) { ++calls; }),
              NIXL_SUCCESS);
    std::thread sender([&] {
        for (size_t i = 0; !stop; ++i) {
            pair.trySend(sequenceMessage(prefix, i));
        }
    });
    const bool started = pair.pollUntil([&] { return calls >= 100; });
    pair.destroyReceiver();
    const size_t after_destroy = calls;
    stop = true;
    sender.join();
    std::this_thread::sleep_for(quiet_time);
    EXPECT_TRUE(started);
    EXPECT_EQ(calls, after_destroy);
}

TEST(NotifStress, AgentDestructionReleasesCallback) {
    auto token = std::make_shared<int>(0);
    const std::weak_ptr<int> watch = token;
    {
        nixlAgent agent("agent", nixlAgentConfig());
        ASSERT_EQ(agent.registerNotifCallback(prefix,
                                              [token](nixlNotifCallbackArgs &&) { ++*token; }),
                  NIXL_SUCCESS);
        token.reset();
        EXPECT_FALSE(watch.expired());
    }
    EXPECT_TRUE(watch.expired());
}
