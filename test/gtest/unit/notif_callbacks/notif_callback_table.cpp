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
#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "backend/notif_callbacks.h"

namespace notif_callbacks {
namespace {

using namespace std::chrono_literals;

const nixl_notif_callback_t noop_callback([](nixlNotifCallbackArgs &&) {});

[[nodiscard]] bool
isPrefixOf(const std::string &prefix, const std::string &message) {
    return message.size() >= prefix.size() && message.compare(0, prefix.size(), prefix) == 0;
}

[[nodiscard]] std::string
randomString(std::mt19937 &rng, size_t min_size, size_t max_size) {
    static const std::string alphabet{'a', 'b', '\0', '\xff'};
    std::uniform_int_distribution<size_t> size_dist(min_size, max_size);
    std::uniform_int_distribution<size_t> char_dist(0, alphabet.size() - 1);
    std::string result(size_dist(rng), ' ');
    for (auto &c : result) {
        c = alphabet[char_dist(rng)];
    }
    return result;
}

class labelledRouter {
public:
    explicit labelledRouter(const std::vector<std::string> &prefixes) {
        for (const auto &prefix : prefixes) {
            EXPECT_EQ(router_.add(prefix, labelCallback(prefix)), NIXL_SUCCESS) << prefix;
        }
    }

    [[nodiscard]] std::optional<std::string>
    route(const std::string &message) {
        takenBy_.reset();
        nixlNotifCallbackArgs args{"peer", message};
        const nixl_notif_route_t route = router_.call(args);
        EXPECT_EQ(route == nixl_notif_route_t::NIXL_NOTIF_ROUTE_CALLBACK, takenBy_.has_value());
        return takenBy_;
    }

    [[nodiscard]] nixl_status_t
    add(const std::string &prefix) {
        return router_.add(prefix, labelCallback(prefix));
    }

    [[nodiscard]] nixl_status_t
    remove(const std::string &prefix) {
        return router_.remove(prefix);
    }

private:
    [[nodiscard]] nixl_notif_callback_t
    labelCallback(const std::string &label) {
        return [this, label](nixlNotifCallbackArgs &&) { takenBy_ = label; };
    }

    std::optional<std::string> takenBy_;
    nixlNotifRouter router_;
};

} // namespace

TEST(NotifRouter, AddSuccess) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add("notif", noop_callback), NIXL_SUCCESS);
}

TEST(NotifRouter, AddRejectsEmptyPrefix) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add("", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, AddRejectsEmptyCallback) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add("notif", nixl_notif_callback_t()), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, AddRejectsDuplicate) {
    nixlNotifRouter router;
    ASSERT_EQ(router.add("notif", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("notif", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, AddRejectsOverlapShortFirst) {
    nixlNotifRouter router;
    ASSERT_EQ(router.add("x", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("xy", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(router.add("xyz", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, AddRejectsOverlapLongFirst) {
    nixlNotifRouter router;
    ASSERT_EQ(router.add("xyz", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("x", noop_callback), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(router.add("xy", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, NeighboursAccepted) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add("ab", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("ac", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("b", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.add("a", noop_callback), NIXL_ERR_INVALID_PARAM);
}

TEST(NotifRouter, PrefixLengthLimits) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add("a", noop_callback), NIXL_SUCCESS);
    GTEST_SKIP() << "The maximum prefix length is not decided";
}

TEST(NotifRouter, BinaryPrefixAccepted) {
    nixlNotifRouter router;
    EXPECT_EQ(router.add(std::string{'\0', '\xff', 'a'}, noop_callback), NIXL_SUCCESS);
}

TEST(NotifRouter, FailedAddChangesNothing) {
    labelledRouter router({"xyz"});
    ASSERT_EQ(router.add("x"), NIXL_ERR_INVALID_PARAM);
    EXPECT_EQ(router.route("xyz_1"), "xyz");
    EXPECT_EQ(router.route("x_1"), std::nullopt);
}

TEST(NotifRouter, RemoveSuccess) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.remove("notif"), NIXL_SUCCESS);
    EXPECT_EQ(router.route("notif_1"), std::nullopt);
}

TEST(NotifRouter, RemoveUnknownNotFound) {
    nixlNotifRouter router;
    EXPECT_EQ(router.remove("notif"), NIXL_ERR_NOT_FOUND);
    ASSERT_EQ(router.add("notif", noop_callback), NIXL_SUCCESS);
    EXPECT_EQ(router.remove("other"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(router.remove("not"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(router.remove("notify"), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(router.remove(""), NIXL_ERR_NOT_FOUND);
    EXPECT_EQ(router.remove("notif"), NIXL_SUCCESS);
    EXPECT_EQ(router.remove("notif"), NIXL_ERR_NOT_FOUND);
}

TEST(NotifRouter, RemoveOnlyNamedPrefix) {
    labelledRouter router({"aa", "ab"});
    ASSERT_EQ(router.remove("aa"), NIXL_SUCCESS);
    EXPECT_EQ(router.route("aa_1"), std::nullopt);
    EXPECT_EQ(router.route("ab_1"), "ab");
}

TEST(NotifRouter, AddAgainAfterRemove) {
    labelledRouter router({"notify"});
    ASSERT_EQ(router.remove("notify"), NIXL_SUCCESS);
    ASSERT_EQ(router.add("notif"), NIXL_SUCCESS);
    EXPECT_EQ(router.route("notify_1"), "notif");
}

TEST(NotifRouter, EmptyFollowsContent) {
    nixlNotifRouter router;
    EXPECT_TRUE(router.empty());
    ASSERT_EQ(router.add("notif", noop_callback), NIXL_SUCCESS);
    EXPECT_FALSE(router.empty());
    ASSERT_EQ(router.remove("notif"), NIXL_SUCCESS);
    EXPECT_TRUE(router.empty());
}

TEST(NotifRouter, PrefixMatch) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route("notification_message_1"), "notif");
}

TEST(NotifRouter, ExactMatch) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route("notif"), "notif");
}

TEST(NotifRouter, ShortMessageNoMatch) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route("not"), std::nullopt);
}

TEST(NotifRouter, EmptyMessageNoMatch) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route(""), std::nullopt);
}

TEST(NotifRouter, EmptyRouterNoMatch) {
    labelledRouter router({});
    EXPECT_EQ(router.route("notif_1"), std::nullopt);
}

TEST(NotifRouter, OutsideRangeNoMatch) {
    labelledRouter router({"b", "c"});
    EXPECT_EQ(router.route("a"), std::nullopt);
    EXPECT_EQ(router.route("d"), std::nullopt);
}

TEST(NotifRouter, BetweenNeighboursNoMatch) {
    labelledRouter router({"aa", "ac"});
    EXPECT_EQ(router.route("ab"), std::nullopt);
    EXPECT_EQ(router.route("abz"), std::nullopt);
}

TEST(NotifRouter, PrefixInsideMessageNoMatch) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route("xnotif"), std::nullopt);
}

TEST(NotifRouter, MatchIsCaseSensitive) {
    labelledRouter router({"notif"});
    EXPECT_EQ(router.route("NOTIF"), std::nullopt);
    EXPECT_EQ(router.route("Notif"), std::nullopt);
}

TEST(NotifRouter, BinaryPrefixMatch) {
    const std::string prefix{'\0', '\xff', 'a'};
    labelledRouter router({prefix});
    EXPECT_EQ(router.route(prefix + std::string{'\0', 'z'}), prefix);
    EXPECT_EQ(router.route(std::string{'\0', '\xff', 'b'}), std::nullopt);
}

TEST(NotifRouter, MatchMovesArgsIntoCallback) {
    nixlNotifRouter router;
    nixlNotifCallbackArgs received;
    ASSERT_EQ(router.add("notif",
                         [&](nixlNotifCallbackArgs &&args) { received = std::move(args); }),
              NIXL_SUCCESS);
    nixlNotifCallbackArgs args{"peer", "notif_1"};
    EXPECT_EQ(router.call(args), nixl_notif_route_t::NIXL_NOTIF_ROUTE_CALLBACK);
    EXPECT_EQ(received.remoteAgent, "peer");
    EXPECT_EQ(received.notifMessage, "notif_1");
}

TEST(NotifRouter, UnmatchedLeavesArgs) {
    nixlNotifRouter router;
    ASSERT_EQ(router.add("notif", noop_callback), NIXL_SUCCESS);
    nixlNotifCallbackArgs args{"peer", "other_1"};
    EXPECT_EQ(router.call(args), nixl_notif_route_t::NIXL_NOTIF_ROUTE_UNMATCHED);
    EXPECT_EQ(args.remoteAgent, "peer");
    EXPECT_EQ(args.notifMessage, "other_1");
}

TEST(NotifRouter, MatchesReferenceLoop) {
    std::mt19937 rng(1455);
    for (int round = 0; round < 200; ++round) {
        std::vector<std::string> prefixes;
        for (int attempt = 0; attempt < 6; ++attempt) {
            const std::string candidate = randomString(rng, 1, 4);
            bool overlaps = false;
            for (const auto &prefix : prefixes) {
                overlaps = overlaps || isPrefixOf(prefix, candidate) || isPrefixOf(candidate, prefix);
            }
            if (!overlaps) {
                prefixes.push_back(candidate);
            }
        }
        labelledRouter router(prefixes);
        for (int i = 0; i < 200; ++i) {
            const std::string message = randomString(rng, 0, 6);
            std::optional<std::string> expected;
            for (const auto &prefix : prefixes) {
                if (isPrefixOf(prefix, message)) {
                    expected = prefix;
                }
            }
            ASSERT_EQ(router.route(message), expected) << "round " << round;
        }
    }
}

TEST(NotifRouter, ConcurrentCallsDuringChanges) {
    std::atomic<size_t> callbacks{0};
    std::atomic<size_t> unmatched{0};
    std::atomic<size_t> lookups{0};
    std::atomic<bool> stop{false};
    nixlNotifRouter router;
    const nixl_notif_callback_t counting([&](nixlNotifCallbackArgs &&) { ++callbacks; });

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&] {
            while (!stop) {
                nixlNotifCallbackArgs args{"peer", "notif_1"};
                if (router.call(args) == nixl_notif_route_t::NIXL_NOTIF_ROUTE_UNMATCHED) {
                    ++unmatched;
                }
                ++lookups;
            }
        });
    }
    for (int i = 0; i < 2000; ++i) {
        EXPECT_EQ(router.add("notif", counting), NIXL_SUCCESS);
        EXPECT_EQ(router.remove("notif"), NIXL_SUCCESS);
    }
    stop = true;
    for (auto &reader : readers) {
        reader.join();
    }
    EXPECT_EQ(callbacks + unmatched, lookups);
}

TEST(NotifRouter, RemoveWaitsForRunningCallback) {
    std::promise<void> entered;
    std::promise<void> release;
    auto release_future = release.get_future().share();
    nixlNotifRouter router;
    ASSERT_EQ(router.add("notif",
                         [&](nixlNotifCallbackArgs &&) {
                             entered.set_value();
                             release_future.wait();
                         }),
              NIXL_SUCCESS);

    std::thread receiver([&] {
        nixlNotifCallbackArgs args{"peer", "notif_1"};
        EXPECT_EQ(router.call(args), nixl_notif_route_t::NIXL_NOTIF_ROUTE_CALLBACK);
    });
    entered.get_future().wait();
    auto removed = std::async(std::launch::async, [&] { return router.remove("notif"); });
    EXPECT_EQ(removed.wait_for(200ms), std::future_status::timeout);
    release.set_value();
    receiver.join();
    EXPECT_EQ(removed.get(), NIXL_SUCCESS);
}

} // namespace notif_callbacks
