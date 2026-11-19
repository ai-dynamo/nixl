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

#include "ucx_split_failure.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <latch>
#include <thread>
#include <vector>

namespace {

constexpr std::array<nixl_status_t, 4> publisher_statuses{NIXL_ERR_BACKEND,
                                                          NIXL_ERR_REMOTE_DISCONNECT,
                                                          NIXL_ERR_INVALID_PARAM,
                                                          NIXL_ERR_NOT_FOUND};
constexpr std::array<ucs_status_t, 4> publisher_natives{UCS_ERR_NO_MEMORY,
                                                        UCS_ERR_CONNECTION_RESET,
                                                        UCS_ERR_INVALID_PARAM,
                                                        UCS_ERR_ENDPOINT_TIMEOUT};

} // namespace

TEST(UcxSplitFailure, FirstPublicationWins) {
    nixlUcxSplitFailure failure;
    EXPECT_TRUE(failure.publish(NIXL_ERR_REMOTE_DISCONNECT, UCS_ERR_CONNECTION_RESET));
    EXPECT_FALSE(failure.publish(NIXL_ERR_BACKEND, UCS_ERR_NO_MEMORY));
    EXPECT_EQ(failure.status(), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(failure.native(), UCS_ERR_CONNECTION_RESET);
}

TEST(UcxSplitFailure, ResetClearsTheFailureAndTheSubmission) {
    nixlUcxSplitFailure failure;
    ASSERT_TRUE(failure.publish(NIXL_ERR_BACKEND, UCS_ERR_NO_MEMORY));
    failure.finishSubmission(NIXL_ERR_BACKEND);
    failure.reset();
    EXPECT_EQ(failure.status(), NIXL_SUCCESS);
    EXPECT_EQ(failure.native(), UCS_OK);
    EXPECT_EQ(failure.decide(), nixlUcxSplitFailure::decision_t::WAIT);
    EXPECT_TRUE(failure.publish(NIXL_ERR_REMOTE_DISCONNECT, UCS_ERR_CONNECTION_RESET));
}

TEST(UcxSplitFailure, FirstChunkWaitsUntilTheSubmissionIsKnown) {
    nixlUcxSplitFailure failure;
    EXPECT_EQ(failure.decide(), nixlUcxSplitFailure::decision_t::WAIT);
    failure.finishSubmission(NIXL_SUCCESS);
    EXPECT_EQ(failure.decide(), nixlUcxSplitFailure::decision_t::RECORD);
    failure.reset();
    failure.finishSubmission(NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(failure.decide(), nixlUcxSplitFailure::decision_t::DROP);
}

TEST(UcxSplitFailure, FailedSubmissionReturnsTheClaimedStatusOrItsOwn) {
    nixlUcxSplitFailure failure;
    EXPECT_EQ(failure.submissionStatus(NIXL_SUCCESS), NIXL_SUCCESS);
    EXPECT_EQ(failure.submissionStatus(NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
    ASSERT_TRUE(failure.publish(NIXL_ERR_REMOTE_DISCONNECT, UCS_ERR_CONNECTION_RESET));
    EXPECT_EQ(failure.submissionStatus(NIXL_ERR_BACKEND), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(failure.submissionStatus(NIXL_SUCCESS), NIXL_SUCCESS);
}

TEST(UcxSplitFailure, ConcurrentPublicationsLeaveOnePublishersPair) {
    constexpr std::size_t publishers = publisher_statuses.size();
    nixlUcxSplitFailure failure;
    for (int round = 0; round < 100; ++round) {
        failure.reset();
        std::latch start(publishers);
        std::array<bool, publishers> won{};
        std::vector<std::thread> threads;
        threads.reserve(publishers);
        for (std::size_t i = 0; i < publishers; ++i) {
            threads.emplace_back([&, i] {
                start.arrive_and_wait();
                won[i] = failure.publish(publisher_statuses[i], publisher_natives[i]);
            });
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
        ASSERT_EQ(std::ranges::count(won, true), 1) << "round " << round;
        const auto winner = static_cast<std::size_t>(std::ranges::find(won, true) - won.begin());
        EXPECT_EQ(failure.status(), publisher_statuses[winner]) << "round " << round;
        EXPECT_EQ(failure.native(), publisher_natives[winner]) << "round " << round;
    }
}
