// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "runtime/runtime.h"
#include "utils/utils.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <sstream>

namespace {
class ReportingRT : public xferBenchRT {
public:
    int calls = 0;
    int contributors = 0;
    int destination = -1;
    int status = 0;
    double other_rates = 0;
    int copy_status = 0;
    int copy_peer = -1;
    std::string message;

    ReportingRT(int size, int rank) {
        setSize(size);
        setRank(rank);
    }

    int
    sendInt(int *, int) override {
        return 0;
    }

    int
    recvInt(int *, int) override {
        return 0;
    }

    int
    broadcastInt(int *, size_t, int) override {
        return 0;
    }

    int
    sendChar(char *buffer, size_t count, int peer) override {
        message.assign(buffer, count);
        copy_peer = peer;
        return copy_status;
    }

    int
    recvChar(char *buffer, size_t count, int peer) override {
        std::copy_n(message.begin(), std::min(count, message.size()), buffer);
        copy_peer = peer;
        return copy_status;
    }

    int
    barrier(const std::string &, bool) override {
        return 0;
    }

    int
    reduceSumDouble(double *local, double *global, int dest, int count) override {
        ++calls;
        contributors = count;
        destination = dest;
        *global = *local + other_rates;
        return status;
    }
};

class AggregateStatsTest : public ::testing::Test {
    std::string mode = xferBenchConfig::mode;
    std::string scheme = xferBenchConfig::scheme;
    int senders = xferBenchConfig::num_initiator_dev;
    int iterations = xferBenchConfig::num_iter;
    int threads = xferBenchConfig::num_threads;

protected:
    void
    SetUp() override {
        xferBenchConfig::mode = XFERBENCH_MODE_SG;
        xferBenchConfig::scheme = XFERBENCH_SCHEME_MANY_TO_ONE;
        xferBenchConfig::num_initiator_dev = 3;
        xferBenchConfig::num_iter = 1000;
        xferBenchConfig::num_threads = 1;
    }

    void
    TearDown() override {
        xferBenchUtils::setRT(nullptr);
        xferBenchConfig::mode = mode;
        xferBenchConfig::scheme = scheme;
        xferBenchConfig::num_initiator_dev = senders;
        xferBenchConfig::num_iter = iterations;
        xferBenchConfig::num_threads = threads;
    }

    std::pair<bool, std::string>
    report(ReportingRT &rt, bool target = false) {
        xferBenchUtils::setRT(&rt);
        xferBenchStats stats;
        stats.total_duration.add(1e6); // 1000 x 1 MB in one second: 1 GB/s.
        stats.prepare_duration.add(1);
        stats.post_duration.add(1);
        stats.transfer_duration.add(1);
        testing::internal::CaptureStdout();
        const bool success = xferBenchUtils::printStats(target, 1000000, 1, stats);
        return {success, testing::internal::GetCapturedStdout()};
    }
};

TEST_F(AggregateStatsTest, IncastSumsAllSendersOnRankZeroAndPreservesTenColumns) {
    for (const int count : {1, 2, 3}) {
        xferBenchConfig::num_initiator_dev = count;
        ReportingRT rt(count + 1, 0);
        rt.other_rates = count == 3 ? 6 : count == 2 ? 2 : 0;
        const auto [success, text] = report(rt);
        ASSERT_TRUE(success);
        EXPECT_EQ(rt.calls, count == 1 ? 0 : 1);
        if (count > 1) {
            EXPECT_EQ(rt.contributors, count);
            EXPECT_EQ(rt.destination, 0);
        }
        std::istringstream lines(text);
        std::string row;
        std::getline(lines, row);
        std::istringstream fields(row);
        std::vector<double> numbers;
        double number;
        while (fields >> number) {
            numbers.push_back(number);
        }
        ASSERT_EQ(numbers.size(), 10);
        EXPECT_DOUBLE_EQ(numbers[2], 1);
        EXPECT_NE(
            text.find("Sum of sender bandwidths (GB/Sec): " + std::to_string(1 + rt.other_rates)),
            std::string::npos);
        EXPECT_NE(text.find("senders=" + std::to_string(count) + ", rank=0]"), std::string::npos);
        EXPECT_EQ(text.find("Network Util"), std::string::npos);
        EXPECT_EQ(rt.copy_peer, count);
        ReportingRT receiver(count + 1, count);
        receiver.message = rt.message;
        const auto [copied, target_text] = report(receiver, true);
        ASSERT_TRUE(copied);
        EXPECT_EQ(receiver.calls, 0);
        EXPECT_EQ(receiver.copy_peer, 0);
        const auto root_summary = text.substr(text.find("Sum of sender bandwidths"));
        std::string expected = root_summary;
        expected.replace(expected.find("rank=0"), 6, "rank=" + std::to_string(count));
        EXPECT_EQ(target_text, expected);
    }
}

TEST_F(AggregateStatsTest, OtherSenderKeepsItsOwnRowWithoutSummary) {
    ReportingRT rt(4, 1);
    const auto [success, text] = report(rt);
    EXPECT_TRUE(success);
    EXPECT_EQ(rt.calls, 1);
    EXPECT_NE(text.find("1.000000"), std::string::npos);
    EXPECT_EQ(text.find("Sum of sender"), std::string::npos);
}

TEST_F(AggregateStatsTest, PairwiseTargetDoesNotContributeOrPrint) {
    xferBenchConfig::scheme = XFERBENCH_SCHEME_PAIRWISE;
    ReportingRT rt(4, 3);
    const auto [success, text] = report(rt, true);
    EXPECT_TRUE(success);
    EXPECT_EQ(rt.calls, 0);
    EXPECT_TRUE(text.empty());
}

TEST_F(AggregateStatsTest, FailedSummarySendOrReceiveProducesNoOutput) {
    for (const bool target : {false, true}) {
        ReportingRT rt(4, target ? 3 : 0);
        rt.copy_status = -1;
        const auto [success, text] = report(rt, target);
        EXPECT_FALSE(success);
        EXPECT_TRUE(text.empty());
    }
}

TEST_F(AggregateStatsTest, ReceiverRejectsMalformedOrWrongCaseSummary) {
    for (const std::string message :
         {"incast-summary-v1 1000000 1 3 7", // Truncated completion marker.
          "incast-summary-v1 2000000 1 3 7 end-incast-summary",
          "incast-summary-v1 1000000 2 3 7 end-incast-summary",
          "incast-summary-v1 1000000 1 2 7 end-incast-summary",
          "incast-summary-v1 1000000 1 3 -1 end-incast-summary",
          "incast-summary-v1 1000000 1 3 nan end-incast-summary",
          "incast-summary-v1 1000000 1 3 inf end-incast-summary",
          "incast-summary-v2 1000000 1 3 7 end-incast-summary",
          "incast-summary-v1 1000000 1 3 7 end-incast-summary extra"}) {
        ReportingRT rt(4, 3);
        rt.message = message;
        const auto [success, text] = report(rt, true);
        EXPECT_FALSE(success) << message;
        EXPECT_TRUE(text.empty());
    }
}

TEST_F(AggregateStatsTest, FailedReductionProducesNoResultRow) {
    ReportingRT rt(4, 0);
    rt.status = -1;
    rt.other_rates = 2; // Even a plausible partial sum must not be reported.
    const auto [success, text] = report(rt);
    EXPECT_FALSE(success);
    EXPECT_TRUE(text.empty());
}

TEST_F(AggregateStatsTest, PairwiseKeepsItsTwelveColumnFormat) {
    xferBenchConfig::scheme = XFERBENCH_SCHEME_PAIRWISE;
    ReportingRT rt(6, 0);
    rt.other_rates = 6;
    const auto [success, text] = report(rt);
    EXPECT_TRUE(success);
    EXPECT_EQ(rt.contributors, 3);
    std::istringstream fields(text);
    std::vector<double> numbers;
    double number;
    while (fields >> number) {
        numbers.push_back(number);
    }
    ASSERT_EQ(numbers.size(), 12);
    EXPECT_DOUBLE_EQ(numbers[2], 1);
    EXPECT_DOUBLE_EQ(numbers[3], 7);
}
} // namespace
