// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "runtime/etcd/etcd_rt.h"
#include <etcd/SyncClient.hpp>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <thread>
#include <unistd.h>

namespace {
class EtcdReductionTest : public ::testing::Test {
protected:
    std::string endpoint;
    std::string group;
    std::atomic<int> terminate{0};
    std::vector<std::unique_ptr<xferBenchEtcdRT>> ranks;

    void
    SetUp() override {
        const char *value = std::getenv("NIXLBENCH_ETCD_TEST_ENDPOINT");
        if (!value) {
            GTEST_SKIP() << "Set NIXLBENCH_ETCD_TEST_ENDPOINT to an isolated local etcd";
        }
        endpoint = value;
        group = "reduction-test-" + std::to_string(getpid()) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void
    setupRanks(int count) {
        for (int rank = 0; rank < count; ++rank) {
            ranks.push_back(std::make_unique<xferBenchEtcdRT>(group, endpoint, count, &terminate));
            ASSERT_EQ(ranks.back()->setup(), 0);
            ASSERT_EQ(ranks.back()->getRank(), rank);
        }
    }

    int
    contribute(int rank, double value, double &sum, int count = 0) {
        return ranks[rank]->reduceSumDouble(&value, &sum, 0, count);
    }
};

TEST_F(EtcdReductionTest, OneSenderDoesNotWaitForTarget) {
    setupRanks(2);
    double sum = -1;
    ASSERT_EQ(contribute(0, 1.25, sum, 1), 0);
    EXPECT_DOUBLE_EQ(sum, 1.25);
}

TEST_F(EtcdReductionTest, FastContributorsCannotOverwriteLaterRounds) {
    setupRanks(4); // Three senders and one target, which never contributes.
    double unused = -1;
    for (int rank = 1; rank < 3; ++rank) {
        for (double scale : {1., 10., 100.}) {
            ASSERT_EQ(contribute(rank, (rank == 1 ? 2 : 4) * scale, unused, 3), 0);
        }
    }
    for (double scale : {1., 10., 100.}) {
        double sum = -1;
        ASSERT_EQ(contribute(0, scale, sum, 3), 0);
        EXPECT_DOUBLE_EQ(sum, 7 * scale);
    }
}

TEST_F(EtcdReductionTest, CollectorWaitsForDelayedSender) {
    setupRanks(3);
    std::thread delayed([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        double unused = -1;
        EXPECT_EQ(contribute(1, 9, unused, 2), 0);
    });
    double sum = -1;
    const int status = contribute(0, 1, sum, 2);
    delayed.join();
    EXPECT_EQ(status, 0);
    EXPECT_DOUBLE_EQ(sum, 10);
}

TEST_F(EtcdReductionTest, InterruptedMissingContributionDoesNotPublishPartialSum) {
    setupRanks(4);
    double unused = -1;
    ASSERT_EQ(contribute(1, 2, unused, 3), 0);
    std::thread cancel([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        terminate = 1;
    });
    double sum = -123;
    const int status = contribute(0, 1, sum, 3);
    const bool was_cancelled = terminate.load() != 0;
    cancel.join();
    EXPECT_TRUE(was_cancelled);
    EXPECT_NE(status, 0);
    EXPECT_DOUBLE_EQ(sum, -123);
}

TEST_F(EtcdReductionTest, LostPeerFailsWithoutPartialSum) {
    setupRanks(3);
    etcd::SyncClient client(endpoint);
    ASSERT_TRUE(client.rm("xferbench/" + group + "//rank/1").is_ok());
    double sum = -123;
    EXPECT_NE(contribute(0, 1, sum, 2), 0);
    EXPECT_DOUBLE_EQ(sum, -123);
}

TEST_F(EtcdReductionTest, MalformedContributionFailsWithoutPartialSum) {
    setupRanks(3);
    etcd::SyncClient client(endpoint);
    ASSERT_TRUE(client.put("xferbench/" + group + "//reduce/0/dest-0/rank-1", "2invalid").is_ok());
    double sum = -123;
    EXPECT_NE(contribute(0, 1, sum, 2), 0);
    EXPECT_DOUBLE_EQ(sum, -123);
}

TEST_F(EtcdReductionTest, LegacyPairwiseParticipantCountIsPreserved) {
    setupRanks(4);
    double sum = -1;
    ASSERT_EQ(contribute(1, 2, sum), 0);
    ASSERT_EQ(contribute(0, 1, sum), 0);
    EXPECT_DOUBLE_EQ(sum, 3);
}

TEST_F(EtcdReductionTest, LegacyTwoRankCallIncludesBothRanks) {
    setupRanks(2);
    double sum = -1;
    ASSERT_EQ(contribute(1, 2, sum), 0);
    ASSERT_EQ(contribute(0, 1, sum), 0);
    EXPECT_DOUBLE_EQ(sum, 3);
}
} // namespace
