/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/runtime.h"
#include "utils/utils.h"

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>

namespace {

class FakeRT : public xferBenchRT {
public:
    FakeRT(int size, int rank) {
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
    sendChar(char *, size_t, int) override {
        return 0;
    }

    int
    recvChar(char *, size_t, int) override {
        return 0;
    }

    int
    barrier(const std::string &, const bool) override {
        return 0;
    }

    int
    reduceSumDouble(double *local, double *global, int) override {
        *global = *local;
        return 0;
    }
};

// B/W printed by an initiator's printStats() for 1000 iterations of one 1 MB block in 1 s. Without
// any scaling that is 1 GB/s.
double
reportedBandwidth(const std::string &mode, const std::string &scheme, int devices, int ranks) {
    const std::string saved_mode = xferBenchConfig::mode;
    const std::string saved_scheme = xferBenchConfig::scheme;
    const int saved_initiators = xferBenchConfig::num_initiator_dev;
    const int saved_iter = xferBenchConfig::num_iter;
    const int saved_threads = xferBenchConfig::num_threads;

    xferBenchConfig::mode = mode;
    xferBenchConfig::scheme = scheme;
    xferBenchConfig::num_initiator_dev = devices;
    xferBenchConfig::num_iter = 1000;
    xferBenchConfig::num_threads = 1;

    FakeRT rt(ranks, 0);
    xferBenchUtils::setRT(&rt);

    xferBenchStats stats;
    stats.total_duration.add(1e6);
    stats.prepare_duration.add(1);
    stats.post_duration.add(1);
    stats.transfer_duration.add(1);

    std::ostringstream output;
    std::streambuf *const saved_cout = std::cout.rdbuf(output.rdbuf());
    xferBenchUtils::printStats(false, 1000000, 1, stats);
    std::cout.rdbuf(saved_cout);
    std::istringstream row(output.str());

    xferBenchUtils::setRT(nullptr);
    xferBenchConfig::mode = saved_mode;
    xferBenchConfig::scheme = saved_scheme;
    xferBenchConfig::num_initiator_dev = saved_initiators;
    xferBenchConfig::num_iter = saved_iter;
    xferBenchConfig::num_threads = saved_threads;

    size_t block_size = 0, batch_size = 0;
    double bandwidth = 0;
    row >> block_size >> batch_size >> bandwidth;
    return bandwidth;
}

// An MG manytoone initiator posts one descriptor per device per batch entry, like MG pairwise, so
// its single printed B/W must cover all of its devices.
TEST(PrintStatsTest, MgManyToOneReportsAllInitiatorDevices) {
    EXPECT_DOUBLE_EQ(reportedBandwidth("MG", "manytoone", 1, 2), 1.0);
    EXPECT_DOUBLE_EQ(reportedBandwidth("MG", "manytoone", 4, 2), 4.0);
    EXPECT_DOUBLE_EQ(reportedBandwidth("MG", "manytoone", 4, 2),
                     reportedBandwidth("MG", "pairwise", 4, 2));
}

TEST(PrintStatsTest, MgPairwiseReportsAllInitiatorDevices) {
    EXPECT_DOUBLE_EQ(reportedBandwidth("MG", "pairwise", 4, 2), 4.0);
}

// Multi-rank SG reports per rank and must not be scaled by num_initiator_dev (#1463).
TEST(PrintStatsTest, MultiRankSgPairwiseIsNotScaled) {
    EXPECT_DOUBLE_EQ(reportedBandwidth("SG", "pairwise", 2, 4), 1.0);
}

} // namespace
