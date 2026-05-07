/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/benchmark_executor.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace nixlbench {
namespace {

class recordingSync : public benchmarkRuntimeSync {
public:
    explicit
    recordingSync(std::vector<std::string> &events)
        : events_(events) {}

    int
    synchronizeStart() override {
        events_.push_back("sync_start");
        return nextReturn("sync_start");
    }

    int
    beforeTransfer() override {
        events_.push_back("before_transfer");
        return nextReturn("before_transfer");
    }

    int
    afterTransfer() override {
        events_.push_back("after_transfer");
        return nextReturn("after_transfer");
    }

    int
    finish() override {
        events_.push_back("finish");
        return nextReturn("finish");
    }

    std::string failing_event;
    int failure_code = EXIT_FAILURE;

private:
    int
    nextReturn(const std::string &event) const {
        return event == failing_event ? failure_code : EXIT_SUCCESS;
    }

    std::vector<std::string> &events_;
};

class recordingAllocator : public benchmarkMemoryAllocator {
public:
    explicit
    recordingAllocator(std::vector<std::string> &events)
        : events_(events) {}

    std::variant<benchmarkAllocation, int>
    allocate() override {
        events_.push_back("allocate");
        if (allocate_result != EXIT_SUCCESS) {
            return allocate_result;
        }
        return benchmarkAllocation{{{xferBenchIOV(1, 8, 0)}}};
    }

    void
    deallocate(benchmarkAllocation &allocation) override {
        events_.push_back("deallocate");
        deallocated_iov_count = allocation.local_iovs.empty() ? 0 : allocation.local_iovs[0].size();
    }

    int allocate_result = EXIT_SUCCESS;
    size_t deallocated_iov_count = 0;

private:
    std::vector<std::string> &events_;
};

class recordingDescriptorStrategy : public transferDescriptorStrategy {
public:
    explicit
    recordingDescriptorStrategy(std::vector<std::string> &events)
        : events_(events) {}

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(const benchmarkAllocation &allocation) override {
        events_.push_back("descriptors");
        if (create_result != EXIT_SUCCESS) {
            return create_result;
        }
        return allocation.local_iovs;
    }

    int create_result = EXIT_SUCCESS;

private:
    std::vector<std::string> &events_;
};

class recordingTransferStrategy : public benchmarkTransferStrategy {
public:
    explicit
    recordingTransferStrategy(std::vector<std::string> &events)
        : events_(events) {}

    std::variant<xferBenchStats, int>
    execute(const std::vector<std::vector<xferBenchIOV>> &descriptors) override {
        events_.push_back("transfer");
        descriptor_count = descriptors.empty() ? 0 : descriptors[0].size();
        if (execute_result != EXIT_SUCCESS) {
            return execute_result;
        }
        return xferBenchStats();
    }

    int execute_result = EXIT_SUCCESS;
    size_t descriptor_count = 0;

private:
    std::vector<std::string> &events_;
};

class fixedIterationPolicy : public iterationPolicy {
public:
    explicit
    fixedIterationPolicy(int iterations)
        : remaining_(iterations) {}

    bool
    hasNext() const override {
        return remaining_ > 0;
    }

    void
    advance() override {
        --remaining_;
    }

private:
    int remaining_;
};

class recordingResultSink : public benchmarkResultSink {
public:
    explicit
    recordingResultSink(std::vector<std::string> &events)
        : events_(events) {}

    void
    record(const xferBenchStats &stats) override {
        (void)stats;
        events_.push_back("record");
        ++record_count;
    }

    int record_count = 0;

private:
    std::vector<std::string> &events_;
};

benchmarkRunComponents
makeComponents(recordingSync &sync,
               recordingAllocator &allocator,
               recordingDescriptorStrategy &descriptors,
               recordingTransferStrategy &transfer,
               fixedIterationPolicy &iterations,
               recordingResultSink &results) {
    return benchmarkRunComponents{sync, allocator, descriptors, transfer, iterations, results};
}

TEST(BenchmarkExecutorTest, RunsLifecycleInOrder) {
    std::vector<std::string> events;
    recordingSync sync(events);
    recordingAllocator allocator(events);
    recordingDescriptorStrategy descriptors(events);
    recordingTransferStrategy transfer(events);
    fixedIterationPolicy iterations(2);
    recordingResultSink results(events);
    benchmarkRunComponents components =
        makeComponents(sync, allocator, descriptors, transfer, iterations, results);

    benchmarkExecutor executor;
    EXPECT_EQ(executor.run(components), EXIT_SUCCESS);

    const std::vector<std::string> expected_events{
        "sync_start",
        "allocate",
        "before_transfer",
        "descriptors",
        "transfer",
        "record",
        "after_transfer",
        "before_transfer",
        "descriptors",
        "transfer",
        "record",
        "after_transfer",
        "deallocate",
        "finish",
    };
    EXPECT_EQ(events, expected_events);
    EXPECT_EQ(results.record_count, 2);
    EXPECT_EQ(transfer.descriptor_count, 1U);
    EXPECT_EQ(allocator.deallocated_iov_count, 1U);
}

TEST(BenchmarkExecutorTest, DeallocatesAfterTransferFailure) {
    std::vector<std::string> events;
    recordingSync sync(events);
    recordingAllocator allocator(events);
    recordingDescriptorStrategy descriptors(events);
    recordingTransferStrategy transfer(events);
    fixedIterationPolicy iterations(1);
    recordingResultSink results(events);
    benchmarkRunComponents components =
        makeComponents(sync, allocator, descriptors, transfer, iterations, results);
    transfer.execute_result = 17;

    benchmarkExecutor executor;
    EXPECT_EQ(executor.run(components), 17);

    const std::vector<std::string> expected_events{
        "sync_start",
        "allocate",
        "before_transfer",
        "descriptors",
        "transfer",
        "deallocate",
    };
    EXPECT_EQ(events, expected_events);
    EXPECT_EQ(results.record_count, 0);
}

TEST(BenchmarkExecutorTest, DoesNotDeallocateWhenAllocationFails) {
    std::vector<std::string> events;
    recordingSync sync(events);
    recordingAllocator allocator(events);
    recordingDescriptorStrategy descriptors(events);
    recordingTransferStrategy transfer(events);
    fixedIterationPolicy iterations(1);
    recordingResultSink results(events);
    benchmarkRunComponents components =
        makeComponents(sync, allocator, descriptors, transfer, iterations, results);
    allocator.allocate_result = 19;

    benchmarkExecutor executor;
    EXPECT_EQ(executor.run(components), 19);

    const std::vector<std::string> expected_events{"sync_start", "allocate"};
    EXPECT_EQ(events, expected_events);
    EXPECT_EQ(results.record_count, 0);
}

} // namespace
} // namespace nixlbench
