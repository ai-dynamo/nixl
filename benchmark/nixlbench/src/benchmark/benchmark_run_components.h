/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUN_COMPONENTS_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUN_COMPONENTS_H

#include "benchmark/benchmark_runtime_sync.h"
#include "utils/utils.h"

#include <variant>
#include <vector>

namespace nixlbench {

struct benchmarkAllocation {
    std::vector<std::vector<xferBenchIOV>> local_iovs;
    std::vector<std::vector<xferBenchIOV>> remote_iovs;
};

using allocationResult = std::variant<benchmarkAllocation, int>;

class benchmarkMemoryAllocator {
public:
    virtual
    ~benchmarkMemoryAllocator() = default;

    virtual allocationResult
    allocate() = 0;

    virtual void
    deallocate(benchmarkAllocation &allocation) = 0;
};

class transferDescriptorStrategy {
public:
    virtual
    ~transferDescriptorStrategy() = default;

    virtual std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(const benchmarkAllocation &allocation) = 0;
};

class benchmarkTransferStrategy {
public:
    virtual
    ~benchmarkTransferStrategy() = default;

    virtual std::variant<xferBenchStats, int>
    execute(const std::vector<std::vector<xferBenchIOV>> &descriptors) = 0;
};

class iterationPolicy {
public:
    virtual
    ~iterationPolicy() = default;

    virtual bool
    allocateOnce() const = 0;

    virtual bool
    hasNext() const = 0;

    virtual void
    advance() = 0;
};

class benchmarkResultSink {
public:
    virtual
    ~benchmarkResultSink() = default;

    virtual void
    record(const xferBenchStats &stats) = 0;
};

struct benchmarkRunComponents {
    benchmarkRuntimeSync &sync;
    benchmarkMemoryAllocator &allocator;
    transferDescriptorStrategy &descriptorStrategy;
    benchmarkTransferStrategy &transferStrategy;
    iterationPolicy &iterations;
    benchmarkResultSink &results;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUN_COMPONENTS_H
