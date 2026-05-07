/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_TRANSFER_DESCRIPTOR_STRATEGY_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_TRANSFER_DESCRIPTOR_STRATEGY_H

#include "benchmark/benchmark_run_components.h"
#include "benchmark_config.h"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace nixlbench {

class remoteIovStrategy;

enum class benchmarkAllocationLifecycle {
    AllocateOnce,
    AllocatePerIteration,
};

class fixedIterationPolicy : public iterationPolicy {
public:
    fixedIterationPolicy(int iterations,
                         benchmarkAllocationLifecycle lifecycle =
                             benchmarkAllocationLifecycle::AllocateOnce);

    bool
    allocateOnce() const override;

    bool
    hasNext() const override;

    void
    advance() override;

private:
    int remaining_;
    benchmarkAllocationLifecycle lifecycle_;
};

struct transferDescriptorConfig {
    std::size_t block_size = 0;
    std::size_t batch_size = 0;
    int num_threads = 1;
    int num_initiator_dev = 1;
    int num_target_dev = 1;
    std::size_t total_buffer_size = 0;
    std::string scheme = XFERBENCH_SCHEME_PAIRWISE;
    std::string mode = XFERBENCH_MODE_SG;
    bool is_initiator = true;
    bool is_target = false;
};

transferDescriptorConfig
makeTransferDescriptorConfig(const benchmarkConfig &config,
                             std::size_t block_size,
                             std::size_t batch_size,
                             bool is_initiator,
                             bool is_target);

std::unique_ptr<transferDescriptorStrategy>
makeTransferDescriptorStrategy(const benchmarkConfig &config,
                               bool randomized_rw_location,
                               remoteIovStrategy *remote_strategy = nullptr,
                               bool is_initiator = true,
                               bool is_target = false);

std::variant<std::vector<std::vector<xferBenchIOV>>, int>
createTransferDescLists(const transferDescriptorConfig &config,
                        const std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        bool randomized_rw_location,
                        std::mt19937 *rng = nullptr);
                            
class offsetTransferDescriptorStrategy : public transferDescriptorStrategy {
public:
    offsetTransferDescriptorStrategy(transferDescriptorConfig config,
                                     bool randomized_rw_location);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(const benchmarkAllocation &allocation) override;

private:
    transferDescriptorConfig config_;
    bool randomized_rw_location_;
};

class remoteOffsetTransferDescriptorStrategy : public transferDescriptorStrategy {
public:
    remoteOffsetTransferDescriptorStrategy(transferDescriptorConfig config,
                                           remoteIovStrategy &remote_strategy,
                                           bool randomized_rw_location);

    std::variant<std::vector<std::vector<xferBenchIOV>>, int>
    create(const benchmarkAllocation &allocation) override;

private:
    transferDescriptorConfig config_;
    remoteIovStrategy &remote_strategy_;
    bool randomized_rw_location_;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_TRANSFER_DESCRIPTOR_STRATEGY_H
