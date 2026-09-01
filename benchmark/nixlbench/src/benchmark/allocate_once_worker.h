/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_WORKER_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_WORKER_H

#include "benchmark/allocate_once.h"
#include "worker/nixl/nixl_worker.h"

/** @brief NIXL worker strategy that keeps files and registrations alive across requests. */
class xferBenchNixlAllocateOnceWorker final : public xferBenchNixlWorker {
public:
    /** @brief Construct a worker that retains files and registrations for the run. */
    xferBenchNixlAllocateOnceWorker(const std::vector<std::string> &devices,
                                    nixlbench::allocateOnceRequest request);

    /** @brief Open and register fixed files and allocate thread-local buffers. */
    std::vector<std::vector<xferBenchIOV>>
    allocateMemory(int num_threads) override;

    /** @brief Build per-thread remote descriptors for the retained files. */
    std::vector<std::vector<xferBenchIOV>>
    exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                size_t block_size) override;

    /** @brief Run transfers with scenario-owned per-request offset selection. */
    std::variant<xferBenchStats, int>
    transfer(size_t block_size,
             const std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
             const std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

    /** @brief Validate the descriptors touched by the last completed request. */
    bool
    validateTransfer(bool is_initiator,
                     std::vector<std::vector<xferBenchIOV>> &local_iov_lists,
                     std::vector<std::vector<xferBenchIOV>> &remote_iov_lists) override;

private:
    nixlbench::allocateOnceRequest request_;
    std::vector<nixlbench::offsetSequence> offsetSequences_;
    std::vector<std::vector<xferBenchIOV>> lastRemoteIovs_;
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_WORKER_H
