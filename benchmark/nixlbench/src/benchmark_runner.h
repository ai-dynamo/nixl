/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_RUNNER_H
#define NIXLBENCH_BENCHMARK_RUNNER_H

#include <memory>
#include <vector>

#include "benchmark_config.h"
#include "utils/utils.h"
#include "worker/worker.h"

int
runBenchmark(const nixlbench::benchmarkConfig &config);

int
runBenchmark(xferBenchConfig &config);

std::unique_ptr<xferBenchWorker>
createWorker(xferBenchConfig &config);

int
processBatchSizes(xferBenchWorker &worker,
                  const xferBenchConfig &config,
                  std::vector<std::vector<xferBenchIOV>> &iov_lists,
                  size_t block_size,
                  int num_threads,
                  bool randomized_read_location = false);

std::vector<std::vector<xferBenchIOV>>
createTransferDescLists(xferBenchWorker &worker,
                        const xferBenchConfig &config,
                        std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        size_t block_size,
                        size_t batch_size,
                        int num_threads,
                        bool randomized_read_location = false);

size_t parse_file_size(const std::string &file_size);

#endif // NIXLBENCH_BENCHMARK_RUNNER_H
