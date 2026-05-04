/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_RUNNER_H
#define NIXLBENCH_BENCHMARK_RUNNER_H

#include <memory>
#include <vector>

#include "worker/worker.h"

int
runBenchmarkWithCurrentConfig();

std::unique_ptr<xferBenchWorker>
createWorker();

int
processBatchSizes(xferBenchWorker &worker,
                  std::vector<std::vector<xferBenchIOV>> &iov_lists,
                  size_t block_size,
                  int num_threads,
                  bool randomized_read_location = false);

std::vector<std::vector<xferBenchIOV>>
createTransferDescLists(xferBenchWorker &worker,
                        std::vector<std::vector<xferBenchIOV>> &iov_lists,
                        size_t block_size,
                        size_t batch_size,
                        int num_threads,
                        bool randomized_read_location = false);

size_t parse_file_size(const std::string &file_size);

#endif // NIXLBENCH_BENCHMARK_RUNNER_H
