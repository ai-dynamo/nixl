/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_RUNNER_CONFIG_H
#define NIXLBENCH_BENCHMARK_RUNNER_CONFIG_H

#include "benchmark_config.h"

#include <cstddef>
#include <utility>

namespace nixlbench {

std::pair<std::size_t, std::size_t>
getStrideScheme(const benchmarkConfig &config,
                bool is_initiator,
                bool is_target,
                int num_threads);

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_RUNNER_CONFIG_H
