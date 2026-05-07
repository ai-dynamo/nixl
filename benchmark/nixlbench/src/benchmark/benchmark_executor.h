/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_EXECUTOR_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_EXECUTOR_H

#include "benchmark/benchmark_run_components.h"

namespace nixlbench {

class benchmarkExecutor {
public:
    int
    run(benchmarkRunComponents &components);
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_EXECUTOR_H
