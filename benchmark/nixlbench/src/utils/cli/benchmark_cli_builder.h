/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_CLI_BUILDER_H
#define NIXLBENCH_BENCHMARK_CLI_BUILDER_H

#include "utils/cli/benchmark_requests.h"

namespace nixlbench {

class BenchmarkCliBuilder {
public:
    int
    parse(int argc, char **argv, ParsedBenchmarkCommand &parsed) const;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_CLI_BUILDER_H
