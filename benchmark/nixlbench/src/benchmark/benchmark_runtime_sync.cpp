/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/benchmark_runtime_sync.h"

namespace nixlbench {

int
nullBenchmarkRuntimeSync::synchronizeStart() {
    return 0;
}

int
nullBenchmarkRuntimeSync::beforeTransfer() {
    return 0;
}

int
nullBenchmarkRuntimeSync::afterTransfer() {
    return 0;
}

int
nullBenchmarkRuntimeSync::finish() {
    return 0;
}

distributedBenchmarkRuntimeSync::distributedBenchmarkRuntimeSync(xferBenchRT &runtime)
    : runtime_(runtime) {}

int
distributedBenchmarkRuntimeSync::synchronizeStart() {
    return runtime_.barrier("start_barrier");
}

int
distributedBenchmarkRuntimeSync::beforeTransfer() {
    return runtime_.barrier("before_transfer");
}

int
distributedBenchmarkRuntimeSync::afterTransfer() {
    return runtime_.barrier("after_transfer");
}

int
distributedBenchmarkRuntimeSync::finish() {
    return runtime_.barrier("finish");
}

} // namespace nixlbench
