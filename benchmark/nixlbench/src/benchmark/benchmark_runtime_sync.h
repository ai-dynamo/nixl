/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUNTIME_SYNC_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUNTIME_SYNC_H

#include "runtime/runtime.h"

namespace nixlbench {

class benchmarkRuntimeSync {
public:
    virtual
    ~benchmarkRuntimeSync() = default;

    virtual int
    synchronizeStart() = 0;

    virtual int
    beforeTransfer() = 0;

    virtual int
    afterTransfer() = 0;

    virtual int
    finish() = 0;
};

class nullBenchmarkRuntimeSync : public benchmarkRuntimeSync {
public:
    int
    synchronizeStart() override;

    int
    beforeTransfer() override;

    int
    afterTransfer() override;

    int
    finish() override;
};

class distributedBenchmarkRuntimeSync : public benchmarkRuntimeSync {
public:
    explicit
    distributedBenchmarkRuntimeSync(xferBenchRT &runtime);

    int
    synchronizeStart() override;

    int
    beforeTransfer() override;

    int
    afterTransfer() override;

    int
    finish() override;

private:
    xferBenchRT &runtime_;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_BENCHMARK_RUNTIME_SYNC_H
