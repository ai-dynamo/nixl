/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/benchmark_executor.h"

#include "utils/scope_guard.h"

#include <cstdlib>
#include <utility>
#include <variant>

namespace nixlbench {

int
benchmarkExecutor::run(benchmarkRunComponents &components) {
    int ret = components.sync.synchronizeStart();
    if (ret != 0) {
        return ret;
    }

    {
        auto allocation_result = components.allocator.allocate();
        if (std::holds_alternative<int>(allocation_result)) {
            return std::get<int>(allocation_result);
        }

        benchmarkAllocation allocation = std::move(std::get<benchmarkAllocation>(allocation_result));
        auto allocation_guard = make_scope_guard([&] {
            components.allocator.deallocate(allocation);
        });

        for (; components.iterations.hasNext(); components.iterations.advance()) {
            ret = components.sync.beforeTransfer();
            if (ret != 0) {
                return ret;
            }

            auto descriptor_result = components.descriptorStrategy.create(allocation);
            if (std::holds_alternative<int>(descriptor_result)) {
                return std::get<int>(descriptor_result);
            }

            auto transfer_result = components.transferStrategy.execute(
                std::get<std::vector<std::vector<xferBenchIOV>>>(descriptor_result));
            if (std::holds_alternative<int>(transfer_result)) {
                return std::get<int>(transfer_result);
            }

            components.results.record(std::get<xferBenchStats>(transfer_result));

            ret = components.sync.afterTransfer();
            if (ret != 0) {
                return ret;
            }
        }
    }

    ret = components.sync.finish();
    if (ret != 0) {
        return ret;
    }

    return EXIT_SUCCESS;
}

} // namespace nixlbench
