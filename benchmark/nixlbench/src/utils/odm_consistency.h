/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils/utils.h"

struct OdmConsistencyContext {
    bool active = false;
    void *dax_map = nullptr;
    size_t dax_map_size = 0;
    int dax_fd = -1;
    uint64_t dpa_base = 0;

    explicit OdmConsistencyContext(const std::vector<std::vector<xferBenchIOV>> &iov_lists);
    ~OdmConsistencyContext();

    bool
    fetchWriteBuffer(const xferBenchIOV &iov, void **addr_out, bool *allocated_out);
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H
