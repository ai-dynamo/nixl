/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H

#include <vector>

#include "utils/utils.h"

struct OdmConsistencyContext {
    bool active = false;

    explicit OdmConsistencyContext(const std::vector<std::vector<xferBenchIOV>> &iov_lists);

    bool
    fetchWriteBuffer(const xferBenchIOV &iov, void **addr_out, bool *allocated_out);
};

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ODM_CONSISTENCY_H
