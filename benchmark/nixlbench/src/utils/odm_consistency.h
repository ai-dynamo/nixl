/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#endif
