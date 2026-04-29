/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_RAW_EXECUTION_H
#define NIXLBENCH_RAW_EXECUTION_H

#include "utils/cli/benchmark_requests.h"

namespace nixlbench {

int
runRawRequest(const RawRequest &request);

} // namespace nixlbench

#endif // NIXLBENCH_RAW_EXECUTION_H
