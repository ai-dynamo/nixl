/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_ODM_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_ODM_H

#include <cstdint>
#include <string>
#include <vector>

#include <nixl.h>

#include "utils/utils.h"
#include "worker/nixl/nixl_mem_region.h"

namespace xferBenchOdm {

struct State {
    std::string device_path_;
    nixlAgent *agent_ = nullptr;
    nixlBackendH *backend_ = nullptr;
    std::string target_;

    void
    bindNixl(nixlAgent *agent, nixlBackendH *backend, const std::string &target);
    bool
    fetchWriteBuffer(const xferBenchIOV &iov, void **addr_out, bool *allocated_out);
    bool
    seedRegisteredBuffers(const std::vector<NixlMemRegion> &remote_regs,
                          size_t total_size,
                          uint8_t pattern);
    bool
    seedDramForRead(const std::vector<NixlMemRegion> &remote_regs, size_t total_size);
};

void
configureBackend(const std::vector<std::string> &devices,
                 State &state,
                 nixl_b_params_t &backend_params);

void
setConsistencyState(State *state);

bool
fetchWriteBufferForConsistency(const xferBenchIOV &iov, void **addr_out, bool *allocated_out);

} // namespace xferBenchOdm

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_ODM_H
