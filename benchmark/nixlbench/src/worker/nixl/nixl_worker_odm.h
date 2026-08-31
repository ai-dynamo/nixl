/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <map>
#include <string>
#include <vector>

namespace xferBenchOdm {

constexpr int kQidStart = 0;
constexpr int kQidEnd = 7;

struct State {
    std::string device_path_;
    uint64_t base_addr_ = 0;
    int iova_fd_ = -1;
    uint32_t iova_size_ = 0;
    bool use_get_iova_ = false;
    uint64_t dpa_base_ = 0;

    void
    freeIova();
    void
    seedViaHostWrite(size_t total_size, uint8_t pattern);
    uint64_t
    discoverBaseAddr();
    void
    seedDramForRead(size_t total_size);
};

void
configureBackend(const std::vector<std::string> &devices,
                 State &state,
                 std::map<std::string, std::string> &backend_params);

} // namespace xferBenchOdm

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_WORKER_NIXL_NIXL_WORKER_ODM_H
