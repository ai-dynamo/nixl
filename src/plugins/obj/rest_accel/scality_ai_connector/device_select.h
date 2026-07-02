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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DEVICE_SELECT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DEVICE_SELECT_H

#include <cstdint>

#include "nixl_types.h"

/**
 * Select the CUDA device whose closest NIC cuObject should bind a buffer's
 * memory region (and thus its primary GID) to.
 *
 * cuObject bakes a single primary device GID into the RDMA descriptor at MR
 * registration, chosen from the current CUDA device context. We exploit that:
 *  - VRAM: bind to the buffer's own GPU (devId).
 *  - DRAM: host memory has no GPU affinity, so spread successive buffers across
 *    all GPUs (devId % gpuCount). Each GPU maps to a different NIC, fanning
 *    host transfers across the available NICs instead of pinning them to one.
 *
 * @return target CUDA device, or -1 to leave the current device unchanged
 *         (DRAM with no GPUs present, or an unsupported segment type).
 */
inline int
targetCudaDevice(nixl_mem_t mem, uint64_t devId, int gpuCount) {
    if (mem == VRAM_SEG) {
        return static_cast<int>(devId);
    }
    if (mem == DRAM_SEG && gpuCount > 0) {
        return static_cast<int>(devId % static_cast<uint64_t>(gpuCount));
    }
    return -1;
}

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_DEVICE_SELECT_H
