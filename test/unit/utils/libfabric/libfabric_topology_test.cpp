/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
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

#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#endif

int
main() {
    NIXL_INFO << "=== Testing Libfabric Topology Implementation ===";
    try {
        // Create topology instance - discovery happens automatically in constructor
        NIXL_INFO << "1. Testing topology discovery...";
        nixlLibfabricTopology topology;

        NIXL_INFO << "   SUCCESS: Topology discovery completed successfully";

        // Print topology information
        NIXL_INFO << "2. Topology Information:";
        topology.printTopologyInfo();

        // Test GPU-specific queries only if GPUs are detected
        int num_gpus = topology.getNumGpus();
        if (num_gpus > 0) {
            NIXL_INFO << "3. Testing GPU-specific queries (detected " << num_gpus << " GPUs)...";
            int test_gpus = std::min(num_gpus, 3); // Test up to 3 GPUs or all available
            for (int gpu_id = 0; gpu_id < test_gpus; ++gpu_id) {
                auto gpu_devices = topology.getEfaDevicesForGpu(gpu_id);
                std::string device_list;
                for (const auto &device : gpu_devices) {
                    if (!device_list.empty()) device_list += " ";
                    device_list += device;
                }
                NIXL_INFO << "   GPU " << gpu_id << " mapped to " << gpu_devices.size()
                          << " EFA devices: " << device_list;
            }
        } else {
            NIXL_INFO << "3. Skipping GPU-specific tests (no GPUs detected)";
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "   Topology discovery failed: " << e.what();
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
