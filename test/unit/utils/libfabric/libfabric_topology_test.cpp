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

        // Test memory-based device selection
        NIXL_INFO << "3. Testing memory-based device selection...";

        // Test with dummy host memory
        void *host_mem = malloc(1024);
        auto host_devices = topology.getEfaDevicesForMemory(host_mem, DRAM_SEG);
        NIXL_INFO << "   Host memory (" << host_mem << ") mapped to " << host_devices.size()
                  << " EFA devices";
        for (const auto &device : host_devices) {
            NIXL_INFO << "     - " << device;
        }

        // Test with actual GPU memory allocation if CUDA is available
        void *gpu_mem = nullptr;

#ifdef CUDA_FOUND
        bool cuda_allocation_succeeded = false;

        // Try to allocate GPU memory using CUDA
        cudaError_t cuda_err = cudaMalloc(&gpu_mem, 1024);
        if (cuda_err == cudaSuccess) {
            cuda_allocation_succeeded = true;
            NIXL_INFO << "   SUCCESS: CUDA GPU memory allocated successfully";
            auto gpu_devices = topology.getEfaDevicesForMemory(gpu_mem, VRAM_SEG);
            NIXL_INFO << "   GPU memory (" << gpu_mem << ", CUDA) mapped to " << gpu_devices.size()
                      << " EFA devices";
            for (const auto &device : gpu_devices) {
                NIXL_INFO << "     - " << device;
            }

            // Test GPU memory detection
            bool is_gpu_mem = topology.isGpuMemory(gpu_mem);
            int detected_gpu_id = topology.detectGpuIdForMemory(gpu_mem);
            NIXL_DEBUG << "   GPU memory detection: is_gpu=" << (is_gpu_mem ? "true" : "false")
                       << ", gpu_id=" << detected_gpu_id;
        } else {
            NIXL_WARN << "   CUDA allocation failed: " << cudaGetErrorString(cuda_err);
            gpu_mem = malloc(1024); // Fallback to host memory
            auto gpu_devices = topology.getEfaDevicesForMemory(gpu_mem, VRAM_SEG);
            NIXL_INFO << "   GPU memory (" << gpu_mem << ", host fallback) mapped to "
                      << gpu_devices.size() << " EFA devices";
            for (const auto &device : gpu_devices) {
                NIXL_INFO << "     - " << device;
            }
        }
#else
        NIXL_INFO << "   CUDA not available, using host memory as fallback";
        gpu_mem = malloc(1024); // Fallback to host memory
        auto gpu_devices = topology.getEfaDevicesForMemory(gpu_mem, VRAM_SEG);
        NIXL_INFO << "   GPU memory (" << gpu_mem << ", host fallback) mapped to "
                  << gpu_devices.size() << " EFA devices";
        for (const auto &device : gpu_devices) {
            NIXL_INFO << "     - " << device;
        }
#endif
        // Test GPU-specific queries only if GPUs are detected
        int num_gpus = topology.getNumGpus();
        if (num_gpus > 0) {
            NIXL_INFO << "4. Testing GPU-specific queries (detected " << num_gpus << " GPUs)...";
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
            NIXL_INFO << "4. Skipping GPU-specific tests (no GPUs detected)";
        }
        // Test NUMA-specific queries
        int num_numa = topology.getNumNumaNodes();
        NIXL_INFO << "5. Testing NUMA-specific queries (detected " << num_numa << " NUMA nodes)...";
        int test_numa = std::min(num_numa, 2); // Test up to 2 NUMA nodes or all available
        for (int numa_node = 0; numa_node < test_numa; ++numa_node) {
            auto numa_devices = topology.getEfaDevicesForNumaNode(numa_node);
            std::string device_list;
            for (const auto &device : numa_devices) {
                if (!device_list.empty()) device_list += " ";
                device_list += device;
            }
            NIXL_INFO << "   NUMA " << numa_node << " mapped to " << numa_devices.size()
                      << " EFA devices: " << device_list;
        }

        // Clean up memory
        free(host_mem);
#ifdef CUDA_FOUND
        if (cuda_allocation_succeeded) {
            cudaFree(gpu_mem);
        } else {
            free(gpu_mem); // Host memory fallback
        }
#else
        free(gpu_mem); // Always host memory when CUDA not available
#endif
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "   Topology discovery failed: " << e.what();
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
