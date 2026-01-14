/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H

#include "libfabric_common.h"
#include "nixl.h"
#include <hwloc.h>
#include <unordered_map>

/**
 * @brief Topology discovery and management for AWS instances with EFA devices
 *
 * Automatically discovers system topology using hwloc and maps GPUs to EFA devices
 * based on PCIe proximity for optimal performance. Falls back to TCP/sockets
 * when EFA devices are not available.
 */
class nixlLibfabricTopology {
private:
    // PCI bus ID to EFA device mapping: "0000:72:00.0"→[efa0,efa1], etc.
    std::unordered_map<std::string, std::vector<std::string>> pci_to_efa_devices;

    // All available network devices discovered on this system
    std::vector<std::string> all_devices;

    // Network fabric name (efa-direct, efa, tcp, sockets, etc.)
    std::string provider_name;

    // System information
    int num_gpus;
    int num_numa_nodes;
    int num_devices;

    // Discovery state
    bool topology_discovered;

    // hwloc topology handle
    hwloc_topology_t hwloc_topology;

    // PCIe to Libfabric device mapping
    std::unordered_map<std::string, std::string> pcie_to_libfabric_map;
    std::unordered_map<std::string, std::string> libfabric_to_pcie_map;

    // Helper methods
    nixl_status_t
    discoverEfaDevices();
    nixl_status_t
    discoverTopology();

    // hwloc-based discovery methods
    nixl_status_t
    initHwlocTopology();
    nixl_status_t
    discoverHwlocTopology();
    nixl_status_t
    buildPcieToLibfabricMapping();
    nixl_status_t
    discoverGpusWithHwloc();
    nixl_status_t
    discoverEfaDevicesWithHwloc();
    nixl_status_t
    buildGpuToEfaMapping();
    void
    cleanupHwlocTopology();

    // Data structures for NIXL topology-aware grouping algorithm
    struct NicInfo {
        std::string libfabric_name;
        hwloc_obj_t hwloc_node;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
    };

    struct GpuInfo {
        hwloc_obj_t hwloc_node;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
    };

    struct NicGroup {
        std::vector<NicInfo> nics;
        GpuInfo closest_gpu;
        hwloc_obj_t common_ancestor;
        bool has_gpu;
    };

    // NIXL topology-aware grouping algorithm methods
    nixl_status_t
    buildTopologyAwareGrouping();
    nixl_status_t
    buildFallbackMapping();
    nixl_status_t
    groupNicsWithGpus(const std::vector<NicInfo> &discovered_nics,
                      const std::vector<GpuInfo> &discovered_gpus,
                      std::vector<NicGroup> &nic_groups);

    // hwloc helper methods
    std::string
    getPcieAddressFromHwlocObj(hwloc_obj_t obj) const;
    bool
    isNvidiaGpu(hwloc_obj_t obj) const;
    bool
    isEfaDevice(hwloc_obj_t obj) const;

public:
    nixlLibfabricTopology(); // Automatically discovers topology
    ~nixlLibfabricTopology();

    // GPU-based queries (main interface)
    std::vector<std::string>
    getEfaDevicesForGPUPci(const std::string &pci_bus_id) const;

    // System information
    int
    getNumGpus() const {
        return num_gpus;
    }

    const std::vector<std::string> &
    getAllDevices() const {
        return all_devices;
    }

    const std::string &
    getProviderName() const {
        return provider_name;
    }

    // Validation
    bool
    isTopologyDiscovered() const {
        return topology_discovered;
    }

    bool
    isValidDevice(const std::string &efa_device) const;

    // Debug/info
    void
    printTopologyInfo() const;
    std::string
    getTopologyString() const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
