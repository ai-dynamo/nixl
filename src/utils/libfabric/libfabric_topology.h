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
 * Automatically discovers system topology using hwloc and maps accelerators to EFA devices
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
    int num_aws_accel; // AWS Trainium accelerators
    int num_nvidia_accel; // NVIDIA GPU accelerators
    int num_numa_nodes;
    int num_devices;

    // Discovery state
    bool topology_discovered;

    // hwloc topology handle
    hwloc_topology_t hwloc_topology;

    // PCIe to Libfabric device mapping
    std::unordered_map<std::string, std::string> pcie_to_libfabric_map;
    std::unordered_map<std::string, std::string> libfabric_to_pcie_map;

    // bandwidth of each NIC
    std::unordered_map<std::string, size_t> nic_speed_map;

    // bandwidth of each NUMA node (i.e. capacity limited by PCIe switch)
    std::vector<size_t> numa_speed_map;
    size_t avg_numa_speed; // average (per NUMA node) PCIe capacity

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
    discoverAccelWithHwloc();
    nixl_status_t
    discoverEfaDevicesWithHwloc();
    nixl_status_t
    buildAccelToEfaMapping();
    void
    cleanupHwlocTopology();

    // invalid NUMA node id
    const uint16_t INVALID_NUMA_NODE_ID = UINT16_MAX;

    // Data structures for NIXL topology-aware grouping algorithm
    struct NicInfo {
        std::string libfabric_name;
        hwloc_obj_t hwloc_node;
        size_t line_speed; // NOTE: multiple of 1000^3
        size_t upstream_link_speed; // NOTE: multiple of 1024^3
        uint16_t numa_node_id;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
        uint8_t parent_switch_bus_id;
        size_t parent_switch_link_speed;
    };

    struct AccelInfo {
        hwloc_obj_t hwloc_node;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
    };

    struct NicGroup {
        std::vector<NicInfo> nics;
        AccelInfo closest_accel;
        hwloc_obj_t common_ancestor;
        bool has_accel;
    };

    // NIC info map (required for NUMA-aware rail selection)
    typedef std::unordered_map<std::string, NicInfo> NicInfoMap;
    NicInfoMap nic_info_map;
    size_t avg_nic_speed; // average NIC speed
    size_t avg_nic_upstream_speed; // average NIC upstream link speed

    // NIXL topology-aware grouping algorithm methods
    nixl_status_t
    buildTopologyAwareGrouping();
    nixl_status_t
    buildFallbackMapping();
    nixl_status_t
    groupNicsWithAccel(const std::vector<NicInfo> &discovered_nics,
                       const std::vector<AccelInfo> &discovered_accel,
                       std::vector<NicGroup> &nic_groups);

    // hwloc helper methods
    std::string
    getPcieAddressFromHwlocPcidev(const hwloc_obj_attr_u::hwloc_pcidev_attr_s &pcidev) const;
    std::string
    getPcieAddressFromHwlocObj(hwloc_obj_t obj) const;
    bool
    isNvidiaAccel(hwloc_obj_t obj) const;
    bool
    isNeuronAccel(hwloc_obj_t obj) const;
    bool
    isEfaDevice(hwloc_obj_t obj) const;

    // retieves line speed of NIC from map
    size_t
    getPcieDevSpeed(const std::string &pcie_addr);

    // finds out the NUMA node id of a PCIe device
    // returns INVALID_NUMA_NODE_ID if not found or error occured
    uint16_t
    getPcieDevNumaNodeId(hwloc_obj_t obj, const std::string &pcie_addr);

    // finds out the PCIe bus id of the topmost parent switch of this device
    // returns UINT8_MAX if not found or error occured
    uint8_t
    getPcieDevParentSwitchBusId(hwloc_obj_t obj, const std::string &pcie_addr, size_t &link_speed);

    // finds out the PCIe bandwidth limit of all NUMA nodes (determined by sum of connected PCIe
    // switches/bridges)
    void
    buildNumaSpeedMap();

    // calculates once the average bandwidth limit per NUMA node
    void
    calcAvgNumaNodeBandwidth();

    // calculates once the average NIC line speed
    void
    calcAvgNicBandwidth();

    // calculates once the average NIC upstream link speed
    void
    calcAvgNicUpstreamBandwidth();

public:
    nixlLibfabricTopology(); // Automatically discovers topology
    ~nixlLibfabricTopology();

    // Accelerator-based queries (main interface)
    std::vector<std::string>
    getEfaDevicesForPci(const std::string &pci_bus_id) const;

    // System information
    int
    getNumAwsAccel() const {
        return num_aws_accel;
    }

    int
    getNumNvidiaAccel() const {
        return num_nvidia_accel;
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

    enum fi_hmem_iface
    getMrAttrIface(int device_id) const {
        return (device_id < num_nvidia_accel) ? FI_HMEM_CUDA : FI_HMEM_NEURON;
    }

    // retrieves the NUMA node id with which the given EFA device is associated
    int
    getDeviceNumaNode(const std::string &efa_device) const;

    // retrieves topology info of an EFA device
    bool
    getPcieDevData(const std::string &efa_device,
                   uint16_t &numa_node_id,
                   size_t &device_link_speed,
                   uint8_t &parent_switch_bus_id,
                   size_t &parent_switch_link_speed);

    // retrieves the average bandwidth limit per NUMA node
    inline size_t
    getAvgNumaNodeBandwidth() const {
        return avg_numa_speed;
    }

    // retrieves the average NIC bandwidth
    inline size_t
    getAvgNicBandwidth() const {
        return avg_nic_speed;
    }

    // retrievs the average NIC upstream link bandwidth
    inline size_t
    getAvgNicUpstreamBandwidth() const {
        return avg_nic_upstream_speed;
    }

    // retrieves the total number of NICs
    inline size_t
    getTotalNicCount() const {
        return nic_info_map.size();
    }

    // retrieves the average number of rails per NUMA node
    size_t
    getNumaRailCount() const;

    // Debug/info
    void
    printTopologyInfo() const;
    std::string
    getTopologyString() const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
