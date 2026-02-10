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

    // Data structures for NIXL topology-aware grouping algorithm
    struct NicInfo {
        std::string libfabric_name;
        hwloc_obj_t hwloc_node;
        // NOTE: NIC line speed is in multiples of 1000^3, as reported by fi_info - this is compared
        // against user env override, which is specified as 100, 200, etc, in Gbps (Gigabit per
        // second) so we can deduce number of rails from user override
        size_t line_speed;
        // NOTE: upstream link speed is in multiples of 1024^3 (Gigabit per second), as reported by
        // hwloc (originally float GB/s, converted to size_t Gbps) - this is compared against PCIe
        // switch link speed (also in Gbps 1024^3) to deduce number of rails from topology
        size_t upstream_link_speed;
        uint16_t numa_node_id;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
        uint16_t parent_switch_domain;
        uint8_t parent_switch_bus_id;
        // NOTE: switch link speed is in multiples of 1024^3 (Gigabit per second), see
        // upstream_link_speed above for more details
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

    // retrieves line speed of NIC from map
    size_t
    getPcieDevSpeed(const std::string &pcie_addr);

    // finds out the NUMA node id of a PCIe device
    // returns INVALID_NUMA_NODE_ID if not found or error occurred
    uint16_t
    getPcieDevNumaNodeId(hwloc_obj_t obj, const std::string &pcie_addr);

    // finds out the PCIe domain, bus id and link speed of the topmost parent switch of this device
    bool
    getPcieDevParentSwitchData(hwloc_obj_t obj,
                               const std::string &pcie_addr,
                               uint16_t &domain,
                               uint8_t &bus_id,
                               size_t &link_speed);

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

    /** @brief Invalid NUMA node id constant. */
    static const uint16_t INVALID_NUMA_NODE_ID = UINT16_MAX;

    /**
     * @brief Retrieves the NUMA node id with which the given EFA device is associated.
     * @param efa_device The EFA device for which its associated NUMA node is to be retrieved.
     * @return The NUMA node id or @ref INVALID_NUMA_NODE_ID if failed.
     */
    uint16_t
    getDeviceNumaNode(const std::string &efa_device) const;

    /**
     * @brief Retrieves topology info of an EFA device.
     * @param efa_device The EFA device name.
     * @param[out] numa_node_id The NUMA node id of the EFA device.
     * @param[out] device_link_speed The upstream link speed of the EFA device.
     * @param[out] parent_switch_domain The PCIe domain of the topmost parent PCIe switch/bridge of
     * the EFA device.
     * @param[out] parent_switch_bus_id The PCIe bus id of the topmost parent PCIe switch/bridge of
     * the EFA device.
     * @param[out] parent_switch_link_speed The link speed (in Gbps) of the parent PCI
     * switch/bridge.
     * @return True if succeeded, otherwise (EFA device not found) false.
     */
    bool
    getPcieDevData(const std::string &efa_device,
                   uint16_t &numa_node_id,
                   size_t &device_link_speed,
                   uint16_t &parent_switch_domain,
                   uint8_t &parent_switch_bus_id,
                   size_t &parent_switch_link_speed) const;

    /**
     * @brief Retrieves the average bandwidth limit per NUMA node. This bandwidth limit of a single
     * NUMA node is the sum of the link speed of all topmost PCIe switches connected to the parent
     * package of the NUMA node, that have at least one subordinate EFA device.
     * @return The average bandwidth limit per NUMA node.
     */
    inline size_t
    getAvgNumaNodeBandwidth() const {
        return avg_numa_speed;
    }

    /**
     * @brief Retrieves the average NIC bandwidth (Gbps). This is the speed as reported by
     * fi_getinfo(), as multiples of 1000^3 (and not 1024^3).
     */
    inline size_t
    getAvgNicBandwidth() const {
        return avg_nic_speed;
    }

    /**
     * @brief Retrieves the average NIC upstream link bandwidth (Gbps). This is the link speed of
     * the PCIe device as reported by hwloc, as multiples of 1024^3.
     */
    inline size_t
    getAvgNicUpstreamBandwidth() const {
        return avg_nic_upstream_speed;
    }

    /**
     * @brief Retrieves the total number of NICs, as correlated from hwloc. This differs from
     * all_devices array which gathers info from fi_getinfo.
     */
    inline size_t
    getTotalNicCount() const {
        return nic_info_map.size();
    }

    /**
     * @brief Retrieves the average number of rails per NUMA node.
     */
    size_t
    getNumaRailCount() const;

    // Debug/info
    void
    printTopologyInfo() const;
    std::string
    getTopologyString() const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
