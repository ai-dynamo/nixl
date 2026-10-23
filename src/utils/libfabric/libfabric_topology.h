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
#include "libfabric_accel.h"
#include "nixl.h"
#include <hwloc.h>
#include <unordered_map>

/**
 * @brief Topology discovery and management for the RDMA providers (efa, cxi, verbs)
 *
 * Automatically discovers system topology using hwloc and maps accelerators to NICs based on PCIe
 * proximity for optimal performance. Which providers get this treatment is
 * LibfabricUtils::providerNeedsFullTopology(); the rest (tcp, sockets) reach every peer through the
 * host stack and use a simplified topology.
 */
class nixlLibfabricTopology {
private:
    // Accelerator PCI bus ID to NIC domain names: "0000:72:00.0"→[efa0,efa1], etc.
    std::unordered_map<std::string, std::vector<std::string>> pci_to_nic_devices;

    // All available network devices discovered on this system
    std::vector<std::string> all_devices;

    // Network fabric name (efa-direct, efa, tcp, sockets, etc.)
    std::string provider_name;

    /**
     * @brief Every selected domain as device discovery saw it, in rail order.
     *
     * The source for @ref all_devices and for buildPcieToLibfabricMapping(). Kept rather than
     * re-queried: a second fi_getinfo() with different hints could disagree with the one that chose
     * the provider, and used to.
     */
    std::vector<LibfabricUtils::nixlLibfabricDeviceInfo> discovered_devices;

    // System information
    /**
     * Accelerators discovered per HMEM interface, keyed by the vendor table's ifaces. A map rather
     * than one counter per vendor so that adding a vendor to nixl_accel_vendors[] needs no change
     * here; absent key means zero.
     */
    std::unordered_map<int, int> accel_counts;
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
    discoverProviderWithDevices();
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
    discoverRDMADevicesWithHwloc();
    nixl_status_t
    buildAccelToNicMapping();
    void
    buildNicInfoMap();
    void
    cleanupHwlocTopology();

    // Neuron/EFA preflight helpers -- log an actionable NIXL_INFO/NIXL_WARN at plugin
    // init time when the discovered Neuron accelerator count doesn't match the EFA NIC
    // count. Called from discoverTopology() after accelerator + NIC discovery are done.
    // See ai-dynamo/nixl#1994 for background.
    void
    neuronEfaPreflightForEfaProvider();
    void
    neuronEfaPreflightForNonEfaProvider();

    // Data structures for NIXL topology-aware grouping algorithm
    struct NicInfo {
        std::string libfabric_name;
        hwloc_obj_t hwloc_node;
        size_t line_speed; // Gbps (decimal, as multiple of 10^9, not 2^30)
        size_t upstream_link_speed; // Gbps (decimal, as multiple of 10^9, not 2^30)
        uint16_t numa_node_id;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
        uint16_t parent_switch_domain;
        uint8_t parent_switch_bus_id;
        size_t parent_switch_link_speed; // Gbps (decimal, as multiple of 10^9, not 2^30)
        // True when this NIC and the accelerator paired with it sit under a common PCIe
        // switch. buildTopologyAwareGrouping() sets it from NicGroup::common_ancestor.
        bool accel_via_pcie_switch;
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
    bool has_pcie_devices_;

    // NIXL topology-aware grouping algorithm methods
    nixl_status_t
    buildTopologyAwareGrouping();
    nixl_status_t
    buildFallbackMapping();
    nixl_status_t
    groupNicsWithAccel(const std::vector<NicInfo> &discovered_nics,
                       const std::vector<AccelInfo> &discovered_accel,
                       std::vector<NicGroup> &nic_groups);

    // The node where a NIC sharing no PCIe switch with any accelerator meets the
    // accelerators it reaches through the CPU, or nullptr when its own socket holds none.
    //
    // The search stops at the NIC's Package: every accelerator on the host appears under
    // the Machine object, so a climb that accepted it would pair a NIC with accelerators
    // on another socket, which it reaches only across the inter-socket link.
    static hwloc_obj_t
    findHostRouteNode(hwloc_obj_t nic_node,
                      const std::unordered_map<hwloc_obj_t, std::vector<AccelInfo>> &subtree_accel);

    // hwloc helper methods
    std::string
    getPcieAddressFromHwlocPcidev(const hwloc_obj_attr_u::hwloc_pcidev_attr_s &pcidev) const;
    std::string
    getPcieAddressFromHwlocObj(hwloc_obj_t obj) const;
    /**
     * @brief PCIe address of the device behind an hwloc OS device, or "" if there is none.
     *
     * @param osdev_name An hwloc OS device name. For verbs this is the libfabric domain name,
     *                   which is also the InfiniBand device name, e.g. "rocep153s0f0".
     *
     * Used when libfabric itself reports no bus info for a NIC; see
     * @ref buildPcieToLibfabricMapping.
     */
    std::string
    getPcieAddressForOsDev(const std::string &osdev_name) const;
    /**
     * @brief HMEM interface of the vendor that claims @p obj, or FI_HMEM_SYSTEM if none does.
     *
     * The single place hwloc objects are attributed to a vendor. Asks each entry of the accelerator
     * vendor table in turn via hmemIsAccel(); there is no vendor knowledge in this class.
     */
    enum fi_hmem_iface
    accelIfaceOf(hwloc_obj_t obj) const;

    /** @brief Whether @p obj is an accelerator that takes part in PCIe-proximity NIC grouping. */
    bool
    isGroupableAccel(hwloc_obj_t obj) const;

    /** @brief Whether any discovered accelerator takes part in PCIe-proximity NIC grouping. */
    bool
    hasGroupableAccel() const;

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

    void
    collectNicInfo(NicInfo &nic,
                   const std::string &name,
                   const std::string &pcie_addr,
                   hwloc_obj_t hwloc_node,
                   uint16_t domain_id,
                   uint8_t bus_id,
                   uint8_t device_id,
                   uint8_t function_id);

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
    /**
     * @brief NICs closest to the device at @p pci_bus_id, or every NIC if it is not in the map.
     *
     * Named for NICs rather than EFA devices because it also serves cxi, and because the BDF may
     * be an accelerator's (NVIDIA, Intel) or a NIC's (Neuron, whose Trainium cards carry their
     * EFA device onboard).
     */
    std::vector<std::string>
    getNicsForPci(const std::string &pci_bus_id) const;

    /** @brief Deprecated spelling of @ref getNicsForPci. */
    std::vector<std::string>
    getEfaDevicesForPci(const std::string &pci_bus_id) const {
        return getNicsForPci(pci_bus_id);
    }

    // True when obj is a PCIe switch, that is a bridge whose upstream side is PCI.
    // Applied to the common ancestor of two devices, it reports whether a PCIe path runs
    // between them.
    static bool
    isPcieSwitch(hwloc_obj_t obj);

    // True when this EFA device and the accelerator paired with it sit under a common
    // PCIe switch, so the device reaches GPU memory over PCIe.
    //
    // A caller exporting a CUDA dmabuf combines this with the registration's
    // iface == FI_HMEM_CUDA: it requests the GPU's PCIe aperture (BAR1) mapping when both
    // hold, and the platform default mapping otherwise. The two mappings differ on a
    // platform where GPU memory carries several address windows, such as GB200, where HBM
    // is also mapped into the Grace CPU's coherent space and a device reaching HBM through
    // the CPU addresses it through that window.
    //
    // Grouping pairs NICs with NVIDIA and AMD accelerators, so an AMD/ROCr host answers
    // true here as well, and every device on a Trainium host answers false. An unknown
    // device, a device grouping leaves unpaired, and a host where grouping stays idle all
    // answer false.
    bool
    nicSharesPcieSwitchWithAccel(const std::string &efa_device) const;

    // System information

    /** @brief Accelerators discovered for @p iface, zero if that vendor found none. */
    int
    getNumAccel(enum fi_hmem_iface iface) const {
        const auto it = accel_counts.find(static_cast<int>(iface));
        return (it != accel_counts.end()) ? it->second : 0;
    }

    /** @brief Accelerators discovered across every vendor. */
    int
    getTotalNumAccel() const;

    /** @brief Per-vendor spellings of @ref getNumAccel, kept for existing callers and tests. */
    int
    getNumNeuronAccel() const {
        return getNumAccel(FI_HMEM_NEURON);
    }

    int
    getNumNvidiaAccel() const {
        return getNumAccel(FI_HMEM_CUDA);
    }

    int
    getNumIntelAccel() const {
        return getNumAccel(FI_HMEM_ZE);
    }

    int
    getNumAmdAccel() const {
        return getNumAccel(FI_HMEM_ROCR);
    }

    /** @brief Upstream's spelling of @ref getNumNeuronAccel. */
    int
    getNumAwsAccel() const {
        return getNumNeuronAccel();
    }

    const std::vector<std::string> &
    getAllDevices() const {
        return all_devices;
    }

    std::string
    getProviderName() const {
        return provider_name.empty() ? "libfabric" : provider_name;
    }

    // Validation
    bool
    isTopologyDiscovered() const {
        return topology_discovered;
    }

    bool
    isValidDevice(const std::string &nic_device) const;

    /** @brief Invalid NUMA node id constant. */
    static const uint16_t INVALID_NUMA_NODE_ID = UINT16_MAX;

    /** @brief Queries whether there is any NIC with PCIe bus info. */
    inline bool
    hasPcieDevices() const {
        return has_pcie_devices_;
    }

    /**
     * @brief PCIe address of @p nic_device in hwloc's normalisation, or "" if it has none.
     *
     * Empty means the NIC takes no part in PCIe-proximity grouping -- either libfabric reported no
     * bus info and hwloc could not resolve the domain name either, or the provider does not do
     * topology-based rail selection at all.
     */
    inline std::string
    getPcieAddress(const std::string &nic_device) const {
        const auto it = libfabric_to_pcie_map.find(nic_device);
        return (it != libfabric_to_pcie_map.end()) ? it->second : std::string();
    }

    /**
     * @brief Retrieves the NUMA node id with which the given NIC is associated.
     * @param nic_device The NIC for which its associated NUMA node is to be retrieved.
     * @return The NUMA node id or @ref INVALID_NUMA_NODE_ID if failed.
     */
    uint16_t
    getDeviceNumaNode(const std::string &nic_device) const;

    /**
     * @brief Retrieves topology info of a NIC.
     * @param nic_device The libfabric domain name of the NIC.
     * @param[out] numa_node_id The NUMA node id of the NIC.
     * @param[out] device_link_speed The upstream link speed of the NIC.
     * @param[out] parent_switch_domain The PCIe domain of the topmost parent PCIe switch/bridge of
     * the NIC.
     * @param[out] parent_switch_bus_id The PCIe bus id of the topmost parent PCIe switch/bridge of
     * the NIC.
     * @param[out] parent_switch_link_speed The link speed (in Gbps) of the parent PCI
     * switch/bridge.
     * @return True if succeeded, otherwise (NIC not found) false.
     */
    bool
    getPcieDevData(const std::string &nic_device,
                   uint16_t &numa_node_id,
                   size_t &device_link_speed,
                   uint16_t &parent_switch_domain,
                   uint8_t &parent_switch_bus_id,
                   size_t &parent_switch_link_speed) const;

    /**
     * @brief Retrieves the average bandwidth limit per NUMA node. This bandwidth limit of a single
     * NUMA node is the sum of the link speed of all topmost PCIe switches connected to the parent
     * package of the NUMA node, that have at least one subordinate NIC.
     * @return The average bandwidth limit per NUMA node.
     */
    inline size_t
    getAvgNumaNodeBandwidth() const {
        return avg_numa_speed;
    }

    /**
     * @brief Retrieves the average NIC bandwidth in Gbps (decimal, as multiple of 10^9, not 2^30).
     * This is the speed as reported by fi_getinfo().
     */
    inline size_t
    getAvgNicBandwidth() const {
        return avg_nic_speed;
    }

    /**
     * @brief Retrieves the average NIC upstream link bandwidth in Gbps (decimal, as multiple of
     * 10^9, not 2^30). This is the link speed of the PCIe device as reported by hwloc.
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

    /** @brief "CUDA=8 NEURON=0 ZE=0 ", one term per vendor table entry. For logs. */
    std::string
    accelCountsString() const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
