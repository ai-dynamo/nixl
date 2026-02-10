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

#include "libfabric_topology.h"
#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <cmath>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

nixlLibfabricTopology::nixlLibfabricTopology()
    : num_aws_accel(0),
      num_nvidia_accel(0),
      num_numa_nodes(0),
      num_devices(0),
      topology_discovered(false),
      hwloc_topology(nullptr),
      avg_numa_speed(0),
      avg_nic_speed(0),
      avg_nic_upstream_speed(0) {

    NIXL_TRACE << "Starting automatic topology discovery";

    // Discover topology immediately - hard error if it fails
    nixl_status_t status = discoverTopology();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Topology discovery failed - no suitable network providers found";
        throw std::runtime_error(
            "Failed to discover system topology - cannot proceed without topology information");
    }
    NIXL_TRACE << "Topology discovery completed successfully";
    printTopologyInfo();
}

nixlLibfabricTopology::~nixlLibfabricTopology() {
    cleanupHwlocTopology();
}

nixl_status_t
nixlLibfabricTopology::discoverTopology() {
    NIXL_TRACE << "Starting hwloc-based topology discovery";
    // Initialize hwloc topology
    nixl_status_t status = initHwlocTopology();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to initialize hwloc topology";
        return status;
    }
    // Discover EFA devices using libfabric
    status = discoverEfaDevices();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    // For EFA devices, build PCIe to Libfabric device mapping and full topology
    if (provider_name == "efa") {
        // Build PCIe to Libfabric device mapping
        status = buildPcieToLibfabricMapping();
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to build PCIe to Libfabric mapping - this is required for EFA "
                          "topology discovery";
            return status;
        }
        // Discover hardware topology using hwloc
        status = discoverHwlocTopology();
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to discover hwloc topology";
            return status;
        }
        // Build nVidia accelerator to EFA mapping based on PCIe topology
        if (num_nvidia_accel > 0) {
            status = buildAccelToEfaMapping();
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to build accelerator to EFA mapping";
                return status;
            }
        }
    } else {
        // For TCP/sockets devices, bypass complex topology discovery
        NIXL_INFO << "Using simplified topology for " << provider_name
                  << " devices (no topology mapping needed)";

        // Set basic values without hwloc discovery
        num_aws_accel = 0; // TCP doesn't need accelerator topology
        num_numa_nodes = 1; // Simple fallback

        // For TCP/sockets devices, no accelerator-mapping required.
        NIXL_INFO << "TCP devices available globally - no accelerator-specific mapping required";
    }
    topology_discovered = true;
    NIXL_TRACE << "Topology discovery completed successfully";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::discoverEfaDevices() {
    // Use the utility function from libfabric_common
    auto network_device = LibfabricUtils::getAvailableNetworkDevices();
    provider_name = network_device.first;
    all_devices = network_device.second;

    num_devices = all_devices.size();

    // Set device type based on discovered provider
    if (provider_name == "efa") {
        NIXL_INFO << "Discovered " << num_devices << " EFA devices";
    } else if (provider_name == "sockets") {
        NIXL_INFO << "Discovered " << num_devices << " socket devices (TCP fallback)";
    } else if (provider_name == "none" || all_devices.empty()) {
        NIXL_WARN << "No network devices found";
        return NIXL_ERR_BACKEND;
    }

    for (size_t i = 0; i < all_devices.size(); ++i) {
        NIXL_TRACE << "Network device " << i << ": " << all_devices[i]
                   << " (provider=" << provider_name << ")";
    }
    return NIXL_SUCCESS;
}

std::vector<std::string>
nixlLibfabricTopology::getEfaDevicesForPci(const std::string &pci_bus_id) const {
    // Normalize PCI bus ID format to match hwloc format
    // CUDA format: "0000:59:00.0" → hwloc format: "0:59:00.0"
    unsigned int domain, bus, device, function;
    if (sscanf(pci_bus_id.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4) {
        char normalized_pci[32];
        snprintf(normalized_pci,
                 sizeof(normalized_pci),
                 "%x:%02x:%02x.%x",
                 domain,
                 bus,
                 device,
                 function);
        std::string normalized_id(normalized_pci);

        // GPU query, lookup based on GPU BDF
        if (auto it = pci_to_efa_devices.find(normalized_id); it != pci_to_efa_devices.end()) {
            NIXL_DEBUG << "Found EFA devices for PCI " << pci_bus_id << " (normalized to "
                       << normalized_id << ")";
            return it->second;
        }

        // Neuron query, lookup based on EFA BDF
        if (auto it = pcie_to_libfabric_map.find(normalized_id);
            it != pcie_to_libfabric_map.end()) {
            NIXL_DEBUG << "Found EFA devices for PCI " << pci_bus_id << " (normalized to "
                       << normalized_id << ")";
            return {it->second};
        }

        // PCI ID parsed successfully but not found in mapping
        NIXL_WARN << "PCI bus ID " << pci_bus_id << " (normalized to " << normalized_id
                  << ") not found in accelerator-EFA mapping, returning all devices";
    } else {
        // Failed to parse PCI bus ID format
        NIXL_WARN << "Failed to parse PCI bus ID format: " << pci_bus_id
                  << ", returning all devices";
    }

    return all_devices;
}

bool
nixlLibfabricTopology::isValidDevice(const std::string &efa_device) const {
    return std::find(all_devices.begin(), all_devices.end(), efa_device) != all_devices.end();
}

uint16_t
nixlLibfabricTopology::getDeviceNumaNode(const std::string &efa_device) const {
    int device_numa_node = -1;
    NicInfoMap::const_iterator itr = nic_info_map.find(efa_device);
    if (itr == nic_info_map.end()) {
        NIXL_WARN << "EFA device " << efa_device << " not found in nic_info_map";
    } else {
        device_numa_node = itr->second.numa_node_id;
        if (device_numa_node == INVALID_NUMA_NODE_ID) {
            NIXL_WARN << "EFA device " << efa_device << " is not associated with a NUMA node";
        } else {
            NIXL_DEBUG << "EFA device " << efa_device << " is on NUMA node " << device_numa_node;
        }
    }
    return device_numa_node;
}

bool
nixlLibfabricTopology::getPcieDevData(const std::string &efa_device,
                                      uint16_t &numa_node_id,
                                      size_t &device_link_speed,
                                      uint16_t &parent_switch_domain,
                                      uint8_t &parent_switch_bus_id,
                                      size_t &parent_switch_link_speed) const {
    bool found = false;
    NicInfoMap::const_iterator itr = nic_info_map.find(efa_device);
    if (itr == nic_info_map.end()) {
        NIXL_WARN << "EFA device " << efa_device << " not found in nic_info_map";
    } else {
        found = true;
        numa_node_id = itr->second.numa_node_id;
        device_link_speed = itr->second.upstream_link_speed;
        parent_switch_domain = itr->second.parent_switch_domain;
        parent_switch_bus_id = itr->second.parent_switch_bus_id;
        parent_switch_link_speed = itr->second.parent_switch_link_speed;
        NIXL_DEBUG << "EFA device " << efa_device << " has upstream link speed "
                   << device_link_speed << " Gbps, and is associated with NUMA node "
                   << numa_node_id << " through PCIe switch on domain/bus-id "
                   << parent_switch_domain << "/" << parent_switch_bus_id
                   << " with upstream link speed of " << parent_switch_link_speed << " Gbps";
    }
    return found;
}

size_t
nixlLibfabricTopology::getNumaRailCount() const {
    size_t numa_rail_count = 0;
    size_t numa_node_count = numa_speed_map.size();
    if (numa_node_count > 0) {
        numa_rail_count = nic_info_map.size() / numa_node_count;
    }
    return numa_rail_count;
}

void
nixlLibfabricTopology::printTopologyInfo() const {
    NIXL_TRACE << "=== Libfabric Topology Information ===";
    NIXL_TRACE << "Topology discovered: " << (topology_discovered ? "Yes" : "No");
    NIXL_TRACE << "Number of AWS accelerators: " << num_aws_accel;
    NIXL_TRACE << "Number of NUMA nodes: " << num_numa_nodes;
    NIXL_TRACE << "Number of EFA devices: " << num_devices;
    NIXL_TRACE << "EFA devices: ";
    for (size_t i = 0; i < all_devices.size(); ++i) {
        NIXL_TRACE << "  [" << i << "] " << all_devices[i];
    }
    NIXL_TRACE << "Accelerator-PCI → EFA mapping:";
    for (const auto &pair : pci_to_efa_devices) {
        std::stringstream ss;
        ss << "  Accelerator-PCI " << pair.first << " → [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << pair.second[i];
        }
        ss << "]";
        NIXL_TRACE << ss.str();
    }
    NIXL_TRACE << "Host memory (DRAM) will limit number of EFA devices used per-NUMA node "
                  "according to maximum PCIe switch bandwidth";
    NIXL_TRACE << "=====================================";
}

std::string
nixlLibfabricTopology::getTopologyString() const {
    std::stringstream ss;
    ss << "Libfabric Topology: ";
    ss << "AWS_Accelerators=" << num_aws_accel << ", ";
    ss << "NUMA=" << num_numa_nodes << ", ";
    ss << "EFA=" << num_devices << ", ";
    ss << "Discovered=" << (topology_discovered ? "Yes" : "No");
    return ss.str();
}

// hwloc-based implementation methods

nixl_status_t
nixlLibfabricTopology::initHwlocTopology() {
    if (hwloc_topology) {
        cleanupHwlocTopology();
    }

    // Initialize hwloc_topology to nullptr first for safety
    hwloc_topology = nullptr;

    int ret = hwloc_topology_init(&hwloc_topology);
    if (ret != 0) {
        NIXL_ERROR << "Failed to initialize hwloc topology: " << ret;
        hwloc_topology = nullptr;
        return NIXL_ERR_BACKEND;
    }

    // Verify topology was properly initialized
    if (!hwloc_topology) {
        NIXL_ERROR << "hwloc_topology_init succeeded but topology is null";
        return NIXL_ERR_BACKEND;
    }

    // Enable I/O device discovery - this is the key to seeing EFA devices!
#if (HWLOC_API_VERSION >= 0x00020000)
    enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_ALL;
    ret = hwloc_topology_set_io_types_filter(hwloc_topology, filter);
    if (ret != 0) {
        NIXL_WARN << "Failed to set IO types filter: " << ret << ", continuing anyway";
    }
#else
    unsigned long flags = hwloc_topology_get_flags(hwloc_topology);
    flags |= HWLOC_TOPOLOGY_FLAG_WHOLE_IO;
    ret = hwloc_topology_set_flags(hwloc_topology, flags);
    if (ret != 0) {
        NIXL_WARN << "Failed to set WHOLE_IO flag: " << ret << ", continuing anyway";
    }
#endif

    // Add additional safety check before loading
    if (!hwloc_topology) {
        NIXL_ERROR << "hwloc topology became null before loading";
        return NIXL_ERR_BACKEND;
    }

    ret = hwloc_topology_load(hwloc_topology);
    if (ret != 0) {
        NIXL_ERROR << "Failed to load hwloc topology: " << ret;
        // Clean up the partially initialized topology to prevent double-free
        if (hwloc_topology) {
            hwloc_topology_destroy(hwloc_topology);
            hwloc_topology = nullptr;
        }
        return NIXL_ERR_BACKEND;
    }

    // Final verification that topology loaded successfully
    if (!hwloc_topology) {
        NIXL_ERROR << "hwloc topology became null after loading";
        return NIXL_ERR_BACKEND;
    }

    NIXL_TRACE << "hwloc topology initialized successfully with IO device support";
    return NIXL_SUCCESS;
}

void
nixlLibfabricTopology::cleanupHwlocTopology() {
    if (hwloc_topology) {
        hwloc_topology_destroy(hwloc_topology);
        hwloc_topology = nullptr;
    }
}

nixl_status_t
nixlLibfabricTopology::discoverHwlocTopology() {
    if (!hwloc_topology) {
        NIXL_ERROR << "hwloc topology not initialized";
        return NIXL_ERR_BACKEND;
    }
    // Discover accelerators and EFA devices using hwloc
    nixl_status_t status = discoverAccelWithHwloc();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to discover accelerators with hwloc";
        return status;
    }
    status = discoverEfaDevicesWithHwloc();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to discover EFA devices with hwloc";
        return status;
    }
    // Discover NUMA topology
    num_numa_nodes = hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_NUMANODE);
    if (num_numa_nodes == 0) {
        num_numa_nodes = 1; // Fallback to single NUMA node
    }
    NIXL_TRACE << "Discovered " << num_aws_accel << " AWS accelerators and " << num_numa_nodes
               << " NUMA nodes via hwloc";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::discoverAccelWithHwloc() {
    num_aws_accel = 0;
    num_nvidia_accel = 0;
    // Find all PCI devices and log detailed information
    static const char *vendor_names[2] = {"NEURON", "NVIDIA"};
    hwloc_obj_t pci_obj = nullptr;
    while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
        const bool is_nvidia_accel = isNvidiaAccel(pci_obj);
        if (is_nvidia_accel || isNeuronAccel(pci_obj)) {
            std::string pcie_addr = getPcieAddressFromHwlocObj(pci_obj);
            // Get device and vendor info
            uint16_t vendor_id = pci_obj->attr->pcidev.vendor_id;
            uint16_t device_id = pci_obj->attr->pcidev.device_id;
            uint16_t class_id = pci_obj->attr->pcidev.class_id;

            NIXL_TRACE << "Found " << vendor_names[is_nvidia_accel] << " accelerator "
                       << num_aws_accel << ": " << pcie_addr << " (vendor=" << std::hex << vendor_id
                       << ", device=" << device_id << ", class=" << class_id << std::dec << ")";

            num_aws_accel++;
            num_nvidia_accel += is_nvidia_accel;
        }
    }

    NIXL_TRACE << "Discovered " << num_aws_accel << " "
               << vendor_names[num_aws_accel == num_nvidia_accel] << " devices via hwloc";

    // If we found more than 8 NVIDIA accelerators on P5en, investigate further
    if (num_nvidia_accel > 8) {
        NIXL_WARN << "Found " << num_aws_accel
                  << " NVIDIA accelerators, but P5en should have 8. Investigating...";

        // List all NVIDIA devices to understand what we're seeing
        pci_obj = nullptr;
        int accel_count = 0;
        while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
            if (pci_obj->attr->pcidev.vendor_id == 0x10de) { // NVIDIA
                std::string pcie_addr = getPcieAddressFromHwlocObj(pci_obj);
                uint16_t device_id = pci_obj->attr->pcidev.device_id;
                uint16_t class_id = pci_obj->attr->pcidev.class_id;

                NIXL_WARN << "NVIDIA device " << accel_count << ": " << pcie_addr << " (device"
                          << std::hex << device_id << ", class=" << class_id << std::dec << ")";
                accel_count++;
            }
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::discoverEfaDevicesWithHwloc() {
    // EFA devices are already discovered via libfabric
    // This method validates the hwloc discovery matches libfabric discovery
    int hwloc_efa_count = 0;
    hwloc_obj_t pci_obj = nullptr;
    while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
        if (isEfaDevice(pci_obj)) {
            hwloc_efa_count++;
            NIXL_TRACE << "Found EFA device via hwloc: " << getPcieAddressFromHwlocObj(pci_obj);
        }
    }

    NIXL_TRACE << "hwloc found " << hwloc_efa_count << " EFA devices, libfabric found "
               << num_devices;

    if (hwloc_efa_count != num_devices) {
        NIXL_DEBUG << "Mismatch between hwloc (" << hwloc_efa_count << ") and libfabric ("
                   << num_devices << ") EFA device counts";
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildPcieToLibfabricMapping() {
    pcie_to_libfabric_map.clear();
    libfabric_to_pcie_map.clear();

    // Get EFA device info with PCIe addresses from libfabric
    struct fi_info *hints, *info;

    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info for PCIe mapping";
        return NIXL_ERR_BACKEND;
    }

    // Configure hints for the discovered provider
    // This ensures consistency between device discovery and PCIe mapping
    hints->fabric_attr->prov_name = strdup(provider_name.c_str());

    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed for PCIe mapping with provider " << provider_name << ": "
                   << fi_strerror(-ret);
        fi_freeinfo(hints);
        return NIXL_ERR_BACKEND;
    }

    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->domain_attr && cur->domain_attr->name && cur->nic && cur->nic->bus_attr) {
            std::string libfabric_name = cur->domain_attr->name;
            // Extract PCIe address from bus_attr if available
            if (cur->nic->bus_attr->bus_type == FI_BUS_PCI &&
                cur->nic->bus_attr->attr.pci.domain_id != FI_ADDR_UNSPEC) {
                char pcie_addr[32];
                snprintf(pcie_addr,
                         sizeof(pcie_addr),
                         "%x:%02x:%02x.%x",
                         cur->nic->bus_attr->attr.pci.domain_id,
                         cur->nic->bus_attr->attr.pci.bus_id,
                         cur->nic->bus_attr->attr.pci.device_id,
                         cur->nic->bus_attr->attr.pci.function_id);

                std::string pcie_address = pcie_addr;
                pcie_to_libfabric_map[pcie_address] = libfabric_name;
                libfabric_to_pcie_map[libfabric_name] = pcie_address;

                // save also speed for rail selection policy
                size_t nic_speed = 0;
                if (cur->nic->link_attr != nullptr) {
                    nic_speed = cur->nic->link_attr->speed;
                } else {
                    NIXL_WARN << "Could not get NIC link speed for device " << libfabric_name
                              << " at PCIe address " << pcie_addr << " (link_attr is null)";
                }
                nic_speed_map[pcie_addr] = nic_speed;

                NIXL_TRACE << "Mapped PCIe " << pcie_address << " → Libfabric " << libfabric_name
                           << " (provider=" << provider_name << ", NIC link speed: " << nic_speed
                           << ")";
            }
        }
    }

    fi_freeinfo(info);
    fi_freeinfo(hints);
    NIXL_TRACE << "Built PCIe to Libfabric mapping for " << pcie_to_libfabric_map.size()
               << " devices using provider " << provider_name;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildAccelToEfaMapping() {
    pci_to_efa_devices.clear();
    // Implement NIXL's topology-aware accelerator-EFA grouping algorithm
    nixl_status_t status = buildTopologyAwareGrouping();
    if (status != NIXL_SUCCESS) {
        NIXL_WARN << "Topology-aware grouping failed, using fallback to use all available devices";
        return buildFallbackMapping();
    }

    NIXL_TRACE << "Built PCI→EFA mapping for " << pci_to_efa_devices.size()
               << " accelerators using topology-aware algorithm";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildTopologyAwareGrouping() {
    // Step 1: Build NIC info structures by correlating libfabric with hwloc
    std::vector<NicInfo> discovered_nics;
    std::vector<AccelInfo> discovered_accel;
    // Discover NICs by correlating libfabric devices with hwloc objects
    for (const auto &pair : pcie_to_libfabric_map) {
        const std::string &pcie_addr = pair.first;
        const std::string &libfabric_name = pair.second;

        // Parse PCIe address
        uint16_t domain_id;
        uint8_t bus_id, device_id, function_id;
        if (sscanf(pcie_addr.c_str(),
                   "%hx:%hhx:%hhx.%hhx",
                   &domain_id,
                   &bus_id,
                   &device_id,
                   &function_id) != 4) {
            NIXL_WARN << "Failed to parse PCIe address: " << pcie_addr;
            continue;
        }

        // Find corresponding hwloc object
        hwloc_obj_t hwloc_node =
            hwloc_get_pcidev_by_busid(hwloc_topology, domain_id, bus_id, device_id, function_id);

        if (hwloc_node) {
            NicInfo nic;
            nic.libfabric_name = libfabric_name;
            nic.hwloc_node = hwloc_node;
            nic.line_speed = getPcieDevSpeed(pcie_addr);
            // NOTE: upstream link speed is given in GB/s as float, we convert it to size_t Gbps
            nic.upstream_link_speed = (size_t)(hwloc_node->attr->pcidev.linkspeed * 8.0f);
            nic.numa_node_id = getPcieDevNumaNodeId(hwloc_node, pcie_addr);
            nic.domain_id = domain_id;
            nic.bus_id = bus_id;
            nic.device_id = device_id;
            nic.function_id = function_id;
            if (!getPcieDevParentSwitchData(hwloc_node,
                                            pcie_addr,
                                            nic.parent_switch_domain,
                                            nic.parent_switch_bus_id,
                                            nic.parent_switch_link_speed)) {
                NIXL_TRACE << "Could not locate parent PCIe bridge/switch of NIC "
                           << libfabric_name;
                nic.parent_switch_domain = UINT16_MAX;
                nic.parent_switch_bus_id = UINT8_MAX;
            }
            nic_info_map.insert(NicInfoMap::value_type(libfabric_name, nic));
            NIXL_DEBUG << "EFA device " << libfabric_name << " mapped to NUMA node "
                       << nic.numa_node_id << " (PCIe address " << pcie_addr
                       << ", speed: " << nic.line_speed
                       << " Gbps, upstream link speed: " << nic.upstream_link_speed
                       << " Gbps, parent switch domain/bus-id: " << nic.parent_switch_domain << "/"
                       << nic.parent_switch_bus_id << ")";
            discovered_nics.push_back(nic);
            NIXL_TRACE << "Correlated NIC: " << pcie_addr << " → " << libfabric_name;
        } else {
            NIXL_WARN << "Could not find hwloc object for PCIe address: " << pcie_addr;
        }
    }
    // Step 2: Discover accelerators
    hwloc_obj_t pci_obj = nullptr;
    while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
        if (isNvidiaAccel(pci_obj)) {
            AccelInfo accel;
            accel.hwloc_node = pci_obj;
            accel.domain_id = pci_obj->attr->pcidev.domain;
            accel.bus_id = pci_obj->attr->pcidev.bus;
            accel.device_id = pci_obj->attr->pcidev.dev;
            accel.function_id = pci_obj->attr->pcidev.func;
            discovered_accel.push_back(accel);
        }
    }

    NIXL_TRACE << "Discovered " << discovered_nics.size() << " NICs and " << discovered_accel.size()
               << " accelerators for grouping";

    if (discovered_nics.empty() || discovered_accel.empty()) {
        NIXL_WARN << "No NICs or accelerators found for grouping";
        return NIXL_ERR_BACKEND;
    }
    // Step 3: Implement NIXL's topology-aware grouping algorithm
    std::vector<NicGroup> nic_groups;
    nixl_status_t status = groupNicsWithAccel(discovered_nics, discovered_accel, nic_groups);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    // Step 4: Convert groups to Accelerator→EFA mapping
    for (size_t group_idx = 0; group_idx < nic_groups.size(); ++group_idx) {
        const auto &group = nic_groups[group_idx];
        if (group.has_accel) {
            std::vector<std::string> accel_efa_devices;
            for (const auto &nic : group.nics) {
                accel_efa_devices.push_back(nic.libfabric_name);
            }
            // Find accelerator index in our discovered accelerators list
            int accel_index = -1;
            for (size_t i = 0; i < discovered_accel.size(); ++i) {
                const auto &accel = discovered_accel[i];
                if (accel.domain_id == group.closest_accel.domain_id &&
                    accel.bus_id == group.closest_accel.bus_id &&
                    accel.device_id == group.closest_accel.device_id &&
                    accel.function_id == group.closest_accel.function_id) {
                    accel_index = static_cast<int>(i);
                    break;
                }
            }

            if (accel_index >= 0) {
                // Store mapping using PCI bus ID as key
                std::string pci_bus_id = getPcieAddressFromHwlocObj(group.closest_accel.hwloc_node);
                pci_to_efa_devices[pci_bus_id] = accel_efa_devices;

                NIXL_TRACE << "PCI " << pci_bus_id << " (Accelerator " << accel_index << ") → "
                           << accel_efa_devices.size() << " EFA devices: [";
                for (size_t i = 0; i < accel_efa_devices.size(); ++i) {
                    if (i > 0) NIXL_TRACE << ", ";
                    NIXL_TRACE << accel_efa_devices[i];
                }
                NIXL_TRACE << "]";
            }
        }
    }
    // step 5: compute the capacity limit of each NUMA node and some other topology metrics
    buildNumaSpeedMap();
    calcAvgNumaNodeBandwidth();
    calcAvgNicBandwidth();
    calcAvgNicUpstreamBandwidth();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildFallbackMapping() {
    // Fallback: if specific mapping failed, use simple approach
    // We can't build PCI-based mapping without topology, so just return success
    // getEfaDevicesForPci() will return all_devices when no mapping is found
    NIXL_WARN << "Using fallback: all accelerators will use all available EFA devices";
    return NIXL_SUCCESS;
}

// hwloc helper methods
std::string
nixlLibfabricTopology::getPcieAddressFromHwlocPcidev(
    const hwloc_obj_attr_u::hwloc_pcidev_attr_s &pcidev) const {
    char pcie_addr[32];
    snprintf(pcie_addr,
             sizeof(pcie_addr),
             "%x:%02x:%02x.%x",
             pcidev.domain,
             pcidev.bus,
             pcidev.dev,
             pcidev.func);
    return std::string(pcie_addr);
}

std::string
nixlLibfabricTopology::getPcieAddressFromHwlocObj(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return "";
    }
    return getPcieAddressFromHwlocPcidev(obj->attr->pcidev);
}

bool
nixlLibfabricTopology::isNvidiaAccel(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    // NVIDIA vendor ID is 0x10de
    if (obj->attr->pcidev.vendor_id != 0x10de) {
        return false;
    }
    // Only count devices with GPU class (0x300-0x3ff for display controllers)
    // Class 0x302 is 3D controller (GPU), 0x680 is other devices (network, etc.)
    uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400);
}

bool
nixlLibfabricTopology::isNeuronAccel(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    // Amazon vendor ID is 0x1d0f
    if (obj->attr->pcidev.vendor_id != 0x1d0f) {
        return false;
    }
    static const uint16_t NEURON_DEVICE_IDS[] = {
        0x7264, // INF2
        0x7164, // TRN1
        0x7364, // TRN2
        0x7564, // TRN3_DEVICE_0
        0x7565, // TRN3_DEVICE_1
    };
    return std::find(std::begin(NEURON_DEVICE_IDS),
                     std::end(NEURON_DEVICE_IDS),
                     obj->attr->pcidev.device_id) != std::end(NEURON_DEVICE_IDS);
}

bool
nixlLibfabricTopology::isEfaDevice(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    NIXL_TRACE << "Checking isEfaDevice on device " << std::hex << std::showbase
               << obj->attr->pcidev.vendor_id << " " << obj->attr->pcidev.device_id;

    // Amazon EFA vendor ID is 0x1d0f, device ID matches 0xefa* (wildcard for any EFA device)
    return obj->attr->pcidev.vendor_id == 0x1d0f &&
        (obj->attr->pcidev.device_id & 0xfff0) == 0xefa0;
}

size_t
nixlLibfabricTopology::getPcieDevSpeed(const std::string &pcie_addr) {
    size_t speed = 0;
    std::unordered_map<std::string, size_t>::const_iterator itr = nic_speed_map.find(pcie_addr);
    if (itr != nic_speed_map.end()) {
        // convert from bits to Giga BITS per second
        // NOTE: device reports in multiples of 1000 and not 1024
        const uint64_t GIGA = 1000ull * 1000ull * 1000ull;
        speed = itr->second / GIGA;
        NIXL_DEBUG << "Found speed for NIC at PCIe address " << pcie_addr << ": " << speed
                   << " (Gbps)";
    } else {
        NIXL_WARN << "Could not verify speed of NIC at PCIe address " << pcie_addr;
    }
    return speed;
}

uint16_t
nixlLibfabricTopology::getPcieDevNumaNodeId(hwloc_obj_t obj, const std::string &pcie_addr) {
    // get numa node id closest to NIC, if there is more than one, then choose the first
    // first prepare a location object with the PCIe device object
    uint16_t numa_id = INVALID_NUMA_NODE_ID;
    hwloc_location location = {};
    location.type = HWLOC_LOCATION_TYPE_OBJECT;
    location.location.object = obj;

    // request for at most one NUMA node in response
    // NOTE: flags (last parameter) is passed in as zero for exact match of CPU-set in non-I/O
    // parent node
    unsigned int node_count = 1;
    hwloc_obj_t node_obj = nullptr;
    int res = hwloc_get_local_numanode_objs(hwloc_topology, &location, &node_count, &node_obj, 0);
    if (res != 0) {
        NIXL_ERROR << "Failed to identify the NUMA node closest to NIC PCIe device at address "
                   << pcie_addr << ", error code: " << res;
        return INVALID_NUMA_NODE_ID;
    }
    if (node_count == 0) {
        // this is possible in some instance types (e.g. g5.48xl), so we issue only a warning
        NIXL_WARN << "Failed to identify the NUMA node closest to NIC PCIe device at address "
                  << pcie_addr << ": no node found";
        return INVALID_NUMA_NODE_ID;
    }
    if (node_count > 1) {
        // highly unlikely, but we are better off checking
        NIXL_ERROR << "Failed to identify the NUMA node closest to NIC PCIe device at address "
                   << pcie_addr
                   << ": invalid node count returned (requesting at most 1, instead got "
                   << node_count << ")";
        return INVALID_NUMA_NODE_ID;
    }
    assert(node_count == 1);
    if (node_obj == nullptr) {
        NIXL_ERROR << "Failed to identify the NUMA node closest to NIC PCIe device at address "
                   << pcie_addr << ": NUMA hwloc object returned null";
        return INVALID_NUMA_NODE_ID;
    }

    // os_index is enough (no need to check in nodeset bitset)
    unsigned numa_id_unsigned = node_obj->os_index;
    NIXL_DEBUG << "NIC at PCIe address " << pcie_addr << " is closest to NUMA node "
               << numa_id_unsigned << " (by os_index)";

    // sanity check
    int max_node = -1;
    if (!LibfabricUtils::getMaxNumaNode(max_node)) {
        return INVALID_NUMA_NODE_ID;
    }
    if (numa_id_unsigned > (unsigned)max_node) {
        NIXL_ERROR << "Failed to identify the NUMA node closest to NIC PCIe device at address "
                   << pcie_addr << ": NUMA node ID " << numa_id_unsigned
                   << " is out of range (max: " << max_node << ")";
        return INVALID_NUMA_NODE_ID;
    }

    // NOTE: we are NOT checking that the returned node id is found in the allowed nodes as reported
    // by numa_get_mems_allowed(), assuming that hwloc already ensures that (since it uses libnuma)

    // check the unsigned numa id fits in uint16_t, so we can cast safely
    if (numa_id_unsigned > UINT16_MAX) {
        NIXL_ERROR << "NUMA node ID " << numa_id_unsigned << " is out of range";
        return INVALID_NUMA_NODE_ID;
    }
    numa_id = static_cast<uint16_t>(numa_id_unsigned);
    return numa_id;
}

bool
nixlLibfabricTopology::getPcieDevParentSwitchData(hwloc_obj_t obj,
                                                  const std::string &pcie_addr,
                                                  uint16_t &domain,
                                                  uint8_t &bus_id,
                                                  size_t &link_speed) {
    bool found = false;
    float topmost_speed = 0.0f;
    if (obj != nullptr) {
        hwloc_obj_t itr = obj->parent;
        while (itr != nullptr) {
            if (itr->type == HWLOC_OBJ_BRIDGE) {
                if (itr->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI &&
                    itr->attr->bridge.upstream.pci.linkspeed != 0.0f) {
                    domain = itr->attr->bridge.upstream.pci.domain;
                    bus_id = itr->attr->bridge.upstream.pci.bus;
                    topmost_speed = itr->attr->bridge.upstream.pci.linkspeed;
                    found = true;
                }
            }
            itr = itr->parent;
        }
    }

    // round up result, we don't want to leave unused capacity
    link_speed = (size_t)(std::ceil(topmost_speed * 8.0f));
    return found;
}

void
nixlLibfabricTopology::buildNumaSpeedMap() {
    // build the NUMA bandwidth limit map
    // traverse from each NIC to its topmost parent bridge object and record link info
    // finally use the link info to compute bandwidth limit per NUMA node

    // since topmost links may repeat themselves a few times (when traversing upwards from NICs), we
    // need to record each switch only once (by link bus id).
    // so we maintain a map of link bus id and corresponding switch speed and associated NUMA node

    // map from <domain, bus> to <link speed, NUMA node id>
    typedef std::unordered_map<std::pair<uint16_t, uint8_t>,
                               std::pair<float, uint16_t>,
                               pair_hash<uint16_t, uint8_t>>
        LinkSpeedMap;
    LinkSpeedMap link_speed_map;

    uint16_t max_node_id = 0;
    for (const auto &entry : nic_info_map) {
        const NicInfo &nic_info = entry.second;
        if (nic_info.numa_node_id == INVALID_NUMA_NODE_ID) {
            NIXL_TRACE << "NIC " << nic_info.libfabric_name
                       << " is not assoicated with a NUMA node and therefore will not be taken "
                          "into consideration for DRAM_SEG NUMA-aware rail selection";
            continue;
        }
        max_node_id = std::max(max_node_id, nic_info.numa_node_id);

        hwloc_obj_t hwloc_node = hwloc_get_pcidev_by_busid(hwloc_topology,
                                                           nic_info.domain_id,
                                                           nic_info.bus_id,
                                                           nic_info.device_id,
                                                           nic_info.function_id);
        if (hwloc_node != nullptr) {
            NIXL_DEBUG << "Bridge info for NIC " << nic_info.libfabric_name << ":";
            hwloc_obj_t itr = hwloc_node->parent;
            float topmost_speed = 0.0f;
            uint16_t topmost_domain = UINT16_MAX;
            uint8_t topmost_bus_id = UINT8_MAX;
            while (itr != nullptr) {
                if (itr->type == HWLOC_OBJ_BRIDGE) {
                    // print info
                    std::string up_pcie_addr =
                        getPcieAddressFromHwlocPcidev(itr->attr->bridge.upstream.pci);
                    char down_pcie_addr[32];
                    snprintf(down_pcie_addr,
                             sizeof(down_pcie_addr),
                             "%x:%02x:%02x",
                             itr->attr->bridge.downstream.pci.domain,
                             itr->attr->bridge.downstream.pci.secondary_bus,
                             itr->attr->bridge.downstream.pci.subordinate_bus);

                    NIXL_DEBUG << "Inspecting PCIe bridge " << itr->name
                               << " addr [up: " << up_pcie_addr << ", down: " << down_pcie_addr
                               << "] with speed " << itr->attr->bridge.upstream.pci.linkspeed
                               << ", upstream type: " << itr->attr->bridge.upstream_type
                               << ", downstream type: " << itr->attr->bridge.downstream_type;
                    if (itr->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI &&
                        itr->attr->bridge.upstream.pci.linkspeed != 0.0f) {
                        topmost_speed = itr->attr->bridge.upstream.pci.linkspeed;
                        topmost_domain = itr->attr->bridge.upstream.pci.domain;
                        topmost_bus_id = itr->attr->bridge.upstream.pci.bus;
                    }
                }
                itr = itr->parent;
            }

            if (topmost_speed != 0.0f) {
                // NOTE: the topmost switch may appear several time, each time arriving from a
                // different NIC, and we expect to find the same NUMA node id
                uint16_t numa_node_id = nic_info.numa_node_id;
                assert(numa_node_id != INVALID_NUMA_NODE_ID);
                std::pair<LinkSpeedMap::iterator, bool> pairib =
                    link_speed_map.insert(LinkSpeedMap::value_type({topmost_domain, topmost_bus_id},
                                                                   {topmost_speed, numa_node_id}));
                if (!pairib.second) {
                    // entry already exists, let's just verify it has the same NUMA node id
                    // (otherwise we are probably completely off here, or there is HW issue)
                    if (pairib.first->second.second != numa_node_id) {
                        NIXL_WARN << "Invalid NUMA node id " << numa_node_id << " for link bus "
                                  << topmost_bus_id << ", expecting instead "
                                  << pairib.first->second.second
                                  << ", entry will be ignored (sub-optimal DRAM performance may be "
                                     "observed)";
                    }
                } else {
                    NIXL_DEBUG << "Recording link bus " << topmost_bus_id << " speed "
                               << topmost_speed << " GB/s to capacity of NUMA node "
                               << numa_node_id;
                }
            }
        }
    }

    // now we process the link speed map and accumulate per NUMA node
    numa_speed_map.resize(max_node_id + 1, 0);
    for (const auto &entry : link_speed_map) {
        float topmost_speed = entry.second.first;
        uint16_t numa_node_id = entry.second.second;
        if (numa_node_id != INVALID_NUMA_NODE_ID) {
            // convert from GB/s to Gbps
            size_t speed_gbps = (size_t)(std::ceil(topmost_speed * 8.0f));
            numa_speed_map[numa_node_id] += speed_gbps;
            NIXL_DEBUG << "Adding link speed " << speed_gbps << " Gbps to capacity of NUMA node "
                       << numa_node_id;
        }
    }
    for (size_t i = 0; i < numa_speed_map.size(); ++i) {
        NIXL_DEBUG << "NUMA node " << i << " capacity is " << numa_speed_map[i]
                   << " Gbps, by topmost PCIe brdige/switch link speed";
    }
}

void
nixlLibfabricTopology::calcAvgNumaNodeBandwidth() {
    // calculate average NUMA node capacity, and print warning if not the same on all nodes
    size_t speed_count = numa_speed_map.size();
    if (speed_count == 0) {
        // return early (avoid division by zero)
        avg_numa_speed = 0;
        return;
    }
    bool speed_uniform = true;
    size_t speed = 0;
    bool speed_valid = false;
    size_t total_speed = 0;
    for (size_t i = 0; i < speed_count; ++i) {
        size_t curr_speed = numa_speed_map[i];
        if (!speed_valid) {
            speed = curr_speed;
            speed_valid = true;
        }
        NIXL_TRACE << "NUMA node " << i << " PCIe link capacity: " << curr_speed << " (Gbps)";
        if (curr_speed != speed) {
            NIXL_WARN << "Non-uniform NUMA node " << i << " PCIe capacity: " << curr_speed
                      << " Gbps (expected " << speed << " Gbps)";
            speed_uniform = false;
        }
        total_speed += curr_speed;
    }
    if (speed_uniform) {
        avg_numa_speed = speed;
        NIXL_DEBUG << "NUMA PCIe capacity is uniform across all nodes with link bandwidth: "
                   << avg_numa_speed << " Gbps";
    } else {
        avg_numa_speed = total_speed / speed_count;
        NIXL_WARN
            << "NUMA PCIe capacity is non-uniform across all nodes with average link bandwidth: "
            << avg_numa_speed << " Gbps, rail selection policy may be sub-optimal";
    }
}

void
nixlLibfabricTopology::calcAvgNicBandwidth() {
    // calculate average NIC bandwidth, and print warning if any NIC has a different bandwidth
    if (nic_info_map.empty()) {
        // return early (avoid division by zero)
        avg_nic_speed = 0;
        return;
    }
    bool speed_uniform = true;
    size_t speed = 0;
    bool speed_valid = false;
    size_t total_speed = 0;
    for (const auto &entry : nic_info_map) {
        size_t curr_speed = entry.second.line_speed;
        if (!speed_valid) {
            speed = curr_speed;
            speed_valid = true;
        }
        if (curr_speed != speed) {
            NIXL_WARN << "Non-uniform NIC " << entry.first << " speed: " << curr_speed
                      << " Gbps, expecting " << speed << " Gbps";
            speed_uniform = false;
        }
        total_speed += curr_speed;
    }
    if (speed_uniform) {
        avg_nic_speed = speed;
        NIXL_DEBUG << "NIC bandwidth is uniform across all PCIe devices with bandwidth: "
                   << avg_nic_speed << " Gbps";
    } else {
        avg_nic_speed = total_speed / nic_info_map.size();
        NIXL_WARN << "NIC bandwidth is non-uniform across all PCIe devices with average bandwidth: "
                  << avg_nic_speed << " Gbps, rail selection policy may be sub-optimal";
    }
}

void
nixlLibfabricTopology::calcAvgNicUpstreamBandwidth() {
    // calculate average NIC bandwidth, and print warning if any NIC has a different bandwidth
    if (nic_info_map.empty()) {
        // return early (avoid division by zero)
        avg_nic_upstream_speed = 0;
        return;
    }
    bool speed_uniform = true;
    size_t speed = 0;
    bool speed_valid = false;
    size_t total_speed = 0;
    for (const auto &entry : nic_info_map) {
        size_t curr_speed = entry.second.upstream_link_speed;
        if (!speed_valid) {
            speed = curr_speed;
            speed_valid = true;
        }
        if (curr_speed != speed) {
            NIXL_WARN << "Non-uniform NIC " << entry.first << " upstream link speed: " << curr_speed
                      << " Gbps, expecting " << speed << " Gbps";
            speed_uniform = false;
        }
        total_speed += curr_speed;
    }
    if (speed_uniform) {
        avg_nic_upstream_speed = speed;
        NIXL_DEBUG
            << "NIC upstream link bandwidth is uniform across all PCIe devices with bandwidth: "
            << avg_nic_upstream_speed << " Gbps";
    } else {
        avg_nic_upstream_speed = total_speed / nic_info_map.size();
        NIXL_WARN << "NIC upstream link bandwidth is non-uniform across all PCIe devices with "
                     "average bandwidth: "
                  << avg_nic_upstream_speed << " Gbps, rail selection policy may be sub-optimal";
    }
}

nixl_status_t
nixlLibfabricTopology::groupNicsWithAccel(const std::vector<NicInfo> &discovered_nics,
                                          const std::vector<AccelInfo> &discovered_accel,
                                          std::vector<NicGroup> &nic_groups) {
    nic_groups.clear();

    // Implement NIXL's topology-aware NIC grouping algorithm

    // Step 1: Mark topology nodes that have NICs in their subtree
    std::unordered_map<hwloc_obj_t, int> node_group_counts;
    std::unordered_map<hwloc_obj_t, std::vector<NicInfo>> node_nics;
    std::set<hwloc_obj_t> nic_subtree_nodes;
    // Mark all nodes that have NICs in their subtree and collect NICs per node
    for (const auto &nic : discovered_nics) {
        hwloc_obj_t node = nic.hwloc_node;
        node_nics[node].push_back(nic);
        while (node) {
            nic_subtree_nodes.insert(node);
            node = node->parent;
        }
    }

    // Step 2: For each accelerator, walk up until finding a NIC subtree node and increment its
    // count
    std::unordered_map<hwloc_obj_t, std::vector<AccelInfo>> node_accel;

    for (const auto &accel : discovered_accel) {
        hwloc_obj_t node = accel.hwloc_node;

        while (node) {
            if (nic_subtree_nodes.find(node) != nic_subtree_nodes.end()) {
                node_group_counts[node]++;
                node_accel[node].push_back(accel);
                break;
            }
            node = node->parent;
        }
    }

    // Step 3: Collect all NICs that need to be grouped and assign them to ancestor nodes
    std::unordered_map<hwloc_obj_t, std::vector<NicInfo>> ancestor_nics;

    for (const auto &pair : node_nics) {
        hwloc_obj_t nic_node = pair.first;
        const std::vector<NicInfo> &nics = pair.second;

        // Find the ancestor with group count > 0
        hwloc_obj_t target_node = nic_node;
        while (target_node) {
            if (node_group_counts[target_node] > 0) {
                // Add these NICs to this ancestor
                ancestor_nics[target_node].insert(
                    ancestor_nics[target_node].end(), nics.begin(), nics.end());
                break;
            }
            target_node = target_node->parent;
        }
        // If no ancestor found with groups, create individual groups
        if (!target_node) {
            for (const auto &nic : nics) {
                NicGroup group;
                group.nics.push_back(nic);
                group.has_accel = false;
                group.closest_accel.hwloc_node = nullptr;
                group.common_ancestor = nic.hwloc_node;
                nic_groups.push_back(group);
            }
        }
    }
    // Step 4: Split NICs among accelerators for each ancestor node
    for (const auto &pair : ancestor_nics) {
        hwloc_obj_t ancestor = pair.first;
        std::vector<NicInfo> nics = pair.second;
        int num_groups = node_group_counts[ancestor];
        const std::vector<AccelInfo> &accel = node_accel[ancestor];

        if (num_groups > 0 && !accel.empty()) {
            // Sort NICs by bus ID for consistent assignment
            std::sort(nics.begin(), nics.end(), [](const NicInfo &a, const NicInfo &b) {
                if (a.bus_id != b.bus_id) return a.bus_id < b.bus_id;
                return a.device_id < b.device_id;
            });

            // Split NICs among accelerators
            const int nics_per_group = nics.size() / num_groups;
            const int extra_nics = nics.size() % num_groups;

            size_t nic_idx = 0;
            for (int group_idx = 0; group_idx < num_groups && group_idx < (int)accel.size();
                 ++group_idx) {
                NicGroup group;
                group.has_accel = true;
                group.closest_accel = accel[group_idx];
                group.common_ancestor = ancestor;

                if (nics.size() < (size_t)num_groups) {
                    // Give all NICs to this accelerator
                    NIXL_DEBUG << "Fewer NICs (" << nics.size() << ") than accelerators ("
                               << num_groups
                               << ") at ancestor - sharing all NICs with each accelerator";
                    group.nics = nics;
                } else {
                    // Assign NICs to this group via partitioning
                    int group_size = nics_per_group + (group_idx < extra_nics ? 1 : 0);
                    for (int i = 0; i < group_size && nic_idx < nics.size(); ++i, ++nic_idx) {
                        group.nics.push_back(nics[nic_idx]);
                    }
                }

                if (!group.nics.empty()) {
                    nic_groups.push_back(group);
                }
            }
        }
    }

    NIXL_TRACE << "NIXL topology grouping created " << nic_groups.size() << " NIC groups";

    // Log the groups for debugging
    for (size_t i = 0; i < nic_groups.size(); ++i) {
        const auto &group = nic_groups[i];
        if (group.has_accel) {
            NIXL_TRACE << "Group " << i << ": Accelerator " << std::hex
                       << group.closest_accel.domain_id << ":"
                       << static_cast<int>(group.closest_accel.bus_id) << ":"
                       << static_cast<int>(group.closest_accel.device_id) << "."
                       << static_cast<int>(group.closest_accel.function_id) << std::dec << " → "
                       << group.nics.size() << " NICs";
        } else {
            NIXL_TRACE << "Group " << i << ": No accelerator → " << group.nics.size() << " NICs";
        }
    }
    return NIXL_SUCCESS;
}
