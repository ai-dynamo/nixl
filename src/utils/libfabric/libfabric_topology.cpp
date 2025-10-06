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

#include "libfabric_topology.h"
#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

nixlLibfabricTopology::nixlLibfabricTopology()
    : num_gpus(0),
      num_numa_nodes(0),
      num_devices(0),
      topology_discovered(false),
      hwloc_topology(nullptr) {

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
        // Build GPU to EFA mapping based on PCIe topology
        status = buildGpuToEfaMapping();
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to build GPU to EFA mapping";
            return status;
        }
    } else {
        // For TCP/sockets devices, bypass complex topology discovery
        NIXL_INFO << "Using simplified topology for " << provider_name
                  << " devices (no topology mapping needed)";

        // Set basic values without hwloc discovery
        num_gpus = 0; // TCP doesn't need GPU topology
        num_numa_nodes = 1; // Simple fallback

        // For TCP/sockets devices, no GPU-mapping required.
        NIXL_INFO << "TCP devices available globally - no GPU-specific mapping required";
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
        NIXL_INFO << "Discovered " << num_devices << " EFA-Direct devices";
    } else if (provider_name == "sockets") {
        NIXL_INFO << "Discovered " << num_devices << " socket devices (TCP fallback)";
    } else if (provider_name == "none" || all_devices.empty()) {
        NIXL_WARN << "No network devices found";
        return NIXL_ERR_BACKEND;
    }

    for (size_t i = 0; i < all_devices.size(); ++i) {
        NIXL_TRACE << "Network device " << i << ": " << all_devices[i]
                   << " (provider: " << provider_name << ")";
    }
    return NIXL_SUCCESS;
}

std::vector<std::string>
nixlLibfabricTopology::getEfaDevicesForGpu(int gpu_id) const {
    auto it = gpu_to_efa_devices.find(gpu_id);
    if (it != gpu_to_efa_devices.end()) {
        return it->second;
    }
    NIXL_WARN << "No EFA devices found for GPU " << gpu_id << ", returning all devices";
    return all_devices;
}

bool
nixlLibfabricTopology::isValidGpuId(int gpu_id) const {
    return gpu_id >= 0 && gpu_id < num_gpus;
}

bool
nixlLibfabricTopology::isValidDevice(const std::string &efa_device) const {
    return std::find(all_devices.begin(), all_devices.end(), efa_device) != all_devices.end();
}

void
nixlLibfabricTopology::printTopologyInfo() const {
    NIXL_TRACE << "=== Libfabric Topology Information ===";
    NIXL_TRACE << "Topology discovered: " << (topology_discovered ? "Yes" : "No");
    NIXL_TRACE << "Number of GPUs: " << num_gpus;
    NIXL_TRACE << "Number of NUMA nodes: " << num_numa_nodes;
    NIXL_TRACE << "Number of EFA devices: " << num_devices;
    NIXL_TRACE << "EFA devices: ";
    for (size_t i = 0; i < all_devices.size(); ++i) {
        NIXL_TRACE << "  [" << i << "] " << all_devices[i];
    }
    NIXL_TRACE << "GPU → EFA mapping:";
    for (const auto &pair : gpu_to_efa_devices) {
        std::stringstream ss;
        ss << "  GPU " << pair.first << " → [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << pair.second[i];
        }
        ss << "]";
        NIXL_TRACE << ss.str();
    }
    NIXL_TRACE << "Host memory (DRAM) will use all available EFA devices for maximum bandwidth";
    NIXL_TRACE << "=====================================";
}

std::string
nixlLibfabricTopology::getTopologyString() const {
    std::stringstream ss;
    ss << "Libfabric Topology: ";
    ss << "GPUs=" << num_gpus << ", ";
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
    // Discover GPUs and EFA devices using hwloc
    nixl_status_t status = discoverGpusWithHwloc();
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to discover GPUs with hwloc";
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
    NIXL_TRACE << "Discovered " << num_gpus << " GPUs and " << num_numa_nodes
               << " NUMA nodes via hwloc";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::discoverGpusWithHwloc() {
    num_gpus = 0;
    // Find all PCI devices and log detailed information
    hwloc_obj_t pci_obj = nullptr;
    while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
        if (isNvidiaGpu(pci_obj)) {
            std::string pcie_addr = getPcieAddressFromHwlocObj(pci_obj);
            // Get device and vendor info
            uint16_t vendor_id = pci_obj->attr->pcidev.vendor_id;
            uint16_t device_id = pci_obj->attr->pcidev.device_id;
            uint16_t class_id = pci_obj->attr->pcidev.class_id;

            NIXL_TRACE << "Found NVIDIA GPU " << num_gpus << ": " << pcie_addr << " (vendor=0x"
                       << std::hex << vendor_id << ", device=0x" << device_id << ", class=0x"
                       << class_id << std::dec << ")";

            num_gpus++;
        }
    }

    NIXL_TRACE << "Discovered " << num_gpus << " NVIDIA GPUs via hwloc";

    // If we found more than 8 GPUs on P5en, investigate further
    if (num_gpus > 8) {
        NIXL_WARN << "Found " << num_gpus
                  << " NVIDIA GPUs, but P5en should have 8. Investigating...";

        // List all NVIDIA devices to understand what we're seeing
        pci_obj = nullptr;
        int gpu_count = 0;
        while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
            if (pci_obj->attr->pcidev.vendor_id == 0x10de) { // NVIDIA
                std::string pcie_addr = getPcieAddressFromHwlocObj(pci_obj);
                uint16_t device_id = pci_obj->attr->pcidev.device_id;
                uint16_t class_id = pci_obj->attr->pcidev.class_id;

                NIXL_WARN << "NVIDIA device " << gpu_count << ": " << pcie_addr << " (device=0x"
                          << std::hex << device_id << ", class=0x" << class_id << std::dec << ")";
                gpu_count++;
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
        NIXL_WARN << "Mismatch between hwloc (" << hwloc_efa_count << ") and libfabric ("
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

    int ret = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &info);
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

                NIXL_TRACE << "Mapped PCIe " << pcie_address << " → Libfabric " << libfabric_name
                           << " (provider: " << provider_name << ")";
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
nixlLibfabricTopology::buildGpuToEfaMapping() {
    gpu_to_efa_devices.clear();
    // Implement NIXL's topology-aware GPU-EFA grouping algorithm
    nixl_status_t status = buildTopologyAwareGrouping();
    if (status != NIXL_SUCCESS) {
        NIXL_WARN << "Topology-aware grouping failed, using fallback to use all available devices";
        return buildFallbackMapping();
    }

    NIXL_TRACE << "Built GPU→EFA mapping for " << gpu_to_efa_devices.size()
               << " GPUs using topology-aware algorithm";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildTopologyAwareGrouping() {
    // Step 1: Build NIC info structures by correlating libfabric with hwloc
    std::vector<NicInfo> discovered_nics;
    std::vector<GpuInfo> discovered_gpus;
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
            nic.domain_id = domain_id;
            nic.bus_id = bus_id;
            nic.device_id = device_id;
            nic.function_id = function_id;
            discovered_nics.push_back(nic);
            NIXL_TRACE << "Correlated NIC: " << pcie_addr << " → " << libfabric_name;
        } else {
            NIXL_WARN << "Could not find hwloc object for PCIe address: " << pcie_addr;
        }
    }
    // Step 2: Discover GPUs
    hwloc_obj_t pci_obj = nullptr;
    while ((pci_obj = hwloc_get_next_pcidev(hwloc_topology, pci_obj)) != nullptr) {
        if (isNvidiaGpu(pci_obj)) {
            GpuInfo gpu;
            gpu.hwloc_node = pci_obj;
            gpu.domain_id = pci_obj->attr->pcidev.domain;
            gpu.bus_id = pci_obj->attr->pcidev.bus;
            gpu.device_id = pci_obj->attr->pcidev.dev;
            gpu.function_id = pci_obj->attr->pcidev.func;
            discovered_gpus.push_back(gpu);
        }
    }

    NIXL_TRACE << "Discovered " << discovered_nics.size() << " NICs and " << discovered_gpus.size()
               << " GPUs for grouping";

    if (discovered_nics.empty() || discovered_gpus.empty()) {
        NIXL_WARN << "No NICs or GPUs found for grouping";
        return NIXL_ERR_BACKEND;
    }
    // Step 3: Implement NIXL's topology-aware grouping algorithm
    std::vector<NicGroup> nic_groups;
    nixl_status_t status = groupNicsWithGpus(discovered_nics, discovered_gpus, nic_groups);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    // Step 4: Convert groups to GPU→EFA mapping
    for (size_t group_idx = 0; group_idx < nic_groups.size(); ++group_idx) {
        const auto &group = nic_groups[group_idx];
        if (group.has_gpu) {
            std::vector<std::string> gpu_efa_devices;
            for (const auto &nic : group.nics) {
                gpu_efa_devices.push_back(nic.libfabric_name);
            }
            // Find GPU index in our discovered GPUs list
            int gpu_index = -1;
            for (size_t i = 0; i < discovered_gpus.size(); ++i) {
                const auto &gpu = discovered_gpus[i];
                if (gpu.domain_id == group.closest_gpu.domain_id &&
                    gpu.bus_id == group.closest_gpu.bus_id &&
                    gpu.device_id == group.closest_gpu.device_id &&
                    gpu.function_id == group.closest_gpu.function_id) {
                    gpu_index = static_cast<int>(i);
                    break;
                }
            }

            if (gpu_index >= 0) {
                gpu_to_efa_devices[gpu_index] = gpu_efa_devices;

                NIXL_TRACE << "GPU " << gpu_index << " (" << std::hex << group.closest_gpu.domain_id
                           << ":" << static_cast<int>(group.closest_gpu.bus_id) << ":"
                           << static_cast<int>(group.closest_gpu.device_id) << "."
                           << static_cast<int>(group.closest_gpu.function_id) << std::dec << ") → "
                           << gpu_efa_devices.size() << " EFA devices";
            }
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricTopology::buildFallbackMapping() {
    // Fallback: if specific mapping failed, use simple approach
    gpu_to_efa_devices.clear();
    // Give all devices to all GPUs (not optimal but functional)
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        gpu_to_efa_devices[gpu_id] = all_devices;
    }
    return NIXL_SUCCESS;
}


// hwloc helper methods

std::string
nixlLibfabricTopology::getPcieAddressFromHwlocObj(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return "";
    }
    char pcie_addr[32];
    snprintf(pcie_addr,
             sizeof(pcie_addr),
             "%x:%02x:%02x.%x",
             obj->attr->pcidev.domain,
             obj->attr->pcidev.bus,
             obj->attr->pcidev.dev,
             obj->attr->pcidev.func);
    return std::string(pcie_addr);
}

bool
nixlLibfabricTopology::isNvidiaGpu(hwloc_obj_t obj) const {
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
nixlLibfabricTopology::isEfaDevice(hwloc_obj_t obj) const {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }

    // Amazon EFA vendor ID is 0x1d0f, device ID can be 0xefa0, 0xefa1, or 0xefa2
    return obj->attr->pcidev.vendor_id == 0x1d0f &&
        (obj->attr->pcidev.device_id == 0xefa0 || obj->attr->pcidev.device_id == 0xefa1 ||
         obj->attr->pcidev.device_id == 0xefa2);
}

nixl_status_t
nixlLibfabricTopology::groupNicsWithGpus(const std::vector<NicInfo> &discovered_nics,
                                         const std::vector<GpuInfo> &discovered_gpus,
                                         std::vector<NicGroup> &nic_groups) {
    nic_groups.clear();

    // Implement NIXL's topology-aware NIC grouping algorithm

    // Step 1: Mark topology nodes that have NICs in their subtree
    std::map<hwloc_obj_t, int> node_group_counts;
    std::map<hwloc_obj_t, std::vector<NicInfo>> node_nics;
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

    // Step 2: For each GPU, walk up until finding a NIC subtree node and increment its count
    std::map<hwloc_obj_t, std::vector<GpuInfo>> node_gpus;

    for (const auto &gpu : discovered_gpus) {
        hwloc_obj_t node = gpu.hwloc_node;

        while (node) {
            if (nic_subtree_nodes.find(node) != nic_subtree_nodes.end()) {
                node_group_counts[node]++;
                node_gpus[node].push_back(gpu);
                break;
            }
            node = node->parent;
        }
    }

    // Step 3: Collect all NICs that need to be grouped and assign them to ancestor nodes
    std::map<hwloc_obj_t, std::vector<NicInfo>> ancestor_nics;

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
                group.has_gpu = false;
                group.closest_gpu.hwloc_node = nullptr;
                group.common_ancestor = nic.hwloc_node;
                nic_groups.push_back(group);
            }
        }
    }
    // Step 4: Split NICs among GPUs for each ancestor node
    for (const auto &pair : ancestor_nics) {
        hwloc_obj_t ancestor = pair.first;
        std::vector<NicInfo> nics = pair.second;
        int num_groups = node_group_counts[ancestor];
        const std::vector<GpuInfo> &gpus = node_gpus[ancestor];

        if (num_groups > 0 && !gpus.empty()) {
            // Sort NICs by bus ID for consistent assignment
            std::sort(nics.begin(), nics.end(), [](const NicInfo &a, const NicInfo &b) {
                if (a.bus_id != b.bus_id) return a.bus_id < b.bus_id;
                return a.device_id < b.device_id;
            });

            // Split NICs among GPUs
            int nics_per_group = nics.size() / num_groups;
            int extra_nics = nics.size() % num_groups;

            size_t nic_idx = 0;
            for (int group_idx = 0; group_idx < num_groups && group_idx < (int)gpus.size();
                 ++group_idx) {
                NicGroup group;
                group.has_gpu = true;
                group.closest_gpu = gpus[group_idx];
                group.common_ancestor = ancestor;
                // Assign NICs to this group
                int group_size = nics_per_group + (group_idx < extra_nics ? 1 : 0);
                for (int i = 0; i < group_size && nic_idx < nics.size(); ++i, ++nic_idx) {
                    group.nics.push_back(nics[nic_idx]);
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
        if (group.has_gpu) {
            NIXL_TRACE << "Group " << i << ": GPU " << std::hex << group.closest_gpu.domain_id
                       << ":" << static_cast<int>(group.closest_gpu.bus_id) << ":"
                       << static_cast<int>(group.closest_gpu.device_id) << "."
                       << static_cast<int>(group.closest_gpu.function_id) << std::dec << " → "
                       << group.nics.size() << " NICs";
        } else {
            NIXL_TRACE << "Group " << i << ": No GPU → " << group.nics.size() << " NICs";
        }
    }
    return NIXL_SUCCESS;
}
