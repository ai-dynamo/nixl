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

#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_rail_manager.h"
#include "common/nixl_log.h"

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cassert>
#include <thread>
#include <bitset>

struct TestScenario {
    bool override_bandwidth;
    unsigned bandwidth_limit;
};

// topology info required for NUMA-aware rail selection testing
struct TopologyInfo {
    bool enable;
    const char *instance_type;
    const char *topo_file;
    size_t numa_node_count;
    size_t nic_count;
    size_t nic_line_speed; // Gbps 1000^3
    size_t nic_upstream_link_speed; // GB/s 1024^3
    size_t switch_count;
    size_t numa_capacity; // GB/s 1024^3
    size_t numa_rail_count;
    std::vector<TestScenario> test_scenarios;
    // rail partition: node/switch/rail-ids
    std::vector<std::vector<std::vector<int>>> rail_partition;
};

// list of all topologies that are to be tested
static TopologyInfo topologies[] = {
    //
    // P-series
    //

    // p3dn.24xl instance type
    {.enable = true,
     .instance_type = "p3dn.24xl",
     .topo_file = "p3dn.24xl-topo.xml",
     .numa_node_count = 0, // no NIC is attached to NUMA node, the only NIC is attached to machine
     .nic_count = 1,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 0,
     .switch_count = 0,
     .numa_capacity = 0,
     .numa_rail_count = 1, // should default to all rails, which is 1
     .test_scenarios = {{false, 0}, {true, 50}, {true, 100}, {true, 200}},
     .rail_partition = {{{0}}, {{0}}}}, // pretending single switch in each node, both using rail 0

    // p4d.24xl instance type
    {.enable = true,
     .instance_type = "p4d.24xl",
     .topo_file = "p4d.24xl-topo.xml",
     .numa_node_count = 2,
     .nic_count = 4,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 0,
     .switch_count = 0,
     .numa_capacity = 0, // no switches, this should default to all-rails selection
     .numa_rail_count = 4, // should default to all rails, which is 4
     .test_scenarios = {{false, 0}, {true, 50}, {true, 100}, {true, 200}},
     .rail_partition = {{{0, 1}}, {{2, 3}}}}, // pretending single switch in each node

    // p4de.24xl instance type
    {.enable = false,
     .instance_type = "p4de.24xl",
     .topo_file = "p4de.24xl-topo.xml",
     .numa_node_count = 2,
     .nic_count = 8,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 16,
     .switch_count = 2,
     .numa_capacity = 32,
     .numa_rail_count = 2},

    // p5.48xl instance type
    {.enable = true,
     .instance_type = "p5.48xl",
     .topo_file = "p5.48xl-topo.xml",
     .numa_node_count = 2,
     .nic_count = 32,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 16,
     .switch_count = 4, // 4 switches per numa node
     .numa_capacity = 64, // each switch has 16 GB/s link speed, fits for one NIC
     .numa_rail_count = 4, // 1 rail from each switch
     .test_scenarios = {{false, 0},
                        {true, 50},
                        {true, 400},
                        {true, 800},
                        {true, 1200},
                        {true, 1800},
                        {true, 3000},
                        {true, 10000}},
     .rail_partition = {{{15, 20, 22, 26}, {6, 9, 14, 19}, {5, 8, 17, 21}, {3, 11, 12, 16}},
                        {{24, 25, 27, 30}, {13, 18, 23, 28}, {7, 10, 29, 31}, {0, 1, 2, 4}}}},

    // p5en.48xl instance type
    {.enable = true,
     .instance_type = "p5en.48xl",
     .topo_file = "p5en.48xl-topo.xml",
     .numa_node_count = 2,
     .nic_count = 16,
     .nic_line_speed = 200,
     .nic_upstream_link_speed = 32,
     .switch_count = 2,
     .numa_capacity = 128,
     .numa_rail_count = 4,
     .test_scenarios = {{false, 0},
                        {true, 50},
                        {true, 800},
                        {true, 1200},
                        {true, 1800},
                        {true, 3000},
                        {true, 10000}},
     .rail_partition = {{{0, 2, 3, 10}, {4, 5, 7, 8}}, {{1, 6, 9, 14}, {11, 12, 13, 15}}}},

    // p6-b200.48xl instance type
    {.enable = true,
     .instance_type = "p6-b200.48xl",
     .topo_file = "p6-b200.48xl-topo.xml",
     .numa_node_count = 2,
     .nic_count = 8,
     .nic_line_speed = 400,
     .nic_upstream_link_speed = 64,
     .switch_count = 2,
     .numa_capacity = 128,
     .numa_rail_count = 2,
     .test_scenarios = {{false, 0},
                        {true, 50},
                        {true, 800},
                        {true, 1200},
                        {true, 2000},
                        {true, 2800},
                        {true, 10000}},
     .rail_partition = {{{2, 6}, {3, 7}}, {{0, 4}, {1, 5}}}},

    // p6e-gb200.36/72xl instance type
    {.enable = false,
     .instance_type = "p6e-gb200",
     .topo_file = "p6e-gb200-topo.xml",
     .numa_node_count = 2,
     .nic_count = 16,
     .nic_line_speed = 400,
     .nic_upstream_link_speed = 32,
     .switch_count = 2,
     .numa_capacity = 64,
     .numa_rail_count = 4},

    //
    // G-series
    //

    // g5.48xl instance type
    {.enable = true,
     .instance_type = "g5.48xl",
     .topo_file = "g5.48xl-topo.xml",
     .numa_node_count = 0, // no NIC is attached to NUMA node, the only NIC is attached to machine
     .nic_count = 1,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 0, // g5 reports upstream link speed 0
     .switch_count = 0, // no bridge/switch spec other than host bridge with link speed 0
     .numa_capacity = 0, // topmost switches on g5 report link speed 0
     .numa_rail_count = 1, // should default to all rails, which is 1
     .test_scenarios = {{false, 0}, {true, 50}, {true, 100}, {true, 200}},
     .rail_partition = {{{0}}, {{0}}}}, // pretending single switch in each node, both using rail 0

    // g6.48xl instance type
    {.enable = true,
     .instance_type = "g6.48xl",
     .topo_file = "g6.48xl-topo.xml",
     .numa_node_count = 1, // single NIC is attached to a NUMA node
     .nic_count = 1,
     .nic_line_speed = 100,
     .nic_upstream_link_speed = 0, // g6 reports NIC upstream link speed 0
     .switch_count = 1,
     .numa_capacity = 32, // topmost switch on g6 report link speed 32 GB/s
     .numa_rail_count = 1, // should default to all rails, which is 1
     .test_scenarios = {{false, 0}, {true, 50}, {true, 100}, {true, 200}},
     .rail_partition = {{{0}}, {{0}}}} // pretending single switch in each node, both using rail 0

    // end of list
};
static const size_t topology_count = sizeof(topologies) / sizeof(topologies[0]);

// current topology pointer - used for mocking/injection
static const TopologyInfo *curr_topology = nullptr;

// NIC data loaded from hwloc via XML input test file
struct NicData {
    std::string name;
    std::string pcie_addr;
    int domain;
    int bus;
    int dev;
    int func;
};

typedef std::unordered_map<std::string, NicData> NicMap;

// NIC speed conversion constant
static const uint64_t GIGA = 1000ull * 1000ull * 1000ull;

// testing flag env var name
static const char *NIXL_LIBFABRIC_TESTING_ENV_VAR = "NIXL_LIBFABRIC_TESTING";

// mock fi_info by current topology in use
extern "C" int
__wrap_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info);

// mock fi_fabric to get past rail manager constructor
extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);

// test basic topology loading (unrelated to NUMA-aware rail selection)
static int
testBasicTopology();

// test NUMA-aware rail selection on all enabled topologies
static int
testNumaDramTopologies();

// test NUMA-aware rail selection on a single topology
static int
testNumaDramTopology(const TopologyInfo &topology);

// test NUMA-aware rail selection on a single topology by instance type
static int
testNumaDramTopology(const char *instance_type);

// test NUMA-aware rail selection policy with various topologies and user overrides
static int
testNumaDramRailSelectionPolicy();

static int
testNumaDramRailSelectionPolicy(const TopologyInfo &topology_info,
                                bool override_bandwidth,
                                unsigned bandwidth_limit);

// test NUMA-aware rail selection policy on a single topology by instance type
static int
testNumaDramRailSelectionPolicy(const char *instance_type);

int
main(int argc, char *argv[]) {
    if (argc > 1) {
        // testing for NUMA-aware rail selection for DRAM_SEG
        // the only parameter is the instance type
        // this is required because hwloc caches cannot be flushed, and once it is loaded once, it
        // retains info, and we cannot test for other instance types
        char *instance_type = argv[1];
        int res = testNumaDramTopology(instance_type);
        if (res != 0) {
            return res;
        }
        return testNumaDramRailSelectionPolicy(instance_type);
    }

    // test basic topology
    int res = testBasicTopology();
    if (res != 0) {
        return res;
    }

    // test all topollogies for NUMA-aware DRAM_SEG rail selection
    res = testNumaDramTopologies();
    if (res != 0) {
        return res;
    }

    // test for actual rail selection policy
    return testNumaDramRailSelectionPolicy();
}

int
testBasicTopology() {
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
        int num_gpus = topology.getNumNvidiaAccel();
        if (num_gpus > 0) {
            NIXL_INFO << "3. Testing GPU-specific queries (detected " << num_gpus << " GPUs)...";
            int test_gpus = std::min(num_gpus, 3); // Test up to 3 GPUs or all available
            for (int gpu_id = 0; gpu_id < test_gpus; ++gpu_id) {
#ifdef CUDA_FOUND
                // Get PCI bus ID for this GPU
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, gpu_id);

                char pci_bus_id[32];
                snprintf(pci_bus_id,
                         sizeof(pci_bus_id),
                         "%04x:%02x:%02x.0",
                         prop.pciDomainID,
                         prop.pciBusID,
                         prop.pciDeviceID);

                auto gpu_devices = topology.getEfaDevicesForPci(pci_bus_id);

                std::string device_list;
                for (const auto &device : gpu_devices) {
                    if (!device_list.empty()) device_list += " ";
                    device_list += device;
                }
                NIXL_INFO << "   GPU " << gpu_id << " (PCI: " << pci_bus_id << ") mapped to "
                          << gpu_devices.size() << " EFA devices: " << device_list;
#else
                NIXL_INFO << "   Skipping GPU " << gpu_id << " (CUDA not available)";
#endif
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

static bool
isEfaDevice(hwloc_obj_t obj) {
    // Amazon EFA vendor ID is 0x1d0f, device ID matches 0xefa* (wildcard for any EFA device)
    return obj->attr->pcidev.vendor_id == 0x1d0f &&
        (obj->attr->pcidev.device_id & 0xfff0) == 0xefa0;
}

// hwloc helper methods
static std::string
getPcieAddressFromHwlocPcidev(const hwloc_obj_attr_u::hwloc_pcidev_attr_s &pcidev) {
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

static std::string
getPcieAddressFromHwlocObj(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return "";
    }
    return getPcieAddressFromHwlocPcidev(obj->attr->pcidev);
}

static void
getNicData(NicData &nic, hwloc_obj_t pci_obj) {
    nic.pcie_addr = getPcieAddressFromHwlocObj(pci_obj);
    nic.domain = pci_obj->attr->pcidev.domain;
    nic.bus = pci_obj->attr->pcidev.bus;
    nic.dev = pci_obj->attr->pcidev.dev;
    nic.func = pci_obj->attr->pcidev.func;
}

static int
getEfaDeviceNamesFromHwloc(NicMap &nic_map) {
    // when testing we load topologies form XML, so we cannot mix that with local machine info that
    // comes from libfabric's fi_getinfo() - instead we discover network devices from hwloc, but
    // that is also missing NIC card line speed that comes in multiples of 1000^3. currently this
    // comes from additional env var NIXL_LIBFABRIC_NIC_SPEED set by testing framework.
    // unfortunately, there is no way now to mock fi_getinfo from input file

    hwloc_topology_t hwloc_topology = nullptr;
    int ret = hwloc_topology_init(&hwloc_topology);
    if (ret != 0) {
        NIXL_ERROR << "Failed to initialize hwloc topology: " << ret;
        return 1;
    }

    // Enable I/O device discovery - this is the key to seeing EFA devices!
#if (HWLOC_API_VERSION >= 0x00020000)
    enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_ALL;
    ret = hwloc_topology_set_io_types_filter(hwloc_topology, filter);
    if (ret != 0) {
        NIXL_ERROR << "Failed to set IO types filter: " << ret << ", continuing anyway";
        hwloc_topology_destroy(hwloc_topology);
        return 2;
    }
#else
    unsigned long flags = hwloc_topology_get_flags(hwloc_topology);
    flags |= HWLOC_TOPOLOGY_FLAG_WHOLE_IO;
    ret = hwloc_topology_set_flags(hwloc_topology, flags);
    if (ret != 0) {
        NIXL_ERROR << "Failed to set WHOLE_IO flag: " << ret << ", continuing anyway";
        hwloc_topology_destroy(hwloc_topology);
        return 2;
    }
#endif

    // load topology
    // NOTE: this comes from XML file
    ret = hwloc_topology_load(hwloc_topology);
    if (ret != 0) {
        NIXL_ERROR << "Failed to load hwloc topology: " << ret;
        hwloc_topology_destroy(hwloc_topology);
        return 3;
    }

    // get PCI device list, check if EFA, and build map
    hwloc_obj_t os_obj = nullptr;
    while ((os_obj = hwloc_get_next_osdev(hwloc_topology, os_obj)) != nullptr) {
        if (os_obj->attr->osdev.type == HWLOC_OBJ_OSDEV_OPENFABRICS) {
            hwloc_obj_t pci_obj = os_obj->parent;
            if (!pci_obj || pci_obj->type != HWLOC_OBJ_PCI_DEVICE) {
                NIXL_WARN << "[TEST] OS device " << os_obj->name
                          << " parent is not a PCI device, skipping";
                continue;
            }
            if (isEfaDevice(pci_obj)) {
                NicData &nic = nic_map[os_obj->name];
                nic.name = os_obj->name;
                getNicData(nic, pci_obj);
                NIXL_TRACE << "[TEST] Found OS device: " << nic.name << " with PCIe addr "
                           << nic.pcie_addr;
            }
        }
    }

    hwloc_topology_destroy(hwloc_topology);
    return 0;
}

// we need to differentiate normal execution from test scenario
// this is done via env var
inline bool
isTesting() {
    return (getenv(NIXL_LIBFABRIC_TESTING_ENV_VAR) != nullptr);
}

inline void
setTesting() {
    setenv(NIXL_LIBFABRIC_TESTING_ENV_VAR, "1", 1);
}

inline void
clearTesting() {
    unsetenv(NIXL_LIBFABRIC_TESTING_ENV_VAR);
}

// we need to declare to prototype for the original function wrappers
// compiler/linker provides the shim implementation
extern "C" int
__real_numa_max_node();

extern "C" int
__real_numa_num_configured_nodes();

extern "C" int
__real_numa_distance(int node1, int node2);

extern "C" long
__real_get_mempolicy(int *mode,
                     unsigned long *nmask,
                     unsigned long maxnode,
                     void *addr,
                     unsigned flags);
extern "C" int
__real_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info);

extern "C" int
__real_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);

extern "C" int
__wrap_numa_max_node() {
    if (!isTesting() || curr_topology == nullptr) {
        return __real_numa_max_node();
    }

    size_t node_count = curr_topology->numa_node_count;
    return node_count > 0 ? static_cast<int>(node_count - 1) : -1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    if (!isTesting() || curr_topology == nullptr) {
        return __real_numa_num_configured_nodes();
    }

    return curr_topology->numa_node_count;
}

extern "C" int
__wrap_numa_distance(int node1, int node2) {
    if (!isTesting() || curr_topology == nullptr) {
        return __real_numa_distance(node1, node2);
    }

    // in all topologies we have only 2 nodes which are adjacent
    // if this does not hold true in the future, then distance info should be added to static
    // topology test array
    return node1 == node2 ? 10 : 20;
}

// required for verifying rail selection in multi-threaded test
static thread_local int last_buffer_numa_node = 0;

extern "C" long
__wrap_get_mempolicy(int *mode,
                     unsigned long *nmask,
                     unsigned long maxnode,
                     void *addr,
                     unsigned flags) {
    if (!isTesting() || curr_topology == nullptr) {
        return __real_get_mempolicy(mode, nmask, maxnode, addr, flags);
    }

    // if address is null then we do a round robin on all available nodes of the current topology
    if (curr_topology->numa_node_count == 0) {
        *mode = 0;
    } else if (addr == nullptr) {
        static std::atomic<int> next_node = 0;
        *mode = next_node.fetch_add(1, std::memory_order_relaxed) % curr_topology->numa_node_count;
    } else {
        // otherwise we use address as node id, but we must ensure it does not exceed the number of
        // nodes in the currently tested topology
        *mode = ((uint64_t)addr) % curr_topology->numa_node_count;
    }
    // save artificial selection for later validation
    last_buffer_numa_node = *mode;
    return 0;
}

// helper template function for allocating flat type zeroed memory with malloc but keeping new
// operator semantics of throwing bad_alloc when allocation fails
template<typename T>
T *
malloc_zero() {
    T *res = (T *)calloc(1, sizeof(T));
    if (res == nullptr) {
        std::stringstream ss;
        ss << "Failed to allocate " << sizeof(T) << " bytes";
        throw std::runtime_error(ss.str());
    }
    return res;
}

// mock fi_getinfo by current topology in use
extern "C" int
__wrap_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info) {
    // we need to differentiate normal execution from test scenario
    // this is done via env var
    if (!isTesting()) {
        return __real_fi_getinfo(version, node, service, flags, hints, info);
    }

    // make sure we have a mock topology in hand (using only NIC speed)
    assert(curr_topology != nullptr);

    try {
        *info = malloc_zero<fi_info>();
        fi_info *itr = *info;

        // load topology from XML file and feed result into fi_getinfo
        NicMap nic_map;
        int res = getEfaDeviceNamesFromHwloc(nic_map);
        if (res != 0) {
            return res;
        }

        // build result list (assuming fi_freeinfo() uses simple malloc/free)
        fi_info *prev = nullptr;
        for (const auto &entry : nic_map) {
            if (itr == nullptr) {
                prev->next = malloc_zero<fi_info>();
                itr = prev->next;
            }

            itr->domain_attr = malloc_zero<fi_domain_attr>();
            itr->domain_attr->name = strdup(entry.second.name.c_str());

            itr->fabric_attr = malloc_zero<fi_fabric_attr>();
            itr->fabric_attr->prov_name = strdup("efa");
            itr->fabric_attr->name = strdup("efa");

            itr->ep_attr = malloc_zero<fi_ep_attr>();
            itr->ep_attr->type = FI_EP_RDM;

            itr->nic = malloc_zero<fid_nic>();
            itr->nic->bus_attr = malloc_zero<fi_bus_attr>();
            itr->nic->bus_attr->bus_type = FI_BUS_PCI;
            itr->nic->bus_attr->attr.pci.domain_id = entry.second.domain;
            itr->nic->bus_attr->attr.pci.bus_id = entry.second.bus;
            itr->nic->bus_attr->attr.pci.device_id = entry.second.dev;
            itr->nic->bus_attr->attr.pci.function_id = entry.second.func;

            itr->nic->link_attr = malloc_zero<fi_link_attr>();
            itr->nic->link_attr->speed = curr_topology->nic_line_speed * GIGA;

            prev = itr;
            itr = nullptr;
        }
    }
    catch (std::exception &e) {
        // actually only runtime_error can get us here
        NIXL_ERROR << "Failed to mock fi_getinfo(): " << e.what();
        return 1;
    }
    return 0; // or FI_SUCCESS
}

// stubs
static int
fi_av_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

struct fi_ops fi_av_fid_ops_stub{
    .close = fi_av_close_stub,
};

static fi_ops_av av_ops_stub = {};

static int
fi_av_open_stub(struct fid_domain *domain,
                struct fi_av_attr *attr,
                struct fid_av **av,
                void *context) {
    *av = malloc_zero<fid_av>();
    (*av)->fid.ops = &fi_av_fid_ops_stub;
    (*av)->ops = &av_ops_stub;
    return 0;
}

static int
fi_cq_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

struct fi_ops fi_cq_fid_ops_stub{
    .close = fi_cq_close_stub,
};

static fi_ops_cq cq_ops_stub = {};

static int
fi_cq_open_stub(struct fid_domain *domain,
                struct fi_cq_attr *attr,
                struct fid_cq **cq,
                void *context) {
    *cq = malloc_zero<fid_cq>();
    (*cq)->fid.ops = &fi_cq_fid_ops_stub;
    (*cq)->ops = &cq_ops_stub;
    return 0;
}

static int
fi_ep_setopt_stub(fid_t fid, int level, int optname, const void *optval, size_t optlen) {
    return 0;
}

static fi_ops_ep fi_ep_ops_stub = {
    .setopt = fi_ep_setopt_stub,
};

static int
fi_ep_bind_stub(struct fid *fid, struct fid *bfid, uint64_t flags) {
    return 0;
}

static int
fi_ep_control_stub(struct fid *fid, int command, void *arg) {
    return 0;
}

static int
fi_ep_cm_getname_stub(fid_t fid, void *addr, size_t *addrlen) {
    return 0;
}

static int
fi_ep_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

struct fi_ops fi_ep_fid_ops_stub{
    .close = fi_ep_close_stub,
    .bind = fi_ep_bind_stub,
    .control = fi_ep_control_stub,
};

struct fi_ops_cm fi_ep_cm_ops_stub{
    .getname = fi_ep_cm_getname_stub,
};

static ssize_t
fi_ep_recvmsg_stub(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags) {
    return 0;
}

static fi_ops_msg fi_ep_msg_ops_stub{
    .recvmsg = fi_ep_recvmsg_stub,
};

static int
fi_endpoint_stub(struct fid_domain *domain,
                 struct fi_info *info,
                 struct fid_ep **ep,
                 void *context) {
    *ep = malloc_zero<fid_ep>();
    (*ep)->ops = &fi_ep_ops_stub;
    (*ep)->fid.ops = &fi_ep_fid_ops_stub;
    (*ep)->cm = &fi_ep_cm_ops_stub;
    (*ep)->msg = &fi_ep_msg_ops_stub;
    return 0;
}

static fi_ops_domain domain_ops_stub = {.av_open = fi_av_open_stub,
                                        .cq_open = fi_cq_open_stub,
                                        .endpoint = fi_endpoint_stub,
                                        .scalable_ep = nullptr,
                                        .cntr_open = nullptr,
                                        .poll_open = nullptr,
                                        .stx_ctx = nullptr,
                                        .srx_ctx = nullptr,
                                        .query_atomic = nullptr,
                                        .query_collective = nullptr,
                                        .endpoint2 = nullptr};

static int
fi_mr_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static fi_ops fi_mr_self_ops_stub = {
    .close = fi_mr_close_stub,
};

static int
fi_mr_reg_stub(struct fid *fid,
               const void *buf,
               size_t len,
               uint64_t access,
               uint64_t offset,
               uint64_t requested_key,
               uint64_t flags,
               struct fid_mr **mr,
               void *context) {
    *mr = malloc_zero<fid_mr>();
    (*mr)->fid.ops = &fi_mr_self_ops_stub;
    return 0;
}

static fi_ops_mr fi_mr_ops_stub{
    .reg = fi_mr_reg_stub,
};

static int
fi_domain_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

struct fi_ops fi_domain_ops_stub{
    .close = fi_domain_close_stub,
};

static int
fi_domain_stub(struct fid_fabric *fabric,
               struct fi_info *info,
               struct fid_domain **domain,
               void *context) {
    *domain = malloc_zero<fid_domain>();
    (*domain)->fid.ops = &fi_domain_ops_stub;
    (*domain)->ops = &domain_ops_stub;
    (*domain)->mr = &fi_mr_ops_stub;
    return 0;
}

static fi_ops_fabric fabric_ops_stub{.domain = fi_domain_stub,
                                     .passive_ep = nullptr,
                                     .eq_open = nullptr,
                                     .wait_open = nullptr,
                                     .trywait = nullptr,
                                     .domain2 = nullptr};

static int
fi_fabric_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

struct fi_ops fi_fabric_ops_stub{
    .close = fi_fabric_close_stub,
};

// mock fi_fabric to get past rail manager constructor
extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context) {
    // we need to differentiate normal execution from test scenario
    // this is done via env var
    if (!isTesting()) {
        return __real_fi_fabric(attr, fabric, context);
    }

    *fabric = malloc_zero<fid_fabric>();
    (*fabric)->fid.ops = &fi_fabric_ops_stub;
    (*fabric)->ops = &fabric_ops_stub;
    return 0;
}

int
testNumaDramTopologies() {
    NIXL_INFO << "=== Testing Libfabric Topology processing for DRAM_SEG NUMA-aware rail selection "
                 "policy ===";
    int test_id = 1;
    for (size_t i = 0; i < topology_count; ++i) {
        if (topologies[i].enable) {
            const char *instance_type = topologies[i].instance_type;
            NIXL_INFO << test_id++ << ". Testing topology processing for instance type "
                      << instance_type;
            int res = testNumaDramTopology(topologies[i]);
            if (res != 0) {
                NIXL_ERROR << "Test failed with return code: " << res;
                return res;
            }
        }
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}

int
testNumaDramTopology(const char *instance_type) {
    NIXL_INFO << "Testing topology query for DRAM_SEG NUMA-aware rail selection on instance type "
              << instance_type;
    for (size_t i = 0; i < topology_count; ++i) {
        if (topologies[i].enable) {
            if (strcmp(instance_type, topologies[i].instance_type) == 0) {
                int res = testNumaDramTopology(topologies[i]);
                if (res == 0) {
                    NIXL_INFO << "=== Test completed successfully! ===";
                    return 0;
                } else {
                    NIXL_ERROR << "Test failed with return code: " << res;
                    return res;
                }
            }
        }
    }
    NIXL_ERROR << "Could not find topology spec for instance type " << instance_type;
    return 2;
}

int
testNumaDramTopology(const TopologyInfo &topology_info) {
    curr_topology = &topology_info; // required for NIC speed injection
    NIXL_TRACE << "Testing topology: " << topology_info.instance_type;

    // enable topology injection in runtime code
    setTesting();

    // tell hwloc to load topology from XML file
    setenv("HWLOC_XMLFILE", topology_info.topo_file, 1);

    // load topology and test
    nixlLibfabricTopology topology;

    // check NIC count
    if (topology.getTotalNicCount() != topology_info.nic_count) {
        NIXL_ERROR << "Invalid NIC count, expecting " << topology_info.nic_count << ", instead got "
                   << topology.getTotalNicCount();
        return 1;
    }

    // NUMA node count test is a bit more complex
    const std::vector<std::string> &all_devices = topology.getAllDevices();
    std::vector<bool> nodes;
    for (const std::string &device : all_devices) {
        uint16_t node = topology.getDeviceNumaNode(device);
        // on some instance types (e.g. g5), some NIC cards are not associated with NUMA node
        if (node != nixlLibfabricTopology::INVALID_NUMA_NODE_ID) {
            if (node >= nodes.size()) {
                nodes.resize(node + 1, false);
            }
            nodes[node] = true;
        }
    }
    // NOTE: g6 has NIC card on NUMA node 0 so this works, but this might not hold true in future
    // topologies
    if (nodes.size() != topology_info.numa_node_count) {
        NIXL_ERROR << "Invalid NUMA node count, expecting " << topology_info.numa_node_count
                   << ", instead got " << nodes.size();
        return 2;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i]) {
            NIXL_ERROR << "No NIC cards found on NUMA node " << i;
            return 3;
        }
    }

    // check NIC line speed (convert from Gbps to GB/s)
    if (topology.getAvgNicBandwidth() != topology_info.nic_line_speed) {
        NIXL_ERROR << "Invalid NIC bandwidth, expecting " << topology_info.nic_line_speed
                   << ", instead got " << topology.getAvgNicBandwidth();
        return 4;
    }

    // check NIC upstream link speed (convert from Gbps to GB/s)
    int speed = (int)topology.getAvgNicUpstreamBandwidth() / 8;
    if (std::abs(speed - (int)topology_info.nic_upstream_link_speed) > 1) {
        NIXL_ERROR << "Invalid NIC upstream link bandwidth, expecting "
                   << topology_info.nic_upstream_link_speed << ", instead got " << speed;
        return 5;
    }

    // check single NUMA node capacity (limited by topmost PCIe switch capacity)
    speed = (int)topology.getAvgNumaNodeBandwidth() / 8;
    if (std::abs(speed - (int)topology_info.numa_capacity) > 10) {
        NIXL_ERROR << "Invalid NUMA node bandwidth, expecting " << topology_info.numa_capacity
                   << ", instead got " << speed;
        return 6;
    }

    // finally check for correct rail count
    size_t rail_count = topology.getTotalNicCount();
    // if NICs do not report upstream link capacity, then we use all NICs
    // NOTE: hybrid topologies may be more difficult to handle, right now there is no such case
    if (topology.getAvgNicUpstreamBandwidth() != 0) {
        rail_count = topology.getAvgNumaNodeBandwidth() / topology.getAvgNicUpstreamBandwidth();
    }
    if (rail_count != topology_info.numa_rail_count) {
        NIXL_ERROR << "Invalid rail count per NUMA node, expecting "
                   << topology_info.numa_rail_count << ", instead got " << rail_count;
        return 7;
    }

    // clean up
    clearTesting();
    unsetenv("HWLOC_XMLFILE");

    NIXL_INFO << "   SUCCESS: Topology calculated correct rail count " << rail_count
              << " for DRAM_SEG NUMA-aware rail selection policy";
    return 0;
}

int
testNumaDramRailSelectionPolicy() {
    NIXL_INFO << "=== Testing Libfabric DRAM_SEG NUMA-aware rail selection policy ===";
    int test_id = 1;
    for (size_t i = 0; i < topology_count; ++i) {
        if (topologies[i].enable) {
            const char *instance_type = topologies[i].instance_type;
            NIXL_INFO << test_id++ << ". Testing rail selection for instance type "
                      << instance_type;
            for (size_t j = 0; j < topologies[i].test_scenarios.size(); ++j) {
                size_t bandwidth = topologies[i].test_scenarios[j].bandwidth_limit;
                NIXL_INFO << "Running test scenario [bandwidth=" << bandwidth
                          << "] on instance type " << topologies[i].instance_type;
                int res = testNumaDramRailSelectionPolicy(
                    topologies[i], topologies[i].test_scenarios[j].override_bandwidth, bandwidth);
                if (res != 0) {
                    NIXL_ERROR << "Test scenario [bandwidth=" << bandwidth << "] on instance type "
                               << topologies[i].instance_type
                               << " failed with return code: " << res;
                    return res;
                } else {
                    NIXL_INFO << "Test scenario [bandwidth=" << bandwidth << "] on instance type "
                              << topologies[i].instance_type << " SUCCESS";
                }
            }
            NIXL_INFO << "SUCCESS for instance type " << instance_type;
        }
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}

static std::vector<int>
prepareNumaNodeArray(int src_numa_node) {
    std::vector<int> node_ids(curr_topology->numa_node_count, -1);
    for (size_t i = 0; i < curr_topology->numa_node_count; ++i) {
        node_ids[i] = i;
    }
    std::sort(node_ids.begin(), node_ids.end(), [src_numa_node](int node1, int node2) {
        return __wrap_numa_distance(src_numa_node, node1) <
            __wrap_numa_distance(src_numa_node, node2);
    });
    return node_ids;
}

static bool
removeNodeRails(std::bitset<64> &selected_rails_bs, int node_id) {
    const std::vector<std::vector<int>> &node_switch_rails = curr_topology->rail_partition[node_id];
    for (size_t i = 0; i < node_switch_rails.size(); ++i) {
        const std::vector<int> &switch_rails = node_switch_rails[i];
        for (size_t j = 0; j < switch_rails.size(); ++j) {
            if (!selected_rails_bs.test(switch_rails[j])) {
                NIXL_ERROR << "Missing rail id " << switch_rails[j];
                return false;
            }
            selected_rails_bs.set(switch_rails[j], false);
        }
    }
    return true;
}

static bool
removeSwitchRails(std::bitset<64> &selected_rails_bs, int node_id) {
    const std::vector<std::vector<int>> &node_switch_rails = curr_topology->rail_partition[node_id];
    // get the capacity of each bridge/switch (i.e. the link speed of each topmost switch)
    size_t switch_capacity = curr_topology->numa_capacity / node_switch_rails.size();
    // then compute how many rails fit within one switch
    size_t switch_rail_count = switch_capacity / curr_topology->nic_upstream_link_speed;
    for (size_t i = 0; i < node_switch_rails.size(); ++i) {
        const std::vector<int> &switch_rails = node_switch_rails[i];
        size_t rails_found = 0;
        for (size_t j = 0; j < switch_rails.size(); ++j) {
            if (selected_rails_bs.test(switch_rails[j])) {
                selected_rails_bs.set(switch_rails[j], false);
                if (++rails_found == switch_rail_count) {
                    break;
                }
            }
        }
        if (rails_found != switch_rail_count) {
            NIXL_ERROR << "Missing rails from switch " << i << " at node " << node_id;
            return false;
        }
    }
    return true;
}

static bool
verifyRailsSpreadEvenly(std::bitset<64> &selected_rails_bs, int node_id) {
    // we expect the rest of the rails to be spread eveny among switches, so we count the rails in
    // each switch, and then make sure that max-switch-rails - min_switch_rails <= 1
    const std::vector<std::vector<int>> &node_switch_rails = curr_topology->rail_partition[node_id];
    size_t switch_count = node_switch_rails.size();
    std::vector<size_t> switch_rail_count(switch_count, 0);
    for (size_t i = 0; i < node_switch_rails.size(); ++i) {
        const std::vector<int> &switch_rails = node_switch_rails[i];
        for (size_t j = 0; j < switch_rails.size(); ++j) {
            if (selected_rails_bs.test(switch_rails[j])) {
                selected_rails_bs.set(switch_rails[j], false);
                ++switch_rail_count[i];
            }
        }
    }

    // check for too high variance (including switches with no rails selected)
    size_t max_rail_count = 0;
    size_t max_switch_index = SIZE_MAX;
    size_t min_rail_count = SIZE_MAX;
    size_t min_switch_index = SIZE_MAX;
    for (size_t i = 0; i < switch_rail_count.size(); ++i) {
        size_t rail_count = switch_rail_count[i];
        if (rail_count > max_rail_count) {
            max_rail_count = rail_count;
            max_switch_index = i;
        }
        if (rail_count < min_rail_count) {
            min_rail_count = rail_count;
            min_switch_index = i;
        }
    }

    if (max_rail_count - min_rail_count > 1) {
        NIXL_ERROR << "Too many rails (" << max_rail_count << ") selected from switch "
                   << max_switch_index << ", while too few rails (" << min_rail_count
                   << ") selected from switch " << min_switch_index << " seen at node " << node_id;
        return false;
    }
    return true;
}

static bool
verifySwitchRailsNotBreached(std::bitset<64> &selected_rails_bs, int node_id) {
    const std::vector<std::vector<int>> &node_switch_rails = curr_topology->rail_partition[node_id];
    // get the capacity of each bridge/switch (i.e. the link speed of each topmost switch)
    size_t switch_capacity = curr_topology->numa_capacity / node_switch_rails.size();
    // then compute how many rails fit within one switch
    size_t switch_rail_count = switch_capacity / curr_topology->nic_upstream_link_speed;

    // now we verify that if rails from a switch are selected, then the amount does not exceed the
    // capacity of the switch
    // we also verify that amount of rails selected in each switch reaches its full capacity, except
    // maybe for just one switch
    bool seenNonFullSwitch = false;
    size_t nonFullSwitchIndex = 0;
    for (size_t i = 0; i < node_switch_rails.size(); ++i) {
        const std::vector<int> &switch_rails = node_switch_rails[i];
        size_t rails_found = 0;
        for (size_t j = 0; j < switch_rails.size(); ++j) {
            if (selected_rails_bs.test(switch_rails[j])) {
                selected_rails_bs.set(switch_rails[j], false);
                if (++rails_found > switch_rail_count) {
                    NIXL_ERROR << "Too many rails selected from switch " << i << " at node "
                               << node_id;
                    return false;
                }
            }
        }
        if (rails_found > 0 && rails_found < switch_rail_count) {
            // make there is only one non-full switch (exclude switches with no rails selected)
            if (!seenNonFullSwitch) {
                seenNonFullSwitch = true;
                nonFullSwitchIndex = i;
            } else {
                NIXL_ERROR << "More than one switch with non-full rail selection (switches "
                           << nonFullSwitchIndex << " and " << i << "), at node " << node_id;
                return false;
            }
        }
    }
    return true;
}

static bool
validateRailSelection(const std::vector<size_t> &selected_rails,
                      const nixlLibfabricTopology *topology,
                      bool override_bandwidth,
                      unsigned bandwidth_limit) {
    // 1. check rail count is correct
    size_t rail_count = selected_rails.size();
    size_t bandwidth = override_bandwidth ?
        bandwidth_limit :
        curr_topology->numa_rail_count * curr_topology->nic_line_speed;
    size_t expected_rail_count = bandwidth / curr_topology->nic_line_speed;
    expected_rail_count = std::max(1ul, expected_rail_count);
    expected_rail_count = std::min(curr_topology->nic_count, expected_rail_count);
    if (curr_topology->numa_capacity == 0) {
        expected_rail_count = curr_topology->nic_count;
    }
    if (rail_count != expected_rail_count) {
        NIXL_ERROR << "Wrong number of rails selected, expecting " << expected_rail_count
                   << ", instead got " << rail_count;
        return false;
    }

    // for g series and other instance types with single NIC we are done
    if (rail_count == 1) {
        return true;
    }

    // for validation performance we are MUCH better off with a bit set
    std::bitset<64> selected_rails_bs;
    for (size_t i = 0; i < selected_rails.size(); ++i) {
        selected_rails_bs.set(selected_rails[i], true);
    }

    // 2. if rail count exceeds NUMA node capacity, then we expect to see all rails of "full" nodes
    int numa_node = last_buffer_numa_node;
    size_t full_node_rail_count = curr_topology->nic_count / curr_topology->numa_node_count;
    int node_index = 0;
    std::vector<int> node_ids = prepareNumaNodeArray(numa_node);
    while (selected_rails_bs.count() >= full_node_rail_count) {
        // traverse nodes by distance, and check each node's rails are all selected
        int node_id = node_ids[node_index++];
        if (!removeNodeRails(selected_rails_bs, node_id)) {
            NIXL_ERROR << "Rail selection " << LibfabricUtils::railIdsToString(selected_rails)
                       << " missing rails from node " << node_id;
            return false;
        }
    }

    // check if we are done
    if (selected_rails_bs.count() == 0) {
        return true;
    }

    // 3. within last NUMA node, if rail count exceeds node capacity, then we need to see at least
    // the expected number of rails from each switch
    int node_id = node_ids[node_index];
    if (selected_rails_bs.count() >= curr_topology->numa_rail_count) {
        if (!removeSwitchRails(selected_rails_bs, node_id)) {
            NIXL_ERROR << "Rail selection " << LibfabricUtils::railIdsToString(selected_rails)
                       << " missing rails from a switch in node " << node_id;
            return false;
        }
        // now we expect to see the rest of rails spread evenly
        if (selected_rails_bs.count() > 0) {
            if (!verifyRailsSpreadEvenly(selected_rails_bs, node_id)) {
                NIXL_ERROR
                    << "Rail selection " << LibfabricUtils::railIdsToString(selected_rails)
                    << " is not evenly spread among rails, when excedding switch capacity, at node "
                    << node_id;
                return false;
            }
        }
    } else {
        // otherwise we expect in this case to see rails selected switch by switch
        if (selected_rails_bs.count() > 0) {
            if (!verifySwitchRailsNotBreached(selected_rails_bs, node_id)) {
                NIXL_ERROR << "Rail selection " << LibfabricUtils::railIdsToString(selected_rails)
                           << " is not spread evenly among switches in node " << node_id;
                return false;
            }
        }
    }
    if (selected_rails_bs.count() > 0) {
        NIXL_ERROR << "Rail selection " << LibfabricUtils::railIdsToString(selected_rails)
                   << " is invalid, seeing unexpected excess rails";
        return false;
    }
    return true;
}

static bool
testRailSelection(nixlLibfabricRailManager &rail_manager,
                  bool override_bandwidth,
                  unsigned bandwidth_limit) {
    std::vector<size_t> selected_rails;
    if (!rail_manager.getDramRailSelectionPolicy()->selectRails(nullptr, selected_rails)) {
        NIXL_ERROR << "Failed to select rails for DRAM_SEG memory type";
        return false;
    }
    if (!validateRailSelection(
            selected_rails, rail_manager.getTopology(), override_bandwidth, bandwidth_limit)) {
        NIXL_ERROR << "Invalid rail selectin for DRAM_SEG memory type";
        return false;
    }
    return true;
}

int
testNumaDramRailSelectionPolicy(const TopologyInfo &topology_info,
                                bool override_bandwidth,
                                unsigned bandwidth_limit) {
    curr_topology = &topology_info; // required for NIC speed injection
    NIXL_TRACE << "Testing topology: " << topology_info.instance_type;

    // enable topology injection in runtime code
    setTesting();

    // tell hwloc to load topology from XML file
    setenv("HWLOC_XMLFILE", topology_info.topo_file, 1);

    // set user bandwidth override_bandwidth
    if (override_bandwidth) {
        setenv("NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG", std::to_string(bandwidth_limit).c_str(), 1);
    }

    // load topology and test
    nixlLibfabricRailManager rail_manager(0);
    nixl_status_t res = rail_manager.init(nixl_b_params_t());
    if (res != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to initialize rail manager: " << (unsigned)res;
        return 1;
    }

    // single-threaded test:
    // select rails for buffers as many times as there are nodes
    NIXL_INFO << "Running single-threaded test";
    for (size_t i = 0; i < topology_info.numa_node_count; ++i) {
        if (!testRailSelection(rail_manager, override_bandwidth, bandwidth_limit)) {
            NIXL_ERROR << "Rail selection failed";
            return 2;
        }
    }
    NIXL_INFO << "Single-threaded DRAM_SEG rail selection test SUCCESS";

    // multi-threaded test
    NIXL_INFO << "Running multi-threaded test";
    const size_t THREAD_COUNT = 8;
    const size_t SLEEP_USEC = 100;
    const size_t ITER_COUNT = 1000;
    std::vector<std::thread> threads;
    std::atomic<bool> thread_res[THREAD_COUNT] = {false};
    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        threads.emplace_back(std::thread([i,
                                          SLEEP_USEC,
                                          ITER_COUNT,
                                          &rail_manager,
                                          override_bandwidth,
                                          bandwidth_limit,
                                          &thread_res]() -> void {
            NIXL_TRACE << "Test thread " << i << " starting";
            for (size_t j = 0; j < ITER_COUNT; ++j) {
                if (!testRailSelection(rail_manager, override_bandwidth, bandwidth_limit)) {
                    NIXL_ERROR << "Rail selection failed";
                    thread_res[i] = false;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_USEC));
            }
            thread_res[i] = true;
        }));
    }

    // wait for all threads to finish
    for (auto &t : threads) {
        t.join();
    }

    // check result of each thread
    for (auto &tres : thread_res) {
        if (!tres) {
            NIXL_ERROR << "Multi-threaded DRAM_SEG rail selection test FAILED";
            return 3;
        }
    }
    NIXL_INFO << "Multi-threaded DRAM_SEG rail selection test SUCCESS";

    // clean up
    clearTesting();
    unsetenv("HWLOC_XMLFILE");

    // unset user bandwidth override_bandwidth
    if (override_bandwidth) {
        unsetenv("NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG");
    }

    return 0;
}

static int
testNumaDramRailSelectionPolicy(const char *instance_type) {

    NIXL_INFO << "Testing DRAM_SEG NUMA-aware rail selection policy on instance type "
              << instance_type;
    for (size_t i = 0; i < topology_count; ++i) {
        if (topologies[i].enable) {
            if (strncmp(instance_type, topologies[i].topo_file, strlen(instance_type)) == 0) {
                for (size_t j = 0; j < topologies[i].test_scenarios.size(); ++j) {
                    size_t bandwidth = topologies[i].test_scenarios[j].bandwidth_limit;
                    NIXL_INFO << "Running test scenario [bandwidth=" << bandwidth
                              << "] on instance type " << topologies[i].instance_type;
                    int res = testNumaDramRailSelectionPolicy(
                        topologies[i],
                        topologies[i].test_scenarios[j].override_bandwidth,
                        bandwidth);
                    if (res != 0) {
                        NIXL_ERROR << "Test scenario [bandwidth=" << bandwidth
                                   << "] on instance type " << topologies[i].instance_type
                                   << " failed with return code: " << res;
                        return res;
                    } else {
                        NIXL_INFO << "Test scenario [bandwidth=" << bandwidth
                                  << "] on instance type " << topologies[i].instance_type
                                  << " SUCCESS";
                    }
                }
                NIXL_INFO << "=== Test completed successfully! ===";
                return 0;
            }
        }
    }
    NIXL_ERROR << "Could not find topology spec for instance type " << instance_type;
    return 2;
}