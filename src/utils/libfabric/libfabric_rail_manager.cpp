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

#include "libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_topology.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include <sstream>
#include <algorithm>
#include <numaif.h>

#include <numa.h>

// Forward declaration for LibfabricUtils namespace
namespace LibfabricUtils {
uint16_t
getNextXferId();
uint8_t
getNextSeqId();
void
resetSeqId();
} // namespace LibfabricUtils

// Static round-robin counter for rail selection
static std::atomic<size_t> round_robin_counter{0};
static const std::string NUM_RAILS_TAG{"num_rails"};

// retrieves the NUMA node id of a given memory buffer
static bool
getMemNumaNode(void *buffer, size_t &numa_node_id);

// all-rail selection policy for DRAM_SEG memory registration
class nixlLibfabricAllRailSelectionPolicy : public nixlLibfabricRailSelectionPolicy {
public:
    nixlLibfabricAllRailSelectionPolicy() : rail_count_(0) {}

    ~nixlLibfabricAllRailSelectionPolicy() override {}

    // load policy
    bool
    load(nixlLibfabricRailManager &rail_manager) override;

    // select rails for memory region
    bool
    selectRails(void *buffer, std::vector<size_t> &selected_rails) override;

    // utility static member function for selecting all rails (avoid code duplication)
    static void
    selectAllRails(std::vector<size_t> &selected_rails, size_t rail_count);

private:
    // number of rails
    size_t rail_count_;
};

// define atomic wrapper to be able to use in vector
template<typename T> struct atomic_wrapper {
    atomic_wrapper() : value_(0) {}

    ~atomic_wrapper() {}

    atomic_wrapper(const atomic_wrapper &other) {
        value_.store(other.value_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    atomic_wrapper(atomic_wrapper &&) = delete;

    inline atomic_wrapper &
    operator=(const atomic_wrapper &other) {
        value_.store(other.value_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    std::atomic<T> value_;
};

// NOTE: in order to avoid complex implementation, we use rail count instead of bandwidth - this
// greatly simplifies the rail selection logic (assumes topology is symmetric and uniform)

// NUMA-aware rail selection policy for DRAM_SEG memory registration:
// - selects a restricted amount of rails from the rails that are close to the NUMA node of the
//   registered memory buffer
// - if possible, make sure rail selection does not saturate PCIe switch capacity
// - rails are selected in round-robin fashion (switch by switch of the same NUMA node)
// - if all switches are fully utilized then select extra rails from current NUMA node, each rail
// from the next switch, switch by switch
// - if all rails for the current NUMA node were used, then repeat the same logic with the closest
// NUMA node
class nixlLibfabricNumaRailSelectionPolicy : public nixlLibfabricRailSelectionPolicy {
public:
    nixlLibfabricNumaRailSelectionPolicy(size_t max_rails) : max_rails_(max_rails) {}

    ~nixlLibfabricNumaRailSelectionPolicy() override {}

    // load policy
    bool
    load(nixlLibfabricRailManager &rail_manager) override;

    // select rails for memory region
    bool
    selectRails(void *buffer, std::vector<size_t> &selected_rails) override;

private:
    // NOTE: rail selection is partitioned into switches, such that in each group the number of
    // rails selected does not exceed parent switch capacity

    // maximum number of rails per-NUMA node for a single MR
    size_t max_rails_;

    // single switch rail selection data
    struct SwitchRailData {
        SwitchRailData() : max_rails_(0) {}

        // maximum number of rails for this PCIe switch before it reaches maximum capacity
        size_t max_rails_;

        // data rail arrays per NUMA node - this includes all rails attached to the NUMA node (from
        // which some rails are selected each time)
        std::vector<size_t> rail_ids_;

        // the rolling base index of the next set of rails
        atomic_wrapper<uint64_t> next_rail_index_;
    };

    // rail data per NUMA node
    struct NumaRailData {
        NumaRailData() : rail_count_(0) {}

        // total number of rails on all PCIe switches attached to this NUMA node
        size_t rail_count_;

        // switch rail data array for this NUMA node
        std::vector<SwitchRailData> switch_data_;
    };

    // rail selection data - per NUMA node (grouped by switch)
    std::vector<NumaRailData> numa_data_rails_;

    // per node numa distance array
    std::vector<std::vector<size_t>> numa_distance_map_;

    // retrieves the NUMA node with which a memory buffer is associated
    bool
    getBufferNumaNode(void *buffer, size_t &numa_node_id);

    // selects rails from a given node, first by reaching each switch capacity, then adding rails as
    // needed from each switch, until all rails are used, or required number of rails was selected
    bool
    selectNodeRails(size_t numa_node_id, std::vector<size_t> &selected_rails);

    // selects all available rails on the given node
    bool
    selectAllNodeRails(size_t numa_node_id, std::vector<size_t> &selected_rails);

    // selects rails from a single switch (up to switch capacity)
    // returns true if number of required rails was selected
    bool
    selectSwitchRails(SwitchRailData &switch_rail_data, std::vector<size_t> &selected_rails);

    // selects extra rail from node switches, returns true if number of required rails was selected
    bool
    selectExtraNodeRails(size_t numa_node_id, std::vector<size_t> &selected_rails);

    // selects extra rail form a single switch
    // returns true if number of required rails was selected
    bool
    selectExtraSwitchRail(SwitchRailData &switch_rail_data, std::vector<size_t> &selected_rails);

    // builds rail data ready for applying selection logic
    bool
    buildNumaDataRails(const nixlLibfabricTopology *topology);

    // builds the NUMA distance map - used in case user specifies bandwidth limitation that exceeds
    // a single NUMA node capacity
    void
    buildNumaDistanceMap();
};

nixlLibfabricRailManager::nixlLibfabricRailManager(size_t striping_threshold)
    : striping_threshold_(striping_threshold),
      runtime_(FI_HMEM_CUDA) {
    NIXL_DEBUG << "Creating rail manager with striping threshold: " << striping_threshold_
               << " bytes";

    // Initialize topology system
    topology = std::make_unique<nixlLibfabricTopology>();

    // Determine system runtime type once at initialization
    if (topology->getNumNvidiaAccel() > 0) {
        runtime_ = FI_HMEM_CUDA;
        NIXL_INFO << "System runtime: CUDA for " << topology->getNumNvidiaAccel()
                  << " NVIDIA GPU(s)";
    } else if (topology->getNumAwsAccel() > 0) {
        runtime_ = FI_HMEM_NEURON;
        NIXL_INFO << "System runtime: NEURON for " << topology->getNumAwsAccel()
                  << " AWS Trainium device(s)";
    } else {
        runtime_ = FI_HMEM_SYSTEM;
        NIXL_INFO << "System runtime: SYSTEM (no accelerators)";
    }

    // Get network devices from topology and create rails automatically
    std::vector<std::string> all_devices = topology->getAllDevices();

    std::string selected_provider_name = topology->getProviderName();

    NIXL_DEBUG << "Got " << all_devices.size()
               << " network devices from topology for provider=" << selected_provider_name;

    // Create data rails with selected provider - throw on failure
    nixl_status_t rail_status = createDataRails(all_devices, selected_provider_name);
    if (rail_status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create data rails for libfabric rail manager");
    }

    // Create control rails with selected provider - throw on failure
    nixl_status_t control_rail_status = createControlRails(
        all_devices, selected_provider_name, NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS);
    if (control_rail_status != NIXL_SUCCESS) {
        throw std::runtime_error("Failed to create control rails for libfabric rail manager");
    }

    NIXL_DEBUG << "Successfully created " << data_rails_.size() << " data rails and "
               << control_rails_.size()
               << " control rails using provider=" << selected_provider_name;
}

nixlLibfabricRailManager::~nixlLibfabricRailManager() {
    NIXL_DEBUG << "Destroying rail manager";
}

nixl_status_t
nixlLibfabricRailManager::init(const nixl_b_params_t &custom_params) {
    // load from config or compute bandwidth limit per NUMA node
    // then from that deduce rail count limit
    // finally choose appropriate rail selection policy:
    // 1 - limit is within a single NUMA node bound --> use numa-aware policy
    // 2 - limit exceeds NUMA node capacity, but not machine capacity --> use numa-aware policy
    // 3 - limit exceeds entire machine capcaity --> use default (all) policy
    // in case no EFA device is found, we use default policy
    // in case machine is "regular" (no NUMA nodes) it is regarded as a single NUMA node machine
    // NOTE: in non-uniform topology, the selection policy should have been per NUMA node, but that
    // is not a use case we relate to for now

    // get bandwidth/rail limit from user or compute it, then select policy
    size_t max_bw = 0;
    size_t max_rails = 0;
    size_t nic_speed = topology->getAvgNicBandwidth();
    if (!getDramRailLimit(custom_params, max_bw, max_rails) || max_rails == 0) {
        // had some error in deducing rail count, so just use default policy
        NIXL_WARN << "Using default (all) rail selection policy for DRAM memory type due to "
                     "previous errors";
        dram_rail_selection_policy_ = std::make_unique<nixlLibfabricAllRailSelectionPolicy>();
    } else if (max_rails < topology->getTotalNicCount()) {
        // bandwidth does not exceed total machine capacity, so use NUMA-aware rail selection policy
        NIXL_TRACE << "Using NUMA-aware rail selection policy for DRAM memory type";
        size_t numa_rail_count = topology->getNumaRailCount(); // NOTE: averaged if non-uniform
        if (max_rails > numa_rail_count) {
            size_t numa_speed = numa_rail_count * nic_speed;
            NIXL_WARN << "User-provided configuration value for max_bw_per_dram_seg (" << max_bw
                      << " Gbps) exceeds single NUMA node capacity of " << numa_speed
                      << " Gbps, and will spill over to other NUMA nodes";
        }
        dram_rail_selection_policy_ =
            std::make_unique<nixlLibfabricNumaRailSelectionPolicy>(max_rails);
    } else {
        // this must be coming from user, it exceeds total machine capacity
        size_t total_nic_speed = nic_speed * topology->getTotalNicCount();
        NIXL_WARN << "User-provided configuration value for max_bw_per_dram_seg (" << max_bw
                  << " Gbps) exceeds or equals to total machine capacity of " << total_nic_speed
                  << " Gbps, and will use all available rails";
        NIXL_WARN << "Using default (all) rail selection policy for DRAM_SEG memory type";
        dram_rail_selection_policy_ = std::make_unique<nixlLibfabricAllRailSelectionPolicy>();
    }

    // load policy
    if (!dram_rail_selection_policy_->load(*this)) {
        NIXL_WARN << "Failed to load DRAM_SEG rail selection policy, using default policy";
        dram_rail_selection_policy_ = std::make_unique<nixlLibfabricAllRailSelectionPolicy>();
        if (!dram_rail_selection_policy_->load(*this)) {
            NIXL_ERROR << "Failed to load default DRAM_SEG rail selection policy (internal error)";
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::createDataRails(const std::vector<std::string> &efa_devices,
                                          const std::string &provider_name) {
    num_data_rails_ = efa_devices.size();
    // Pre-allocate to ensure contiguous memory allocation
    data_rails_.reserve(num_data_rails_);

    // Build EFA device to rail index mapping for O(1) lookup
    efa_device_to_rail_map.reserve(num_data_rails_);

    try {
        data_rails_.clear();
        data_rails_.reserve(num_data_rails_);

        for (size_t i = 0; i < num_data_rails_; ++i) {
            data_rails_.emplace_back(std::make_unique<nixlLibfabricRail>(
                efa_devices[i], provider_name, static_cast<uint16_t>(i)));

            // Initialize EFA device mapping
            efa_device_to_rail_map[efa_devices[i]] = i;

            NIXL_DEBUG << "Created data rail " << i << " (device=" << efa_devices[i]
                       << ", provider=" << provider_name << ")";
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create data rails: " << e.what();
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::createControlRails(const std::vector<std::string> &efa_devices,
                                             const std::string &provider_name,
                                             size_t num_control_rails) {
    // Pre-allocate to ensure contiguous memory allocation
    num_control_rails_ = num_control_rails;
    control_rails_.reserve(num_control_rails_);

    try {
        control_rails_.clear();
        control_rails_.reserve(num_control_rails_);

        for (size_t i = 0; i < num_control_rails_; ++i) {
            control_rails_.emplace_back(std::make_unique<nixlLibfabricRail>(
                efa_devices[i], provider_name, static_cast<uint16_t>(i)));
            NIXL_DEBUG << "Created control rail " << i << " (device=" << efa_devices[i]
                       << ", provider=" << provider_name << ")";
        }
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to create control rails: " << e.what();
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

bool
nixlLibfabricRailManager::shouldUseStriping(size_t transfer_size) const {
    return transfer_size >= striping_threshold_;
}

nixl_status_t
nixlLibfabricRailManager::prepareAndSubmitTransfer(
    nixlLibfabricReq::OpType op_type,
    void *local_addr,
    size_t transfer_size,
    uint64_t remote_base_addr,
    const std::vector<size_t> &selected_rails,
    const std::vector<struct fid_mr *> &local_mrs,
    const std::vector<uint64_t> &remote_keys,
    const std::vector<size_t> &remote_selected_endpoints,
    const std::unordered_map<size_t, std::vector<fi_addr_t>> &dest_addrs,
    uint16_t agent_idx,
    uint16_t xfer_id,
    std::function<void()> completion_callback,
    size_t &submitted_count_out) {
    // Initialize output parameter
    submitted_count_out = 0;

    if (selected_rails.empty()) {
        NIXL_ERROR << "No rails selected for transfer";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Determine striping strategy
    bool use_striping = shouldUseStriping(transfer_size) && selected_rails.size() > 1;
    NIXL_DEBUG << "use_striping=" << use_striping;
    if (!use_striping) {
        // Round-robin: use one rail for entire transfer
        const auto counter_value = round_robin_counter.fetch_add(1);
        const size_t rail_id = selected_rails[counter_value % selected_rails.size()];
        const size_t remote_ep_id =
            remote_selected_endpoints[counter_value % remote_selected_endpoints.size()];
        NIXL_DEBUG << "rail " << rail_id << ", remote_ep_id " << remote_ep_id;
        // Allocate request
        nixlLibfabricReq *req = data_rails_[rail_id]->allocateDataRequest(op_type, xfer_id);
        if (!req) {
            NIXL_ERROR << "Failed to allocate request for rail " << rail_id;
            return NIXL_ERR_BACKEND;
        }
        // Set completion callback and populate request
        req->completion_callback = completion_callback;
        req->chunk_offset = 0;
        req->chunk_size = transfer_size;
        req->local_addr = local_addr;

        // For TCP providers, use offset 0 instead of virtual address
        // TCP providers don't support FI_MR_VIRT_ADDR and expect offset-based addressing
        if (data_rails_[rail_id]->getRailInfo()->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
            req->remote_addr = remote_base_addr; // Use virtual address for EFA and other providers
        } else {
            req->remote_addr = 0; // Use offset 0 for TCP providers
            NIXL_DEBUG << "TCP provider detected: using offset 0 instead of virtual address "
                       << (void *)remote_base_addr << " for rail " << rail_id;
        }

        req->local_mr = local_mrs[rail_id];
        req->remote_key = remote_keys[remote_ep_id];
        req->rail_id = rail_id;
        // Submit immediately
        nixl_status_t status;
        if (op_type == nixlLibfabricReq::WRITE) {
            // Generate next SEQ_ID for this specific write operation
            uint8_t seq_id = LibfabricUtils::getNextSeqId();
            uint64_t imm_data =
                NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, agent_idx, xfer_id, seq_id);
            status = data_rails_[rail_id]->postWrite(req->local_addr,
                                                     req->chunk_size,
                                                     fi_mr_desc(req->local_mr),
                                                     imm_data,
                                                     dest_addrs.at(rail_id)[remote_ep_id],
                                                     req->remote_addr,
                                                     req->remote_key,
                                                     req);
        } else {
            status = data_rails_[rail_id]->postRead(req->local_addr,
                                                    req->chunk_size,
                                                    fi_mr_desc(req->local_mr),
                                                    dest_addrs.at(rail_id)[remote_ep_id],
                                                    req->remote_addr,
                                                    req->remote_key,
                                                    req);
        }
        if (status != NIXL_SUCCESS) {
            // Release the allocated request back to pool on failure
            data_rails_[rail_id]->releaseRequest(req);
            NIXL_ERROR << "Failed to submit "
                       << (op_type == nixlLibfabricReq::WRITE ? "write" : "read") << " on rail "
                       << rail_id << ", request released";
            return status;
        }

        // Track submitted request
        submitted_count_out = 1;

        NIXL_DEBUG << "Round-robin: submitted single request on rail " << rail_id << " for "
                   << transfer_size << " bytes, XFER_ID=" << req->xfer_id;

    } else {
        // Striping: distribute across multiple rails
        size_t num_rails = selected_rails.size();
        size_t chunk_size = transfer_size / num_rails;
        size_t remainder = transfer_size % num_rails;
        for (size_t i = 0; i < num_rails; ++i) {
            const size_t rail_id = selected_rails[i];
            const size_t remote_ep_id =
                remote_selected_endpoints[i % remote_selected_endpoints.size()];
            NIXL_DEBUG << "rail " << rail_id << ", remote_ep_id=" << remote_ep_id;
            size_t current_chunk_size = chunk_size + (i == num_rails - 1 ? remainder : 0);
            if (current_chunk_size == 0) break;
            // Allocate request
            nixlLibfabricReq *req = data_rails_[rail_id]->allocateDataRequest(op_type, xfer_id);
            if (!req) {
                NIXL_ERROR << "Failed to allocate request for rail " << rail_id;
                return NIXL_ERR_BACKEND;
            }

            req->completion_callback = completion_callback;

            // Calculate and populate chunk info
            size_t chunk_offset = i * chunk_size;
            req->chunk_offset = chunk_offset;
            req->chunk_size = current_chunk_size;
            req->local_addr = static_cast<char *>(local_addr) + chunk_offset;

            // For TCP providers, use offset instead of virtual address
            // TCP providers don't support FI_MR_VIRT_ADDR and expect offset-based addressing
            if (data_rails_[rail_id]->getRailInfo()->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
                req->remote_addr = remote_base_addr +
                    chunk_offset; // Use virtual address for EFA and other providers
            } else {
                req->remote_addr = chunk_offset; // Use chunk offset for TCP providers
                NIXL_DEBUG << "TCP provider detected: using chunk offset " << chunk_offset
                           << " instead of virtual address "
                           << (void *)(remote_base_addr + chunk_offset) << " for rail " << rail_id;
            }

            req->local_mr = local_mrs[rail_id];
            req->remote_key = remote_keys[remote_ep_id];
            req->rail_id = rail_id;
            nixl_status_t status;
            if (op_type == nixlLibfabricReq::WRITE) {
                // Generate next SEQ_ID for this specific transfer operation
                uint8_t seq_id = LibfabricUtils::getNextSeqId();
                uint64_t imm_data =
                    NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, agent_idx, xfer_id, seq_id);
                status = data_rails_[rail_id]->postWrite(req->local_addr,
                                                         req->chunk_size,
                                                         fi_mr_desc(req->local_mr),
                                                         imm_data,
                                                         dest_addrs.at(rail_id)[remote_ep_id],
                                                         req->remote_addr,
                                                         req->remote_key,
                                                         req);
            } else {
                status = data_rails_[rail_id]->postRead(req->local_addr,
                                                        req->chunk_size,
                                                        fi_mr_desc(req->local_mr),
                                                        dest_addrs.at(rail_id)[remote_ep_id],
                                                        req->remote_addr,
                                                        req->remote_key,
                                                        req);
            }
            if (status != NIXL_SUCCESS) {
                // This request failed to submit - release it immediately
                data_rails_[rail_id]->releaseRequest(req);
                NIXL_ERROR << "Failed to submit "
                           << (op_type == nixlLibfabricReq::WRITE ? "write" : "read") << " on rail "
                           << rail_id << ", request released";
                return status;
            }

            // Track submitted request
            submitted_count_out++;
        }
        NIXL_DEBUG << "Striping: submitted " << submitted_count_out << " requests for "
                   << transfer_size << " bytes";
    }

    NIXL_DEBUG << "Successfully submitted requests for " << transfer_size << " bytes";

    return NIXL_SUCCESS;
}

bool
nixlLibfabricRailManager::getDramRailLimit(const nixl_b_params_t &custom_params,
                                           size_t &max_bw,
                                           size_t &max_rails) {
    // first make sure there are any EFA devices
    if (topology->getTotalNicCount() == 0) {
        NIXL_WARN << "Could not find EFA devices, rail selection for DRAM memory type aborted";
        return false;
    }

    // verify a few more computed values before continuing (avoid division by zero  )
    size_t nic_speed = topology->getAvgNicBandwidth();
    if (nic_speed == 0) {
        NIXL_WARN << "Could not deduce average EFA device line bandwidth, NUMA-aware rail "
                     "selection for DRAM memory type aborted";
        return false;
    }
    size_t nic_upstream_speed = topology->getAvgNicUpstreamBandwidth();
    if (nic_upstream_speed == 0) {
        NIXL_WARN << "Could not deduce average EFA device upstream link bandwidth, NUMA-aware "
                     "rail selection for DRAM memory type aborted";
        return false;
    }

    // now compute rail limit based on bandwidth limit
    max_rails = 0;
    max_bw = 0;

    // get bandwidth from configuration or environment variable, and then deduce rail count
    // NOTE: corresponding env var is NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG, and is specified in
    // Gigabits per second also return value from getCustomIntParam() is not that important
    nixl_status_t res =
        LibfabricUtils::getCustomIntParam(custom_params, "max_bw_per_dram_seg", max_bw);
    if (res != NIXL_SUCCESS) {
        // bandwidth limit could not be obtained from user (either user did not specify, or
        // configuration was malformed) so comput bandwidth limit from topology, and then deduce
        // rail count
        max_bw = topology->getAvgNumaNodeBandwidth();
        if (max_bw == 0) {
            NIXL_WARN << "Could not deduce average bandwidth limit per NUMA node, NUMA-aware rail "
                         "selection for DRAM type aborted";
            return false;
        }
        // NOTE: when rail limit is computed, we divide switch capacity by NIC upstream link
        max_rails = max_bw / nic_upstream_speed;
        NIXL_TRACE << "Computed (average) NUMA node combined PCIe switch bandwidth limit is "
                   << max_bw << " Gbps";
        NIXL_TRACE << "Computed (average) rails per NUMA node is " << max_rails;
    } else {
        // print warning if bandwidth limit provided by user is less than the speed of a single NIC
        if (max_bw < nic_speed) {
            NIXL_WARN << "User-provided configuration value for max_bw_per_dram_seg (" << max_bw
                      << " Gbps) falls below a single NIC speed (" << nic_speed
                      << " Gbps). Bandwidth limitation rectified to " << nic_speed << " Gbps.";
            max_bw = nic_speed;
        }

        // now compute rail count
        // NOTE: when computing rail count based on user input, we divide bandwidth limit by NIC
        // line speed, and not by NIC upstream link speed, because user specifies limit in whole
        // units (e.g. 200, 400, etc.), just as fi_info reports NIC line speed. On the other hand,
        // hwloc reports link speeds in GB/s as float (i.e. 31.5077 GB/s), which is not suitable.
        max_rails = max_bw / nic_speed;

        // print warning if bandwidth limit provided by user is not a whole multiple of NIC speed
        if (max_bw % nic_speed != 0) {
            NIXL_WARN << "User-provided configuration value for max_bw_per_dram_seg (" << max_bw
                      << " Gbps) is not aligned with NIC speed [" << nic_speed
                      << " Gbps). Bandwidth limitation truncated to " << max_rails * nic_speed
                      << " Gbps.";
        }
        NIXL_TRACE << "Computed (from user bandwidth) rails per NUMA node is " << max_rails;
    }
    NIXL_TRACE << "Setting DRAM rail limitation to " << max_rails << " per NUMA node";
    return true;
}

std::vector<size_t>
nixlLibfabricRailManager::selectRailsForMemory(void *mem_addr,
                                               nixl_mem_t mem_type,
                                               int device_id,
                                               const std::string &device_pci_bus_id) const {
    if (mem_type == VRAM_SEG) {
        if (device_id < 0) {
            NIXL_ERROR << "Invalid device ID " << device_id << " for VRAM memory " << mem_addr;
            return {}; // Return empty vector to indicate failure
        }

        // Get EFA devices for this PCI bus ID
        std::vector<std::string> device_efa_devices =
            topology->getEfaDevicesForPci(device_pci_bus_id);
        if (device_efa_devices.empty()) {
            NIXL_ERROR << "No EFA devices found for PCI " << device_pci_bus_id;
            return {}; // Return empty vector to indicate failure
        }
        std::vector<size_t> device_rails;
        for (const std::string &efa_device : device_efa_devices) {
            auto it = efa_device_to_rail_map.find(efa_device);
            if (it != efa_device_to_rail_map.end()) {
                // Bounds check: ensure rail index is valid
                if (it->second < data_rails_.size()) {
                    device_rails.push_back(it->second);
                    NIXL_DEBUG << "VRAM memory " << mem_addr << " on device PCI "
                               << device_pci_bus_id << " mapped to rail " << it->second
                               << " (EFA device=" << efa_device << ")";
                } else {
                    NIXL_WARN << "EFA device " << efa_device << " maps to rail " << it->second
                              << " but only " << data_rails_.size() << " rails available";
                }
            } else {
                NIXL_WARN << "EFA device " << efa_device
                          << " not found in rail mapping for device PCI " << device_pci_bus_id;
            }
        }

        if (device_rails.empty()) {
            NIXL_ERROR << "No valid rail mapping found for device PCI " << device_pci_bus_id
                       << " (checked " << device_efa_devices.size() << " EFA devices)";
            return {};
        }

        NIXL_DEBUG << "VRAM memory " << mem_addr << " on device PCI " << device_pci_bus_id
                   << " will use " << device_rails.size() << " rails total";
        return device_rails;
    }
    if (mem_type == DRAM_SEG) {
        std::vector<size_t> selected_rails;
        if (dram_rail_selection_policy_.get() == nullptr ||
            !dram_rail_selection_policy_->selectRails(mem_addr, selected_rails)) {
            NIXL_WARN << "Failed to select rails for DRAM_SEG according to policy, defaulting to "
                         "all rails";
            // default to all rails
            nixlLibfabricAllRailSelectionPolicy::selectAllRails(selected_rails,
                                                                topology->getAllDevices().size());
        } else {
            NIXL_TRACE << "Selected rails " << LibfabricUtils::railIdsToString(selected_rails)
                       << " for registration of memory buffer at " << std::hex << mem_addr;
        }
        return selected_rails;
    }

    // For unsupported memory types, return empty vector
    NIXL_ERROR << "Unsupported memory type " << mem_type;
    return {};
}

nixl_status_t
nixlLibfabricRailManager::registerMemory(void *buffer,
                                         size_t length,
                                         nixl_mem_t mem_type,
                                         int device_id,
                                         const std::string &device_pci_bus_id,
                                         std::vector<struct fid_mr *> &mr_list_out,
                                         std::vector<uint64_t> &key_list_out,
                                         std::vector<size_t> &selected_rails_out) {
    if (!buffer) {
        NIXL_ERROR << "Invalid buffer parameter";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Select rails based on memory type and PCI bus ID
    // For VRAM: uses PCI bus ID provided by backend to map to topology-aware rails
    // For DRAM: uses all available rails
    std::vector<size_t> selected_rails =
        selectRailsForMemory(buffer, mem_type, device_id, device_pci_bus_id);
    if (selected_rails.empty()) {
        NIXL_ERROR << "No rails selected for memory type " << mem_type;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    enum fi_hmem_iface iface = FI_HMEM_SYSTEM;
    if (mem_type == VRAM_SEG) {
        iface = topology->getMrAttrIface(device_id);
    }

    // Resize output vectors to match all rails
    mr_list_out.resize(data_rails_.size(), nullptr);
    key_list_out.clear();
    key_list_out.resize(data_rails_.size(), FI_KEY_NOTAVAIL);
    selected_rails_out = selected_rails; // Return which rails were selected

    // Register memory on each selected rail
    for (size_t i = 0; i < selected_rails.size(); ++i) {
        size_t rail_idx = selected_rails[i];
        if (rail_idx >= data_rails_.size()) {
            NIXL_ERROR << "Invalid rail index " << rail_idx;
            // Cleanup already registered MRs
            for (size_t cleanup_idx : selected_rails) {
                if (cleanup_idx >= rail_idx) break; // Only cleanup what we've done so far
                if (mr_list_out[cleanup_idx]) {
                    data_rails_[cleanup_idx]->deregisterMemory(mr_list_out[cleanup_idx]);
                    mr_list_out[cleanup_idx] = nullptr;
                }
            }
            return NIXL_ERR_INVALID_PARAM;
        }

        struct fid_mr *mr;
        uint64_t key;
        // Pass device_id parameter to individual rail's registerMemory calls
        nixl_status_t status = data_rails_[rail_idx]->registerMemory(
            buffer, length, mem_type, device_id, iface, &mr, &key);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register memory on rail " << rail_idx;
            // Cleanup already registered MRs
            for (size_t cleanup_idx : selected_rails) {
                if (cleanup_idx >= rail_idx) break; // Only cleanup what we've done so far
                if (mr_list_out[cleanup_idx]) {
                    data_rails_[cleanup_idx]->deregisterMemory(mr_list_out[cleanup_idx]);
                    mr_list_out[cleanup_idx] = nullptr;
                }
            }
            return status;
        }

        mr_list_out[rail_idx] = mr;
        key_list_out[rail_idx] = key;

        // Mark rail as active for progress tracking optimization
        markRailActive(rail_idx);

        NIXL_DEBUG << "Registered memory on rail " << rail_idx
                   << " (mr=" << static_cast<const void *>(mr) << ", key=" << key << ")";
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::deregisterMemory(const std::vector<size_t> &selected_rails,
                                           const std::vector<struct fid_mr *> &mr_list) {
    if (selected_rails.empty() || mr_list.size() != data_rails_.size()) {
        NIXL_ERROR << "Invalid parameters";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t overall_status = NIXL_SUCCESS;

    for (size_t i = 0; i < selected_rails.size(); ++i) {
        size_t rail_idx = selected_rails[i];
        if (rail_idx >= data_rails_.size()) {
            NIXL_ERROR << "Invalid rail index " << rail_idx;
            overall_status = NIXL_ERR_INVALID_PARAM;
            continue;
        }

        if (mr_list[rail_idx]) {
            nixl_status_t status = data_rails_[rail_idx]->deregisterMemory(mr_list[rail_idx]);
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to deregister memory on rail " << rail_idx;
                overall_status = status;
            }
            markRailInactive(rail_idx);
        }
    }

    return overall_status;
}

nixl_status_t
nixlLibfabricRailManager::insertAllAddresses(
    RailType rail_type,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints,
    std::unordered_map<size_t, std::vector<fi_addr_t>> &fi_addrs_out,
    std::vector<char *> &ep_names_out) {
    auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
    const char *rail_type_str = (rail_type == RailType::DATA) ? "data" : "control";

    fi_addrs_out.clear();
    ep_names_out.clear();
    ep_names_out.reserve(rails.size());

    // Process all rails in one operation
    for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
        fi_addrs_out[rail_id].reserve(endpoints.size());
        for (const auto &endpoint : endpoints) {
            fi_addr_t fi_addr;
            nixl_status_t status = rails[rail_id]->insertAddress(endpoint.data(), &fi_addr);
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed for " << rail_type_str << " rail " << rail_id;
                return status;
            }
            fi_addrs_out[rail_id].push_back(fi_addr);
            NIXL_DEBUG << "Processed " << rail_type_str << " rail " << rail_id
                       << " (fi_addr=" << fi_addr << ")";
        }

        ep_names_out.push_back(
            rails[rail_id]
                ->ep_name); // This is char[LF_EP_NAME_MAX_LEN], will be converted to char*
    }

    NIXL_DEBUG << "Successfully processed " << rails.size() << " " << rail_type_str << " rails";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::cleanupConnection(RailType rail_type,
                                            const std::vector<fi_addr_t> &fi_addrs_to_remove) {
    auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
    const char *rail_type_str = (rail_type == RailType::DATA) ? "data" : "control";

    if (fi_addrs_to_remove.size() != rails.size()) {
        NIXL_ERROR << "Expected " << rails.size() << " " << rail_type_str << " fi_addrs, got "
                   << fi_addrs_to_remove.size();
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_DEBUG << "Cleaning up connection for " << rails.size() << " " << rail_type_str << " rails";
    // Remove addresses from all rails
    nixl_status_t overall_status = NIXL_SUCCESS;
    for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
        if (fi_addrs_to_remove[rail_id] != FI_ADDR_UNSPEC) {
            nixl_status_t status = rails[rail_id]->removeAddress(fi_addrs_to_remove[rail_id]);
            if (status != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to remove address from " << rail_type_str << " rail "
                           << rail_id << ", fi_addr=" << fi_addrs_to_remove[rail_id];
                overall_status = status;
                // Continue cleanup for other rails even if one fails
            } else {
                NIXL_DEBUG << "Successfully removed address from " << rail_type_str << " rail "
                           << rail_id << ", fi_addr=" << fi_addrs_to_remove[rail_id];
            }
        } else {
            NIXL_DEBUG << "Skipping FI_ADDR_UNSPEC for " << rail_type_str << " rail " << rail_id;
        }
    }
    NIXL_DEBUG << "Completed cleanup for " << rails.size() << " " << rail_type_str << " rails";
    return overall_status;
}

nixl_status_t
nixlLibfabricRailManager::postControlMessage(ControlMessageType msg_type,
                                             nixlLibfabricReq *req,
                                             fi_addr_t dest_addr,
                                             uint16_t agent_idx,
                                             std::function<void()> completion_callback) {
    // Validation
    if (control_rails_.empty()) {
        NIXL_ERROR << "No control rails available";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!req) {
        NIXL_ERROR << "Pre-allocated request is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    uint64_t msg_type_value;
    switch (msg_type) {
    case ControlMessageType::NOTIFICATION:
        msg_type_value = NIXL_LIBFABRIC_MSG_NOTIFICTION;
        break;
    default:
        NIXL_ERROR << "Unknown message type";
        return NIXL_ERR_INVALID_PARAM;
    }
    size_t control_rail_id = 0;
    uint32_t xfer_id = req->xfer_id;
    // For control messages, use SEQ_ID 0 since they don't need sequence tracking
    // TODO: Add sequencing for connection establishment workflow.
    uint64_t imm_data = NIXL_MAKE_IMM_DATA(msg_type_value, agent_idx, xfer_id, 0);

    // Set completion callback if provided
    if (completion_callback) {
        req->completion_callback = completion_callback;
        NIXL_DEBUG << "Set completion callback for control message request " << req->xfer_id;
    }

    NIXL_DEBUG << "Sending control message type " << msg_type_value << " agent_idx=" << agent_idx
               << " XFER_ID=" << xfer_id << " imm_data=" << imm_data;

    // Rail postSend
    nixl_status_t status = control_rails_[control_rail_id]->postSend(imm_data, dest_addr, req);

    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to send control message type " << static_cast<int>(msg_type)
                   << " on control rail " << control_rail_id;
        // Release the pre-allocated control request back to pool on failure
        control_rails_[control_rail_id]->releaseRequest(req);
        return status;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::progressActiveDataRails() {
    std::vector<size_t> rails_to_process;

    // Copy active rails under lock to avoid iterator invalidation
    {
        std::lock_guard<std::mutex> lock(active_rails_mutex_);
        if (active_rails_.empty()) {
            return NIXL_IN_PROG; // No active rails to process
        }
        rails_to_process.assign(active_rails_.begin(), active_rails_.end());
    }

    // Process rails without holding the lock
    bool any_completions = false;

    for (size_t rail_id : rails_to_process) {
        if (rail_id >= data_rails_.size()) {
            NIXL_ERROR << "Invalid active rail ID: " << rail_id;
            continue;
        }
        // Process completions on active data rails
        nixl_status_t status = data_rails_[rail_id]->progressCompletionQueue();
        if (status == NIXL_SUCCESS) {
            any_completions = true;
            NIXL_DEBUG << "Processed completions on active data rail " << rail_id;
        } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to process completions on active data rail " << rail_id;
            // Continue processing other active rails even if one fails
        }
    }

    if (any_completions) {
        NIXL_TRACE << "Processed " << rails_to_process.size() << " active rails, completions found";
    }

    return any_completions ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricRailManager::progressAllControlRails() {
    bool any_completions = false;
    for (size_t rail_id = 0; rail_id < num_control_rails_; ++rail_id) {
        nixl_status_t status = control_rails_[rail_id]->progressCompletionQueue();
        if (status == NIXL_SUCCESS) {
            any_completions = true;
            NIXL_DEBUG << "Processed completion on control rail " << rail_id;
        } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
            any_completions = true;
            NIXL_ERROR << "Failed to process completion on control rail " << rail_id;
            return NIXL_ERR_BACKEND;
        }
    }
    return any_completions ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricRailManager::validateAllRailsInitialized() {
    for (size_t rail_id = 0; rail_id < data_rails_.size(); ++rail_id) {
        if (!data_rails_[rail_id]->isProperlyInitialized()) {
            NIXL_ERROR << "Rail " << rail_id << " is not properly initialized";
            return NIXL_ERR_BACKEND;
        }
    }
    NIXL_DEBUG << "All " << data_rails_.size() << " rails are properly initialized";
    return NIXL_SUCCESS;
}

struct fid_mr *
nixlLibfabricRailManager::getMemoryDescriptor(size_t rail_id, struct fid_mr *mr) {
    if (rail_id >= data_rails_.size()) {
        NIXL_ERROR << "Invalid rail index " << rail_id;
        return nullptr;
    }
    return static_cast<struct fid_mr *>(data_rails_[rail_id]->getMemoryDescriptor(mr));
}

nixl_status_t
nixlLibfabricRailManager::serializeMemoryKeys(const std::vector<uint64_t> &keys,
                                              void *buffer,
                                              std::string &str) const {
    nixlSerDes ser_des;
    // Serialize all rail keys instead of just the first one
    for (size_t rail_id = 0; rail_id < keys.size(); ++rail_id) {
        std::string key_name = "key_" + std::to_string(rail_id);
        ser_des.addBuf(key_name.c_str(), &keys[rail_id], sizeof(keys[rail_id]));
    }

    ser_des.addBuf("addr", &buffer, sizeof(buffer));
    str = ser_des.exportStr();

    NIXL_DEBUG << "Serialized memory keys for " << keys.size() << " rails" << " (buffer=" << buffer
               << ", size=" << str.length() << " bytes)";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::deserializeMemoryKeys(const std::string &serialized_data,
                                                const size_t num_keys,
                                                std::vector<uint64_t> &keys_out,
                                                uint64_t &remote_addr_out) const {
    nixlSerDes ser_des;
    ser_des.importStr(serialized_data);
    // Load all rail keys instead of just one
    keys_out.clear();
    keys_out.reserve(num_keys);
    for (size_t idx = 0; idx < num_keys; ++idx) {
        std::string key_name = "key_" + std::to_string(idx);
        uint64_t remote_key;
        nixl_status_t status = ser_des.getBuf(key_name.c_str(), &remote_key, sizeof(remote_key));
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get key " << key_name;
            return NIXL_ERR_BACKEND;
        }
        keys_out.push_back(remote_key);
    }
    nixl_status_t addr_status = ser_des.getBuf("addr", &remote_addr_out, sizeof(remote_addr_out));
    if (addr_status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get remote address";
        return NIXL_ERR_BACKEND;
    }
    NIXL_DEBUG << "Deserialized memory keys for " << keys_out.size() << " rails"
               << " (remote addr: " << (void *)remote_addr_out << ")";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::serializeConnectionInfo(const std::string &user_prefix,
                                                  std::string &str) const {

    nixlSerDes ser_des;

    // Use user prefix with standard suffixes
    std::string data_prefix = user_prefix + "_data_ep_";
    std::string control_prefix = user_prefix + "_control_ep_";

    serializeRailEndpoints(ser_des, data_prefix, RailType::DATA);
    serializeRailEndpoints(ser_des, control_prefix, RailType::CONTROL);
    str = ser_des.exportStr();
    NIXL_DEBUG << "Connection info serialized with prefix " << user_prefix
               << ", size=" << str.length();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRailManager::deserializeConnectionInfo(
    const std::string &user_prefix,
    const std::string &serialized_data,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_endpoints_out,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &control_endpoints_out) const {

    nixlSerDes ser_des;
    ser_des.importStr(serialized_data);

    // Use user prefix with standard suffixes
    std::string data_prefix = user_prefix + "_data_ep_";
    std::string control_prefix = user_prefix + "_control_ep_";
    nixl_status_t data_status = deserializeRailEndpoints(ser_des, data_prefix, data_endpoints_out);
    if (data_status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize data rail endpoints with prefix: " << data_prefix;
        return data_status;
    }
    nixl_status_t control_status =
        deserializeRailEndpoints(ser_des, control_prefix, control_endpoints_out);
    if (control_status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize control rail endpoints with prefix: "
                   << control_prefix;
        return control_status;
    }

    NIXL_DEBUG << "Connection info deserialized with prefix " << user_prefix << ": "
               << data_endpoints_out.size() << " data endpoints, " << control_endpoints_out.size()
               << " control endpoints";
    return NIXL_SUCCESS;
}

void
nixlLibfabricRailManager::serializeRailEndpoints(nixlSerDes &ser_des,
                                                 const std::string &key_prefix,
                                                 RailType rail_type) const {
    auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
    const char *rail_type_str = (rail_type == RailType::DATA) ? "data" : "control";

    ser_des.addStr(NUM_RAILS_TAG, std::to_string(rails.size()));

    for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
        std::string rail_key = key_prefix + std::to_string(rail_id);
        const char *ep_name = rails[rail_id]->ep_name;
        size_t ep_name_len = sizeof(rails[rail_id]->ep_name);

        ser_des.addBuf(rail_key.c_str(), ep_name, ep_name_len);
    }

    NIXL_DEBUG << "Serialized " << rails.size() << " " << rail_type_str << " rail endpoints";
}

nixl_status_t
nixlLibfabricRailManager::deserializeRailEndpoints(
    nixlSerDes &ser_des,
    const std::string &key_prefix,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints_out) const {

    std::string str;
    unsigned long num_rails_val;
    try {
        str = ser_des.getStr(NUM_RAILS_TAG);
        num_rails_val = std::stoul(str);
        if (num_rails_val > std::numeric_limits<size_t>::max()) {
            NIXL_ERROR << "Key " << NUM_RAILS_TAG
                       << " value out of range (size_t): " << num_rails_val;
            return NIXL_ERR_BACKEND;
        }
    }
    catch (const std::invalid_argument &) {
        NIXL_ERROR << "Key " << NUM_RAILS_TAG << " not found or invalid.";
        return NIXL_ERR_BACKEND;
    }
    catch (const std::out_of_range &) {
        NIXL_ERROR << "Key " << NUM_RAILS_TAG << " value out of range (unsigned long): " << str;
        return NIXL_ERR_BACKEND;
    }
    const size_t num_rails = static_cast<size_t>(num_rails_val);
    endpoints_out.resize(num_rails);

    for (size_t rail_id = 0; rail_id < num_rails; ++rail_id) {
        std::string rail_key = key_prefix + std::to_string(rail_id);

        // First check if the key exists and get its length
        ssize_t actual_len = ser_des.getBufLen(rail_key);
        if (actual_len <= 0) {
            NIXL_ERROR << "Key " << rail_key << " not found or has invalid length=" << actual_len;
            return NIXL_ERR_BACKEND;
        }

        if (actual_len > (ssize_t)endpoints_out[rail_id].size()) {
            NIXL_ERROR << "Buffer too small for rail " << rail_id << ", need " << actual_len
                       << " bytes, have " << endpoints_out[rail_id].size();
            return NIXL_ERR_BACKEND;
        }

        // Get the actual data
        nixl_status_t status = ser_des.getBuf(rail_key, endpoints_out[rail_id].data(), actual_len);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get endpoint address for rail " << rail_id
                       << " with key=" << rail_key << ", status: " << status;
            return NIXL_ERR_BACKEND;
        }
    }

    NIXL_DEBUG << "Successfully deserialized " << num_rails << " rail endpoints.";
    return NIXL_SUCCESS;
}

void
nixlLibfabricRailManager::markRailActive(size_t rail_id) {
    if (rail_id >= data_rails_.size()) {
        NIXL_ERROR << "Invalid rail ID for markRailActive: " << rail_id;
        return;
    }

    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    bool was_inserted = active_rails_.insert(rail_id).second;

    if (was_inserted) {
        NIXL_DEBUG << "Marked rail " << rail_id
                   << " as active (total active: " << active_rails_.size() << ")";
    } else {
        NIXL_TRACE << "Rail " << rail_id << " was already active";
    }
}

void
nixlLibfabricRailManager::markRailInactive(size_t rail_id) {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    size_t erased = active_rails_.erase(rail_id);
    if (erased > 0) {
        NIXL_DEBUG << "Marked rail " << rail_id
                   << " as inactive (total active: " << active_rails_.size() << ")";
    } else {
        NIXL_TRACE << "Rail " << rail_id << " was not in active set";
    }
}

void
nixlLibfabricRailManager::clearActiveRails() {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    size_t cleared_count = active_rails_.size();
    active_rails_.clear();
    NIXL_DEBUG << "Cleared " << cleared_count << " active rails";
}

size_t
nixlLibfabricRailManager::getActiveRailCount() const {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    return active_rails_.size();
}

// System accelerator type getter
fi_hmem_iface
nixlLibfabricRailManager::getRuntime() const {
    return runtime_;
}

bool
getMemNumaNode(void *buffer, size_t &numa_node_id) {
    int node_id = -1;
    // NOTE: according to get_mempolicy() man page: "If no page has yet been allocated for the
    // specified address, get_mempolicy() will allocate a page as if the thread had performed a read
    // (load) access to that address, and return the ID of the node where that page was allocated."
    if (get_mempolicy(&node_id, nullptr, 0, buffer, MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        int err_code = errno;
        const size_t BUF_LEN = 256;
        char buf[BUF_LEN] = {};
        NIXL_ERROR << "Failed to get memory policy for buffer at " << std::hex << buffer
                   << ", system call get_mempolicy() failed: " << strerror_r(err_code, buf, BUF_LEN)
                   << " (error code: " << err_code << ")";
        return false;
    }

    // do more sanity validations
    NIXL_DEBUG << "Found NUMA node " << node_id << " for memory buffer at " << std::hex << buffer;
    int max_node = -1;
    if (!LibfabricUtils::getMaxNumaNode(max_node)) {
        return false;
    }
    if (node_id < 0 || node_id > max_node) {
        NIXL_ERROR << "Found invalid NUMA node id " << node_id << " for memory buffer at "
                   << std::hex << buffer;
        return false;
    }
    numa_node_id = (size_t)node_id;
    return true;
}

bool
nixlLibfabricAllRailSelectionPolicy::load(nixlLibfabricRailManager &rail_manager) {
    // use all rails
    rail_count_ = rail_manager.getNumDataRails();
    return true;
}

bool
nixlLibfabricAllRailSelectionPolicy::selectRails(void *buffer,
                                                 std::vector<size_t> &selected_rails) {
    // unused parameters
    (void)buffer;

    // select all rails
    selectAllRails(selected_rails, rail_count_);
    return true;
}

void
nixlLibfabricAllRailSelectionPolicy::selectAllRails(std::vector<size_t> &selected_rails,
                                                    size_t rail_count) {
    selected_rails.resize(rail_count);
    for (size_t i = 0; i < rail_count; ++i) {
        selected_rails[i] = i;
    }
}

bool
nixlLibfabricNumaRailSelectionPolicy::load(nixlLibfabricRailManager &rail_manager) {
    // avoid topology checks during runtime, and use instead prepared array of data rail indices for
    // this NUMA node
    if (!buildNumaDataRails(rail_manager.getTopology())) {
        return false;
    }
    buildNumaDistanceMap();
    return true;
}

bool
nixlLibfabricNumaRailSelectionPolicy::selectRails(void *buffer,
                                                  std::vector<size_t> &selected_rails) {
    // get NUMA node id of the buffer
    size_t numa_node_id = 0;
    if (!getBufferNumaNode(buffer, numa_node_id)) {
        return false;
    }

    // verify numa distance map can be used
    if (numa_node_id >= numa_distance_map_.size()) {
        NIXL_ERROR << "NUMA node " << numa_node_id << " for DRAM_SEG memory buffer at " << std::hex
                   << buffer << " is out of range";
        return false;
    }
    const std::vector<size_t> &node_order = numa_distance_map_[numa_node_id];
    if (node_order.empty()) {
        NIXL_ERROR << "Cannot select NUMA-aware rails for DRAM_SEG: NUMA distance map is empty";
        return false;
    }
    if (node_order[0] != numa_node_id) {
        NIXL_ERROR
            << "Cannot select NUMA-aware rails for DRAM_SEG: NUMA distance map is not in order";
        return false;
    }

    // select rails node by node according to distance from source node
    selected_rails.reserve(max_rails_);
    for (size_t i = 0; i < node_order.size(); ++i) {
        size_t node_id = node_order[i];
        if (selectNodeRails(node_id, selected_rails)) {
            break;
        }
    }

    // verify result
    if (selected_rails.size() > max_rails_) {
        // this indicates intenral error, and is not acceptable (due to duplicate rails)
        NIXL_ERROR
            << "NUMA-aware rail selection for DRAM_SEG memory type error: too many rails selected: "
            << selected_rails.size() << ", while should have selected " << max_rails_;
        return false;
    }

    // this indicates intenral error, but we allow it
    if (selected_rails.size() < max_rails_) {
        NIXL_WARN << "NUMA-aware rail selection for DRAM_SEG memory type incomplete: selected only "
                  << selected_rails.size() << " rails, while should have selected " << max_rails_;
    }
    return true;
}

bool
nixlLibfabricNumaRailSelectionPolicy::getBufferNumaNode(void *buffer, size_t &numa_node_id) {
    // get NUMA node id of the buffer
    // NOTE: we assume memory is locked and cannot be swapped out (and then swapped in on a page on
    // another node)
    if (!getMemNumaNode(buffer, numa_node_id)) {
        return false;
    }

    // sanity check
    if (numa_node_id >= numa_data_rails_.size()) {
        NIXL_ERROR << "Invalid NUMA node id " << numa_node_id << " found for memory buffer at "
                   << std::hex << buffer << ", while selecting data rails for memory registration";
        return false;
    }
    NIXL_DEBUG << "buffer at " << std::hex << buffer << " belongs to NUMA node " << std::dec
               << numa_node_id;
    return true;
}

bool
nixlLibfabricNumaRailSelectionPolicy::selectNodeRails(size_t numa_node_id,
                                                      std::vector<size_t> &selected_rails) {
    // check first if required number of rails takes all rails on all switches of this node
    size_t missing_rail_count = max_rails_ - selected_rails.size();
    if (missing_rail_count >= numa_data_rails_[numa_node_id].rail_count_) {
        return selectAllNodeRails(numa_node_id, selected_rails);
    }

    // otherwise select rails from each switch until reaching switch capacity
    std::vector<SwitchRailData> &switch_array = numa_data_rails_[numa_node_id].switch_data_;
    for (SwitchRailData &switch_rail_data : switch_array) {
        if (selectSwitchRails(switch_rail_data, selected_rails)) {
            break;
        }
    }

    // if not enough, then we need to round-robin on switches of current node, and select each time
    // another rail that was not chosen yet.
    if (selected_rails.size() < max_rails_) {
        selectExtraNodeRails(numa_node_id, selected_rails);
    }

    // return true if number of required rails was selected
    return selected_rails.size() == max_rails_;
}

// select all available rails on the given node
bool
nixlLibfabricNumaRailSelectionPolicy::selectAllNodeRails(size_t numa_node_id,
                                                         std::vector<size_t> &selected_rails) {
    std::vector<SwitchRailData> &switch_array = numa_data_rails_[numa_node_id].switch_data_;
    for (SwitchRailData &switch_rail_data : switch_array) {
        const std::vector<size_t> &switch_rails = switch_rail_data.rail_ids_;
        selected_rails.insert(selected_rails.end(), switch_rails.begin(), switch_rails.end());
    }
    return selected_rails.size() == max_rails_;
}

bool
nixlLibfabricNumaRailSelectionPolicy::selectSwitchRails(SwitchRailData &switch_rail_data,
                                                        std::vector<size_t> &selected_rails) {
    // verify input
    if (selected_rails.size() > max_rails_) {
        NIXL_ERROR << "Too many rails selected (internal error): selected " << selected_rails.size()
                   << " while required " << max_rails_;
        return false;
    }
    if (selected_rails.size() == max_rails_) {
        NIXL_TRACE << "Reduandant call to selectSwitchRails() (iternal error)";
        return true;
    }

    // objective: get the available rails for this switch, but don't exceed neither switch capacity
    // nor total number of required rails
    const std::vector<size_t> &switch_rails = switch_rail_data.rail_ids_;
    size_t switch_rail_count = switch_rails.size();
    size_t missing_rail_count = max_rails_ - selected_rails.size();

    // limit number of rails
    assert(switch_rail_data.max_rails_ <= switch_rail_count);
    size_t max_rails =
        std::min({switch_rail_data.max_rails_, missing_rail_count, switch_rail_count});

    // get start index of selected rails
    size_t base_index =
        switch_rail_data.next_rail_index_.value_.fetch_add(max_rails, std::memory_order_relaxed) %
        switch_rail_count;

    // select at most max_rails from the rails found on the same NUMA node
    for (size_t i = 0; i < max_rails; ++i) {
        // get next rail
        selected_rails.push_back(switch_rails[base_index]);

        // wrap around if needed
        if (++base_index == switch_rail_count) {
            base_index = 0;
        }
        if (selected_rails.size() == max_rails_) {
            // reached configured limit, so stop
            break;
        }
    }

    return selected_rails.size() == max_rails_;
}

bool
nixlLibfabricNumaRailSelectionPolicy::selectExtraNodeRails(size_t numa_node_id,
                                                           std::vector<size_t> &selected_rails) {
    // objectice: select rails from the given node, until reaching node limit (all rails selected),
    // or reaching required number of rails to select
    size_t missing_rail_count = max_rails_ - selected_rails.size();
    std::vector<SwitchRailData> &switch_array = numa_data_rails_[numa_node_id].switch_data_;
    size_t switch_index = 0;
    for (size_t i = 0; i < missing_rail_count; ++i) {
        if (selectExtraSwitchRail(switch_array[switch_index], selected_rails)) {
            break;
        }
        if (++switch_index == switch_array.size()) {
            switch_index = 0;
        }
    }
    return selected_rails.size() == max_rails_;
}

bool
nixlLibfabricNumaRailSelectionPolicy::selectExtraSwitchRail(SwitchRailData &switch_rail_data,
                                                            std::vector<size_t> &selected_rails) {
    // objective: select a single rail from the given switch, which was not already selected
    const std::vector<size_t> &switch_rails = switch_rail_data.rail_ids_;
    size_t switch_rail_count = switch_rails.size();

    // select one rail that was not already selected
    for (size_t i = 0; i < switch_rail_count; ++i) {
        size_t rail_id = switch_rails[i];
        // if we really insist we can use bitset here for better performance
        if (std::find(selected_rails.begin(), selected_rails.end(), rail_id) ==
            selected_rails.end()) {
            selected_rails.push_back(rail_id);
            break;
        }
    }

    return selected_rails.size() == max_rails_;
}

bool
nixlLibfabricNumaRailSelectionPolicy::buildNumaDataRails(const nixlLibfabricTopology *topology) {
    int numa_node_count = -1;
    if (!LibfabricUtils::getNumConfiguredNumaNodes(numa_node_count)) {
        NIXL_ERROR << "Failed to build rail data for NUMA-aware rail selection policy, cannot get "
                      "number of NUMA nodes on local machine";
        return false;
    }
    numa_data_rails_.resize(numa_node_count);

    struct SwitchData {
        size_t index_; // in containing numa switch map entry
        size_t switch_link_speed_; // Gbps
        size_t total_nic_link_speed_; // Gbps (upstream link towards PCIe bridge/switch)
        std::vector<size_t> rail_ids_;

        SwitchData(size_t index = 0, size_t switch_link_speed = 0)
            : index_(index),
              switch_link_speed_(switch_link_speed),
              total_nic_link_speed_(0) {}
    };

    typedef std::
        unordered_map<std::pair<uint16_t, uint8_t>, SwitchData, pair_hash<uint16_t, uint8_t>>
            SwitchMap;
    std::vector<SwitchMap> numa_switch_map(numa_node_count);

    uint16_t max_numa_node_id = 0;
    const std::vector<std::string> &all_devices = topology->getAllDevices();
    for (size_t i = 0; i < all_devices.size(); ++i) {
        // get the NUMA node id of the device
        const std::string &device = all_devices[i];
        uint16_t dev_numa_node_id = topology->getDeviceNumaNode(device);
        if (dev_numa_node_id == nixlLibfabricTopology::INVALID_NUMA_NODE_ID) {
            NIXL_WARN << "Failed to get NUMA node id for device " << device << ", skipping";
            continue;
        }
        if (dev_numa_node_id >= numa_data_rails_.size()) {
            NIXL_WARN << "Invalid NUMA node id " << dev_numa_node_id
                      << " (out of range) for device " << device << ", skipping";
            continue;
        }

        // get parent switch bus id
        uint16_t numa_node_id = nixlLibfabricTopology::INVALID_NUMA_NODE_ID;
        size_t nic_link_speed = 0;
        uint16_t domain = UINT16_MAX;
        uint8_t bus_id = UINT8_MAX;
        size_t switch_link_speed = 0;
        if (!topology->getPcieDevData(
                device, numa_node_id, nic_link_speed, domain, bus_id, switch_link_speed)) {
            NIXL_WARN << "Could not find additional data for device " << device << ", skipping";
            continue;
        }

        if (numa_node_id >= numa_switch_map.size()) {
            NIXL_WARN << "Invalid NUMA node id " << numa_node_id << " for device " << device
                      << ", skipping";
            continue;
        }

        // get index for this switch group in the numa node
        SwitchMap &switch_map = numa_switch_map[numa_node_id];
        SwitchMap::iterator itr = switch_map.find({domain, bus_id});
        if (itr == switch_map.end()) {
            size_t index = switch_map.size();
            itr = switch_map
                      .insert(SwitchMap::value_type({domain, bus_id}, {index, switch_link_speed}))
                      .first;
        }
        // add rail and total speed
        itr->second.rail_ids_.push_back(i);
        itr->second.total_nic_link_speed_ += nic_link_speed;

        // compute actual max NUMA node id
        max_numa_node_id = std::max(max_numa_node_id, dev_numa_node_id);
    }

    // reduce array to actual size
    numa_switch_map.resize(max_numa_node_id + 1);

    // debug print switch/node/rail affinity
    NIXL_TRACE << "PCIe rail/switch affinity per NUMA node:";
    for (size_t numa_node_id = 0; numa_node_id < numa_switch_map.size(); ++numa_node_id) {
        NIXL_TRACE << "\tPCIe rail/switch affinity for NUMA node " << numa_node_id;
        const SwitchMap &switch_map = numa_switch_map[numa_node_id];
        for (const auto &entry : switch_map) {
            const SwitchData &switch_data = entry.second;
            NIXL_TRACE << "\t\tRails on PCIe switch on domain/bus-id " << entry.first.first << "/"
                       << entry.first.second << ": "
                       << LibfabricUtils::railIdsToString(switch_data.rail_ids_);
        }
    }

    // now build NUMA/switch rail map
    numa_data_rails_.resize(numa_switch_map.size());
    for (size_t numa_node_id = 0; numa_node_id < numa_switch_map.size(); ++numa_node_id) {
        const SwitchMap &switch_map = numa_switch_map[numa_node_id];
        numa_data_rails_[numa_node_id].rail_count_ = 0;
        numa_data_rails_[numa_node_id].switch_data_.resize(switch_map.size());
        for (const auto &entry : switch_map) {
            const SwitchData &switch_data = entry.second;
            size_t index = switch_data.index_;
            SwitchRailData &switch_rail_data = numa_data_rails_[numa_node_id].switch_data_[index];
            // NOTE: switch data cannot be empty, but we test anyway in case there is an internal
            // logic error
            if (switch_data.rail_ids_.size() == 0) {
                NIXL_ERROR << "Switch " << index << " on NUMA node " << numa_node_id
                           << " has no rails (internal error)";
                return false;
            }

            // compute how many NICs on this switch can fit
            // NOTE: average link speed cannot be zero, but we test anyway in case there is an
            // internal logic error
            size_t avg_nic_link_speed =
                switch_data.total_nic_link_speed_ / switch_data.rail_ids_.size();
            if (avg_nic_link_speed == 0) {
                NIXL_ERROR << "Average NIC upstream link speed on switch " << switch_data.index_
                           << " at NUMA node " << numa_node_id << " is zero (internal error)";
                return false;
            }
            switch_rail_data.max_rails_ = switch_data.switch_link_speed_ / avg_nic_link_speed;
            switch_rail_data.rail_ids_.insert(switch_rail_data.rail_ids_.end(),
                                              switch_data.rail_ids_.begin(),
                                              switch_data.rail_ids_.end());
            switch_rail_data.next_rail_index_.value_ = 0;

            numa_data_rails_[numa_node_id].rail_count_ += switch_data.rail_ids_.size();
        }
    }

    // verify that all NUMA nodes/switches have at least one assigned rail
    // NOTE: it is possible to have nodes/switches without NICs attached, but in that case we
    // shouldn't reach here, so this essentailly checks for internal error
    for (size_t i = 0; i < numa_data_rails_.size(); ++i) {
        if (numa_data_rails_[i].switch_data_.size() == 0) {
            NIXL_ERROR << "Failed to build data rail array for NUMA node " << i
                       << ", no PCIe switches assigned";
            return false;
        }
        for (size_t j = 0; j < numa_data_rails_[i].switch_data_.size(); ++j) {
            if (numa_data_rails_[i].switch_data_[j].rail_ids_.size() == 0) {
                NIXL_ERROR << "Failed to build data rail array for NUMA node " << i
                           << ", PCIe switch " << j << " has no data rails assigned";
                return false;
            }
            if (numa_data_rails_[i].switch_data_[j].max_rails_ == 0) {
                NIXL_ERROR << "Failed to build data rail array for NUMA node " << i
                           << ", PCIe switch " << j << " has no capacity for a single rail";
                return false;
            }
        }
    }
    return true;
}

void
nixlLibfabricNumaRailSelectionPolicy::buildNumaDistanceMap() {
    // build numa distance map (sorted from each origin node)
    size_t numa_node_count = numa_data_rails_.size();
    numa_distance_map_.resize(numa_node_count);
    for (size_t i = 0; i < numa_node_count; ++i) {
        numa_distance_map_[i].resize(numa_node_count);
        for (size_t j = 0; j < numa_node_count; ++j) {
            numa_distance_map_[i][j] = j;
        }
        std::sort(
            numa_distance_map_[i].begin(), numa_distance_map_[i].end(), [i](int node1, int node2) {
                return numa_distance(i, node1) < numa_distance(i, node2);
            });
    }
}
