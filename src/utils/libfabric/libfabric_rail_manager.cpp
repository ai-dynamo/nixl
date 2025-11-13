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

#include "libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_topology.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"

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

nixlLibfabricRailManager::nixlLibfabricRailManager(size_t striping_threshold)
    : striping_threshold_(striping_threshold) {
    NIXL_DEBUG << "Creating rail manager with striping threshold: " << striping_threshold_
               << " bytes";

    // Initialize topology system
    topology = std::make_unique<nixlLibfabricTopology>();

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
    std::function<void()> completion_callback,
    BinaryNotification *binary_notif) {
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
        nixlLibfabricReq *req = data_rails_[rail_id]->allocateDataRequest(op_type);
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
        if (data_rails_[rail_id]->provider_name == "tcp" ||
            data_rails_[rail_id]->provider_name == "sockets") {
            req->remote_addr = 0; // Use offset 0 for TCP providers
            NIXL_DEBUG << "TCP provider detected: using offset 0 instead of virtual address "
                       << (void *)remote_base_addr << " for rail " << rail_id;
        } else {
            req->remote_addr = remote_base_addr; // Use virtual address for EFA and other providers
        }

        req->local_mr = local_mrs[rail_id];
        req->remote_key = remote_keys[remote_ep_id];
        req->rail_id = rail_id;
        // Submit immediately
        nixl_status_t status;
        if (op_type == nixlLibfabricReq::WRITE) {
            // Generate next SEQ_ID for this specific write operation
            uint8_t seq_id = LibfabricUtils::getNextSeqId();
            uint64_t imm_data = NIXL_MAKE_IMM_DATA(
                NIXL_LIBFABRIC_MSG_TRANSFER, agent_idx, binary_notif->xfer_id, seq_id);
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

        binary_notif->expected_completions++;

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
            nixlLibfabricReq *req = data_rails_[rail_id]->allocateDataRequest(op_type);
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
            if (data_rails_[rail_id]->provider_name == "tcp" ||
                data_rails_[rail_id]->provider_name == "sockets") {
                req->remote_addr = chunk_offset; // Use chunk offset for TCP providers
                NIXL_DEBUG << "TCP provider detected: using chunk offset " << chunk_offset
                           << " instead of virtual address "
                           << (void *)(remote_base_addr + chunk_offset) << " for rail " << rail_id;
            } else {
                req->remote_addr = remote_base_addr +
                    chunk_offset; // Use virtual address for EFA and other providers
            }

            req->local_mr = local_mrs[rail_id];
            req->remote_key = remote_keys[remote_ep_id];
            req->rail_id = rail_id;
            nixl_status_t status;
            if (op_type == nixlLibfabricReq::WRITE) {
                // Generate next SEQ_ID for this specific transfer operation
                uint8_t seq_id = LibfabricUtils::getNextSeqId();
                uint64_t imm_data = NIXL_MAKE_IMM_DATA(
                    NIXL_LIBFABRIC_MSG_TRANSFER, agent_idx, binary_notif->xfer_id, seq_id);
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

            binary_notif->expected_completions++;
        }
        NIXL_DEBUG << "Striping: submitted "
                   << (binary_notif ? binary_notif->expected_completions : 0) << " requests for "
                   << transfer_size << " bytes";
    }

    NIXL_DEBUG << "Successfully submitted "
               << (binary_notif ? binary_notif->expected_completions : 0) << " requests for "
               << transfer_size << " bytes";

    return NIXL_SUCCESS;
}

std::vector<size_t>
nixlLibfabricRailManager::selectRailsForMemory(void *mem_addr,
                                               nixl_mem_t mem_type,
                                               int gpu_id) const {
    if (mem_type == VRAM_SEG) {
#ifdef HAVE_CUDA
        if (gpu_id < 0) {
            NIXL_ERROR << "Invalid GPU ID " << gpu_id << " for VRAM memory " << mem_addr;
            return {}; // Return empty vector to indicate failure
        }
        std::vector<std::string> gpu_efa_devices = topology->getEfaDevicesForGpu(gpu_id);
        if (gpu_efa_devices.empty()) {
            NIXL_ERROR << "No EFA devices found for GPU " << gpu_id;
            return {}; // Return empty vector to indicate failure
        }
        std::vector<size_t> gpu_rails;
        for (const std::string &efa_device : gpu_efa_devices) {
            auto it = efa_device_to_rail_map.find(efa_device);
            if (it != efa_device_to_rail_map.end()) {
                // Bounds check: ensure rail index is valid
                if (it->second < data_rails_.size()) {
                    gpu_rails.push_back(it->second);
                    NIXL_DEBUG << "VRAM memory " << mem_addr << " on GPU " << gpu_id
                               << " mapped to rail " << it->second << " (EFA device=" << efa_device
                               << ")";
                } else {
                    NIXL_WARN << "EFA device " << efa_device << " maps to rail " << it->second
                              << " but only " << data_rails_.size() << " rails available";
                }
            } else {
                NIXL_WARN << "EFA device " << efa_device << " not found in rail mapping for GPU "
                          << gpu_id;
            }
        }

        if (gpu_rails.empty()) {
            NIXL_ERROR << "No valid rail mapping found for GPU " << gpu_id << " (checked "
                       << gpu_efa_devices.size() << " EFA devices)";
            return {};
        }

        NIXL_DEBUG << "VRAM memory " << mem_addr << " on GPU " << gpu_id << " will use "
                   << gpu_rails.size() << " rails total";
        return gpu_rails;
#else
        NIXL_ERROR << "VRAM memory type not supported without CUDA";
        return {};
#endif
    }
    if (mem_type == DRAM_SEG) {
        // For DRAM, use all available rails for maximum bandwidth
        std::vector<size_t> all_rails;
        all_rails.reserve(data_rails_.size());
        for (size_t i = 0; i < data_rails_.size(); ++i) {
            all_rails.push_back(i);
        }

        NIXL_DEBUG << "DRAM memory " << mem_addr << " will use all " << all_rails.size()
                   << " available rails for maximum bandwidth";
        return all_rails;
    }

    // For unsupported memory types, return empty vector
    NIXL_ERROR << "Unsupported memory type " << mem_type;
    return {};
}

nixl_status_t
nixlLibfabricRailManager::registerMemory(void *buffer,
                                         size_t length,
                                         nixl_mem_t mem_type,
                                         int gpu_id,
                                         std::vector<struct fid_mr *> &mr_list_out,
                                         std::vector<uint64_t> &key_list_out,
                                         std::vector<size_t> &selected_rails_out) {
    if (!buffer) {
        NIXL_ERROR << "Invalid buffer parameter";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Use internal rail selection with explicit GPU ID
    std::vector<size_t> selected_rails = selectRailsForMemory(buffer, mem_type, gpu_id);
    if (selected_rails.empty()) {
        NIXL_ERROR << "No rails selected for memory type " << mem_type;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Resize output vectors to match all rails
    mr_list_out.resize(data_rails_.size(), nullptr);
    key_list_out.resize(data_rails_.size(), 0);
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
        nixl_status_t status =
            data_rails_[rail_idx]->registerMemory(buffer, length, mem_type, gpu_id, &mr, &key);
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
    case ControlMessageType::CONNECTION_REQ:
        msg_type_value = NIXL_LIBFABRIC_MSG_CONNECT;
        break;
    case ControlMessageType::CONNECTION_ACK:
        msg_type_value = NIXL_LIBFABRIC_MSG_ACK;
        break;
    case ControlMessageType::DISCONNECT_REQ:
        msg_type_value = NIXL_LIBFABRIC_MSG_DISCONNECT;
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
        nixl_status_t status = data_rails_[rail_id]->progressCompletionQueue(false);
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
        nixl_status_t status =
            control_rails_[rail_id]->progressCompletionQueue(true); // Blocking for control rails
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
