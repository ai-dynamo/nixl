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

#include "libfabric_rail.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "libfabric_common.h"

#include <cstring>
#include <stdexcept>
#include <stack>

// RequestPool Base Class Implementation

RequestPool::RequestPool(size_t pool_size, size_t rail_id)
    : rail_id_(rail_id),
      initial_pool_size_(pool_size) {
    initializeBasePool(pool_size);
}

void
RequestPool::initializeBasePool(size_t pool_size) {
    size_t current_size = requests_.size();
    requests_.resize(current_size + pool_size);

    for (size_t i = current_size; i < requests_.size(); ++i) {
        requests_[i].rail_id = rail_id_;
        requests_[i].pool_index = i; // Set the pool index for deque compatibility
        requests_[i].in_use = false;
        free_indices_.push(i);
    }

    NIXL_INFO << "InitializeBasePool - Rail " << rail_id_
              << " completed. Total requests: " << requests_.size()
              << " Free requests: " << free_indices_.size();
}

void
RequestPool::release(nixlLibfabricReq *req) const {
    if (!req) {
        NIXL_WARN << "ReleaseReq on Rail " << rail_id_ << " received null request";
        return;
    }

    std::lock_guard<std::mutex> lock(pool_mutex_);

    // GUARD: Check if already released
    if (!req->in_use) {
        NIXL_WARN << "Attempt to double-release request XFER_ID=" << req->xfer_id << " on rail "
                  << rail_id_ << " - ignoring to prevent corruption";
        return;
    }

    NIXL_TRACE << "ReleaseReq on Rail " << rail_id_ << " releasing request XFER_ID=" << req->xfer_id
               << " pool_index=" << req->pool_index;

    req->in_use = false;
    req->xfer_id = 0;
    req->chunk_offset = 0;
    req->chunk_size = 0;
    req->completion_callback = nullptr;
    memset(&req->ctx, 0, sizeof(fi_context));

    // Use pool_index instead of pointer arithmetic for deque compatibility
    size_t idx = req->pool_index;

    // Validate the index is within bounds
    if (idx >= requests_.size()) {
        NIXL_ERROR << "Release Req on Rail " << rail_id_ << " invalid pool index " << idx
                   << " for request release (pool size: " << requests_.size() << ")";
        return;
    }

    free_indices_.push(idx);
}

nixlLibfabricReq *
RequestPool::findByContext(void *context) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!context) {
        return nullptr;
    }

    // Since fi_context2 ctx is the first member of nixlLibfabricReq,
    // we can directly cast the context pointer to the request pointer
    nixlLibfabricReq *req = reinterpret_cast<nixlLibfabricReq *>(context);

    NIXL_TRACE << "From context the request xfer_id is : " << req->xfer_id;
    return req;
}

size_t
RequestPool::getActiveRequestCount() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return requests_.size() - free_indices_.size();
}

size_t
RequestPool::getPoolUtilization() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return ((requests_.size() - free_indices_.size()) * 100) / requests_.size();
}

nixlLibfabricReq *
RequestPool::allocateReq() {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (free_indices_.empty()) {
        size_t old_size = requests_.size();

        // Try to expand the pool using the derived class implementation
        nixl_status_t expand_status = expandPool();
        if (expand_status != NIXL_SUCCESS) {
            NIXL_ERROR << "AllocateReq on Rail " << rail_id_
                       << " failed to expand pool, status=" << expand_status;
            return nullptr;
        }

        // Check if expansion provided new requests
        if (free_indices_.empty()) {
            NIXL_ERROR << "AllocateReq on Rail " << rail_id_
                       << " pool still exhausted after expansion";
            return nullptr;
        }

        NIXL_INFO << "AllocateReq on Rail " << rail_id_ << " successfully expanded pool from "
                  << old_size << " to " << requests_.size() << " requests";
    }

    size_t idx = free_indices_.top();
    free_indices_.pop();

    nixlLibfabricReq *req = &requests_[idx];
    req->in_use = true;
    req->xfer_id = LibfabricUtils::getNextXferId();

    return req;
}

// ControlRequestPool Implementation

ControlRequestPool::ControlRequestPool(size_t pool_size, size_t rail_id)
    : RequestPool(pool_size, rail_id),
      domain_(nullptr),
      chunk_size_(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE * pool_size) {}

ControlRequestPool::~ControlRequestPool() {
    // Cleanup should have been called explicitly before domain destruction
    // This is just a safety check
    cleanup();
}

void
ControlRequestPool::cleanup() {
    if (buffer_chunks_.empty()) return; // Already cleaned up

    for (auto &chunk : buffer_chunks_) {
        if (chunk.mr) {
            fi_close(&chunk.mr->fid);
            chunk.mr = nullptr;
        }
        if (chunk.buffer) {
            free(chunk.buffer);
            chunk.buffer = nullptr;
        }
    }
    buffer_chunks_.clear();
}

nixl_status_t
ControlRequestPool::createBufferChunk(size_t chunk_size, BufferChunk &chunk) {
    // Allocate buffer memory
    chunk.buffer = malloc(chunk_size);
    if (!chunk.buffer) {
        NIXL_ERROR << "CreateBufferChunk on Rail " << rail_id_
                   << " failed to allocate buffer chunk of size " << chunk_size << " bytes";
        return NIXL_ERR_BACKEND;
    }

    chunk.size = chunk_size;

    // Register buffer chunk with libfabric
    int ret =
        fi_mr_reg(domain_, chunk.buffer, chunk_size, FI_SEND | FI_RECV, 0, 0, 0, &chunk.mr, NULL);
    if (ret) {
        NIXL_ERROR << "CreateBufferChunk on Rail " << rail_id_
                   << " fi_mr_reg failed for buffer chunk: " << fi_strerror(-ret)
                   << " buffer=" << chunk.buffer << " size=" << chunk_size;
        free(chunk.buffer);
        chunk.buffer = nullptr;
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "CreateBufferChunk on Rail " << rail_id_ << " successfully created buffer chunk:"
              << " buffer=" << chunk.buffer << " size=" << chunk.size << " mr=" << chunk.mr
              << " mr_key=" << fi_mr_key(chunk.mr);

    return NIXL_SUCCESS;
}

nixl_status_t
ControlRequestPool::initialize(struct fid_domain *domain) {

    // Store domain for future expansions
    domain_ = domain;

    // Create initial buffer chunk
    BufferChunk initial_chunk;
    nixl_status_t status = createBufferChunk(chunk_size_, initial_chunk);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "InitializeWithBuffers on Rail " << rail_id_
                   << " failed to create initial buffer chunk";
        return status;
    }

    buffer_chunks_.push_back(initial_chunk);

    // Pre-assign buffers to requests
    for (size_t i = 0; i < requests_.size(); ++i) {
        void *buffer_addr =
            static_cast<char *>(initial_chunk.buffer) + (i * NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);
        requests_[i].buffer = buffer_addr;
        requests_[i].mr = initial_chunk.mr;
        requests_[i].buffer_size = NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE;
        requests_[i].operation_type = nixlLibfabricReq::SEND; // Default for control
    }

    NIXL_INFO << "InitializeWithBuffers on Rail " << rail_id_ << " successfully initialized with "
              << buffer_chunks_.size() << " buffer chunks";

    return NIXL_SUCCESS;
}

nixl_status_t
ControlRequestPool::expandPool() {
    NIXL_INFO << "Expanding control request pool on rail " << rail_id_ << " from "
              << requests_.size() << " to " << (requests_.size() * 2) << " requests";

    size_t current_size = requests_.size();
    size_t expansion_size = initial_pool_size_; // Add same amount as initial size

    // Create new buffer chunk for the expansion
    BufferChunk new_chunk;
    nixl_status_t status = createBufferChunk(chunk_size_, new_chunk);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "ExpandPool on Rail " << rail_id_
                   << " failed to create buffer chunk for pool expansion";
        return status;
    }

    buffer_chunks_.push_back(new_chunk);

    // Expand the base pool (adds new requests to requests_ vector and free_indices_)
    initializeBasePool(expansion_size);

    // Assign buffers to new requests
    for (size_t i = current_size; i < requests_.size(); ++i) {
        size_t local_idx = i - current_size;
        void *buffer_addr = static_cast<char *>(new_chunk.buffer) +
            (local_idx * NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);

        // Validate buffer address is within chunk bounds
        size_t buffer_offset = local_idx * NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE;
        if (buffer_offset + NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE > new_chunk.size) {
            NIXL_ERROR << " Rail " << rail_id_ << " buffer assignment out of bounds for request["
                       << i << "]:"
                       << " local_idx=" << local_idx << " buffer_offset=" << buffer_offset
                       << " buffer_size=" << NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE
                       << " chunk_size=" << new_chunk.size;
            return NIXL_ERR_BACKEND;
        }

        requests_[i].buffer = buffer_addr;
        requests_[i].mr = new_chunk.mr;
        requests_[i].buffer_size = NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE;
        requests_[i].operation_type = nixlLibfabricReq::SEND;
    }

    NIXL_INFO << "Successfully expanded control request pool on rail " << rail_id_ << " to "
              << requests_.size() << " requests with " << buffer_chunks_.size() << " buffer chunks";

    return NIXL_SUCCESS;
}

nixlLibfabricReq *
ControlRequestPool::allocate(size_t needed_size) {
    // Validate size before attempting allocation
    if (needed_size > NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE) {
        NIXL_ERROR << "Control pool allocation failed on rail " << rail_id_ << " - requested size "
                   << needed_size << " exceeds buffer size "
                   << NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE;
        return nullptr;
    }

    // Use common allocation logic from base class
    nixlLibfabricReq *req = allocateReq();

    if (req) {
        // Always reset buffer_size to the actual message size needed
        // The buffer itself is always NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE, but we need
        // to set buffer_size to the actual message size for libfabric operations
        req->buffer_size = needed_size;

        NIXL_TRACE << "Allocate on Rail " << rail_id_
                   << " allocated control request XFER_ID=" << req->xfer_id
                   << " buffer_size=" << req->buffer_size;
    } else {
        NIXL_ERROR << "Allocate on Rail " << rail_id_ << " failed to allocate control request";
    }

    return req;
}

// DataRequestPool Implementation

DataRequestPool::DataRequestPool(size_t pool_size, size_t rail_id)
    : RequestPool(pool_size, rail_id) {}

nixl_status_t
DataRequestPool::initialize() {
    // Initialize data requests
    for (size_t i = 0; i < requests_.size(); ++i) {
        requests_[i].buffer = nullptr; // No buffers for data requests
        requests_[i].mr = nullptr;
        requests_[i].buffer_size = 0;
        requests_[i].operation_type = nixlLibfabricReq::WRITE; // Default for data
    }
    return NIXL_SUCCESS;
}

nixl_status_t
DataRequestPool::expandPool() {
    NIXL_INFO << "Expanding data request pool on rail " << rail_id_ << " from " << requests_.size()
              << " to " << (requests_.size() * 2) << " requests";

    size_t current_size = requests_.size();
    size_t expansion_size = initial_pool_size_; // Add same amount as initial size

    // Expand the base pool (adds new requests to requests_ vector and free_indices_)
    initializeBasePool(expansion_size);

    // Initialize new requests
    for (size_t i = current_size; i < requests_.size(); ++i) {
        requests_[i].buffer = nullptr; // No buffers for data requests
        requests_[i].mr = nullptr;
        requests_[i].buffer_size = 0;
        requests_[i].operation_type = nixlLibfabricReq::WRITE; // Default for data
    }

    NIXL_INFO << "Successfully expanded data request pool on rail " << rail_id_ << " to "
              << requests_.size() << " requests";

    return NIXL_SUCCESS;
}

nixlLibfabricReq *
DataRequestPool::allocate(nixlLibfabricReq::OpType op_type) {
    // Use common allocation logic from base class
    nixlLibfabricReq *req = allocateReq();
    if (req) {
        // Set the operation type specific to data requests
        req->operation_type = op_type;
    }
    return req;
}

// Rail Class Implementation

nixlLibfabricRail::nixlLibfabricRail(const std::string &device,
                                     const std::string &provider,
                                     uint16_t id)
    : rail_id(id),
      device_name(device),
      provider_name(provider),
      blocking_cq_sread_supported(true),
      control_request_pool_(NIXL_LIBFABRIC_CONTROL_REQUESTS_PER_RAIL, id),
      data_request_pool_(NIXL_LIBFABRIC_DATA_REQUESTS_PER_RAIL, id),
      provider_supports_hmem_(false) {
    // Initialize all pointers to nullptr
    info = nullptr;
    fabric = nullptr;
    domain = nullptr;
    endpoint = nullptr;
    cq = nullptr;
    av = nullptr;
    memset(ep_name, 0, sizeof(ep_name));

    // Initialize all Libfabric resources for this rail
    NIXL_TRACE << "Initializing rail " << rail_id << " with device: " << device_name
               << ", provider: " << provider;

    // Initialize hints for this rail
    struct fi_info *hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "fi_allocinfo failed for rail " << rail_id;
        throw std::runtime_error("Failed to allocate fi_info for rail " + std::to_string(rail_id));
    }
    hints->caps = 0;
    hints->caps = FI_MSG | FI_RMA | FI_HMEM; // Try with FI_HMEM first
    hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    // Configure memory registration mode based on provider capabilities
    if (provider == "tcp" || provider == "sockets") {
        // TCP provider doesn't support FI_MR_PROV_KEY or FI_MR_VIRT_ADDR, use basic mode
        hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ALLOCATED;
        hints->domain_attr->mr_key_size = 0; // Let provider decide
    } else {
        // EFA and other providers support advanced memory registration
        hints->domain_attr->mr_mode =
            FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
        hints->domain_attr->mr_key_size = 2;
    }
    hints->domain_attr->name = strdup(device_name.c_str());
    hints->domain_attr->threading = FI_THREAD_SAFE;
    try {
        // Get fabric info for this specific device - first try with FI_HMEM
        int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);

        // If no provider found with FI_HMEM, retry without it
        if (ret || !info) {
            NIXL_INFO << "No provider found with FI_HMEM capability for rail " << rail_id
                      << ", retrying without FI_HMEM";

            // Retry without FI_HMEM
            hints->caps = FI_MSG | FI_RMA;
            hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;

            ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
            if (ret) {
                NIXL_ERROR << "fi_getinfo failed for rail " << rail_id << ": " << fi_strerror(-ret);
                throw std::runtime_error("fi_getinfo failed for rail " + std::to_string(rail_id));
            }

            provider_supports_hmem_ = false;
            NIXL_INFO << "Using provider without FI_HMEM support for rail " << rail_id;
        } else {
            // Provider found with FI_HMEM
            provider_supports_hmem_ = true;
            NIXL_INFO << "Using provider with FI_HMEM support for rail " << rail_id;
        }

        // Create fabric for this rail
        ret = fi_fabric(info->fabric_attr, &fabric, NULL);
        if (ret) {
            NIXL_ERROR << "fi_fabric failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_fabric failed for rail " + std::to_string(rail_id));
        }
        NIXL_INFO << "fabric_attr->name " << info->fabric_attr->name;
        // Create domain for this rail
        ret = fi_domain(fabric, info, &domain, NULL);
        if (ret) {
            NIXL_ERROR << "fi_domain failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_domain failed for rail " + std::to_string(rail_id));
        }

        // Create CQ for this rail
        struct fi_cq_attr cq_attr = {};
        cq_attr.format = FI_CQ_FORMAT_DATA;
        cq_attr.wait_obj = FI_WAIT_UNSPEC;
        cq_attr.size = 12288;
        ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
        if (ret) {
            NIXL_INFO << "fi_cq_open failed for rail " << rail_id << ": " << fi_strerror(-ret)
                      << " - trying FI_WAIT_NONE for " << info->fabric_attr->name << " provider";
            if (ret == -FI_ENOSYS) {
                NIXL_TRACE << "FI_WAIT_UNSPEC not supported, falling back to FI_WAIT_NONE for rail "
                           << rail_id;
                blocking_cq_sread_supported = false;
                // If fi_cq_open fails due to FI_WAIT_UNSPEC not supported, we fall back to
                // FI_WAIT_NONE and use fi_cq_read in control rails
                cq_attr.wait_obj = FI_WAIT_NONE;
                ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
                if (ret) {
                    NIXL_ERROR << "fi_cq_open with FI_WAIT_NONE failed for rail " << rail_id << ": "
                               << fi_strerror(-ret);
                    throw std::runtime_error("fi_cq_open with FI_WAIT_NONE failed for rail " +
                                             std::to_string(rail_id));
                }
                NIXL_TRACE << "fi_cq_open with FI_WAIT_NONE succeeded for rail " << rail_id;
            } else {
                throw std::runtime_error("fi_cq_open failed for rail " + std::to_string(rail_id));
            }
        }
        // Verify CQ was properly created
        if (!cq) {
            NIXL_ERROR << "CQ is null after fi_cq_open for rail " << rail_id;
            throw std::runtime_error("CQ creation returned success but pointer is null for rail " +
                                     std::to_string(rail_id));
        }
        // Create AV for this rail
        struct fi_av_attr av_attr = {};
        ret = fi_av_open(domain, &av_attr, &av, NULL);
        if (ret) {
            NIXL_ERROR << "fi_av_open failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_av_open failed for rail " + std::to_string(rail_id));
        }

        // Create endpoint for this rail
        ret = fi_endpoint(domain, info, &endpoint, NULL);
        if (ret) {
            NIXL_ERROR << "fi_endpoint failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_endpoint failed for rail " + std::to_string(rail_id));
        }

        // Bind endpoint with CQ and AV for this rail
        ret = fi_ep_bind(endpoint, &cq->fid, FI_TRANSMIT | FI_RECV);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind cq failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_ep_bind cq failed for rail " + std::to_string(rail_id));
        }

        ret = fi_ep_bind(endpoint, &av->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind av failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_ep_bind av failed for rail " + std::to_string(rail_id));
        }

        // Disable shared memory transfers for EFA provider to fix same-agent transfers
        bool optval = false;
        ret = fi_setopt(&endpoint->fid,
                        FI_OPT_ENDPOINT,
                        FI_OPT_SHARED_MEMORY_PERMITTED,
                        &optval,
                        sizeof(optval));
        if (ret && ret != -FI_ENOSYS) {
            NIXL_WARN << "fi_setopt FI_OPT_SHARED_MEMORY_PERMITTED failed for rail " << rail_id
                      << ": " << fi_strerror(-ret) << " - continuing anyway";
        } else if (ret == 0) {
            NIXL_DEBUG << "Successfully disabled shared memory transfers for rail " << rail_id;
        }

        // Enable endpoint for this rail
        ret = fi_enable(endpoint);
        if (ret) {
            NIXL_ERROR << "fi_enable failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_enable failed for rail " + std::to_string(rail_id));
        }

        // Get endpoint name for this rail
        size_t ep_name_len = sizeof(ep_name);
        ret = fi_getname(&endpoint->fid, ep_name, &ep_name_len);
        if (ret) {
            NIXL_ERROR << "fi_getname failed for rail " << rail_id << ": " << fi_strerror(-ret);
            throw std::runtime_error("fi_getname failed for rail " + std::to_string(rail_id));
        }

        // Initialize control request pool with buffers
        nixl_status_t status = control_request_pool_.initialize(domain);
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to initialize control request pool for rail " +
                                     std::to_string(rail_id));
        }
        // Initialize data request pool
        status = data_request_pool_.initialize();
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to initialize data request pool for rail " +
                                     std::to_string(rail_id));
        }

        NIXL_TRACE << "Initialized request pools: " << NIXL_LIBFABRIC_CONTROL_REQUESTS_PER_RAIL
                   << " control requests, " << NIXL_LIBFABRIC_DATA_REQUESTS_PER_RAIL
                   << " data requests for rail " << rail_id;

        // Post initial receive using new resource management system
        nixlLibfabricReq *recv_req = allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);
        if (!recv_req) {
            NIXL_ERROR << "Failed to allocate request for initial receive on rail " << rail_id;
            throw std::runtime_error("Failed to allocate request for initial receive on rail " +
                                     std::to_string(rail_id));
        }
        status = postRecv(recv_req);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to post initial receive on rail " << rail_id;
            releaseRequest(recv_req);
            throw std::runtime_error("Failed to post initial receive on rail " +
                                     std::to_string(rail_id));
        }
        NIXL_TRACE << "Successfully initialized rail " << rail_id;
    }
    catch (...) {
        fi_freeinfo(hints);
        throw;
    }
    fi_freeinfo(hints);
}

nixlLibfabricRail::~nixlLibfabricRail() {
    cleanup();
}

bool
nixlLibfabricRail::isProperlyInitialized() const {
    return (cq != nullptr && endpoint != nullptr && domain != nullptr);
}

void
nixlLibfabricRail::cleanup() {
    NIXL_TRACE << "Starting cleanup for rail " << rail_id;

    // STEP 1: Close endpoint first to stop any new operations
    if (endpoint) {
        NIXL_TRACE << "Closing endpoint for rail " << rail_id;
        int ret = fi_close(&endpoint->fid);
        if (ret) {
            NIXL_WARN << "fi_close endpoint failed for rail " << rail_id << ": "
                      << fi_strerror(-ret);
        }
        endpoint = nullptr;
    }

    // STEP 2: Close CQ after endpoint
    if (cq) {
        NIXL_TRACE << "Closing completion queue for rail " << rail_id;
        int ret = fi_close(&cq->fid);
        if (ret) {
            NIXL_WARN << "fi_close cq failed for rail " << rail_id << ": " << fi_strerror(-ret);
        }
        cq = nullptr;
    }

    if (av) {
        NIXL_TRACE << "Closing address vector for rail " << rail_id;
        int ret = fi_close(&av->fid);
        if (ret) {
            NIXL_WARN << "fi_close av failed for rail " << rail_id << ": " << fi_strerror(-ret);
        }
        av = nullptr;
    }

    // STEP 3: Clean up request pools while domain is still valid
    // This ensures all memory registrations (MRs) are properly deregistered before domain closure
    NIXL_TRACE << "Cleaning up request pools for rail " << rail_id;
    control_request_pool_.cleanup();
    // STEP 4: Close domain AFTER all MRs, endpoint, CQ, AV are closed
    if (domain) {
        NIXL_TRACE << "Closing domain for rail " << rail_id;
        int ret = fi_close(&domain->fid);
        if (ret) {
            NIXL_WARN << "fi_close domain failed for rail " << rail_id << ": " << fi_strerror(-ret);
        }
        domain = nullptr;
    }
    // STEP 5: Close fabric
    if (fabric) {
        NIXL_TRACE << "Closing fabric for rail " << rail_id;
        int ret = fi_close(&fabric->fid);
        if (ret) {
            NIXL_WARN << "fi_close fabric failed for rail " << rail_id << ": " << fi_strerror(-ret);
        }
        fabric = nullptr;
    }
    // STEP 6: Free info structure
    if (info) {
        NIXL_INFO << "Freeing info structure for rail " << rail_id;
        fi_freeinfo(info);
        info = nullptr;
    }
    NIXL_TRACE << "Cleanup completed for rail " << rail_id;
}

void
nixlLibfabricRail::setNotificationCallback(std::function<void(const std::string &)> callback) {
    notificationCallback = callback;
}

void
nixlLibfabricRail::setConnectionAckCallback(
    std::function<void(uint16_t, nixlLibfabricConnection *, ConnectionState)> callback) {
    connectionAckCallback = callback;
}

void
nixlLibfabricRail::setConnectionReqCallback(
    std::function<nixl_status_t(uint16_t, const std::string &, nixlLibfabricRail *)> callback) {
    connectionReqCallback = callback;
}

void
nixlLibfabricRail::setXferIdCallback(std::function<void(uint32_t)> callback) {
    xferIdCallback = callback;
}

// Per-Rail Completion Processing

// Per-rail completion processing - handles one rail's CQ with configurable blocking behavior
nixl_status_t
nixlLibfabricRail::progressCompletionQueue(bool use_blocking) const {
    // Completion processing
    struct fi_cq_data_entry completion;
    memset(&completion, 0, sizeof(completion));

    int ret;

    // Only protect libfabric CQ hardware operations
    {
        std::lock_guard<std::mutex> cq_lock(cq_progress_mutex_);

        if (use_blocking && blocking_cq_sread_supported) {
            // Blocking read using fi_cq_sread (used by CM thread)
            ret = fi_cq_sread(cq, &completion, 1, nullptr, NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_SEC);
        } else {
            // Non-blocking read (used by progress thread or fallback)
            ret = fi_cq_read(cq, &completion, 1);
        }

        if (ret < 0 && ret != -FI_EAGAIN) {
            NIXL_ERROR << "fi_cq_read returned error " << ret << " on rail " << rail_id << ": "
                       << fi_strerror(-ret);

            // Handle error - but be careful about fi_cq_readerr
            struct fi_cq_err_entry err_entry;
            memset(&err_entry, 0, sizeof(err_entry));

            int err_ret = fi_cq_readerr(cq, &err_entry, 0);
            if (err_ret > 0) {
                NIXL_ERROR << "CQ read failed on rail " << rail_id
                           << " with error: " << fi_strerror(err_entry.err)
                           << " prov_errno: " << err_entry.prov_errno << " len: " << err_entry.len;
            } else {
                NIXL_ERROR << "fi_cq_readerr failed with " << err_ret;
            }
            return NIXL_ERR_BACKEND;
        }
    }
    // CQ lock released here - completion is now local data

    if (ret == -FI_EAGAIN) {
        return NIXL_IN_PROG; // No completions available
    }

    if (ret == 1) {
        NIXL_TRACE << "Completion received on rail " << rail_id << " flags: " << std::hex
                   << completion.flags << " data: " << completion.data
                   << " context: " << completion.op_context << std::dec;

        // Process completion using local data. Callbacks have their own thread safety
        nixl_status_t status = processCompletionQueueEntry(&completion);
        if (status != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to process completion on rail " << rail_id;
            return status;
        }

        NIXL_DEBUG << "Completion processed on rail " << rail_id;
        return NIXL_SUCCESS;
    }

    return NIXL_ERR_BACKEND; // Unexpected case
}

// Route completion to appropriate handler (rail-specific)
nixl_status_t
nixlLibfabricRail::processCompletionQueueEntry(struct fi_cq_data_entry *comp) const {
    uint64_t flags = comp->flags;

    NIXL_TRACE << "Routing completion from rail " << rail_id << " with flags: " << std::hex << flags
               << " FI_SEND: " << (flags & FI_SEND) << " FI_RECV: " << (flags & FI_RECV)
               << " FI_WRITE: " << (flags & FI_WRITE)
               << " FI_REMOTE_WRITE: " << (flags & FI_REMOTE_WRITE) << std::dec;

    if (flags & FI_SEND) {
        // Local send completions (fi_senddata) - use context
        return processLocalSendCompletion(comp);

    } else if (flags & FI_RECV) {
        // Receive completions - use immediate data
        return processRecvCompletion(comp);

    } else if (flags & FI_WRITE) {
        // Local write completions (fi_writedata) - use context
        return processLocalTransferCompletion(comp, "write");

    } else if (flags & FI_READ) {
        // Local read completions (fi_readdata) - use context
        return processLocalTransferCompletion(comp, "read");

    } else if (flags & FI_REMOTE_WRITE) {
        // Remote write completions (from fi_writedata) - use immediate data
        return processRemoteWriteCompletion(comp);

    } else {
        // Add more detailed warning for unknown completion flags
        NIXL_WARN << "Unknown completion flags detected on rail " << rail_id << " - flags: 0x"
                  << std::hex << flags << std::dec << " (FI_SEND=" << !!(flags & FI_SEND)
                  << " FI_RECV=" << !!(flags & FI_RECV) << " FI_WRITE=" << !!(flags & FI_WRITE)
                  << " FI_READ=" << !!(flags & FI_READ)
                  << " FI_REMOTE_WRITE=" << !!(flags & FI_REMOTE_WRITE)
                  << " FI_REMOTE_READ=" << !!(flags & FI_REMOTE_READ) << ")"
                  << " data: 0x" << std::hex << comp->data << std::dec
                  << " context: " << comp->op_context << " len: " << comp->len;

        // Try to find the request associated with this context for debugging
        nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
        if (req) {
            NIXL_WARN << "Found request for zero-flags completion: XFER_ID=" << req->xfer_id
                      << " context=" << &req->ctx << " req_ptr=" << req << " rail=" << rail_id
                      << " in_use=" << req->in_use << " op_type="
                      << (req->operation_type == nixlLibfabricReq::WRITE    ? "WRITE" :
                              req->operation_type == nixlLibfabricReq::READ ? "READ" :
                              req->operation_type == nixlLibfabricReq::SEND ? "SEND" :
                                                                              "RECV");
        } else {
            NIXL_WARN << "No request found for zero-flags completion context " << comp->op_context;
        }

        // Check if this might be a spurious completion with flags=0
        if (flags == 0) {
            NIXL_WARN << "Completion with zero flags detected - this may be a spurious completion "
                         "or cleanup event";
            // Don't treat zero flags as a fatal error, just skip processing
            return NIXL_SUCCESS;
        }

        NIXL_ERROR << "Unknown completion flags: " << std::hex << flags << " data: " << comp->data
                   << " context: " << comp->op_context;
        return NIXL_ERR_BACKEND;
    }
}

// Handle local send completions (establishConnection, genNotif)
nixl_status_t
nixlLibfabricRail::processLocalSendCompletion(struct fi_cq_data_entry *comp) const {
    // Find the request from context to access the completion callback
    nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
    if (req && req->in_use) { // Only process if request is still valid and in use
        // Call completion callback if it exists
        if (req->completion_callback) {
            NIXL_TRACE << "Calling completion callback for send request " << req->xfer_id;
            req->completion_callback();
            NIXL_TRACE << "Completion callback completed for send";
        }
        releaseRequest(req);
    } else if (req && !req->in_use) {
        NIXL_WARN << "Completion received for already released send request " << req->xfer_id
                  << " on rail " << rail_id << " - skipping to prevent double-free";
    } else {
        NIXL_ERROR << "No request found for send completion context " << comp->op_context
                   << " on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// Handle local transfer completions (both read and write operations from postXfer)
nixl_status_t
nixlLibfabricRail::processLocalTransferCompletion(struct fi_cq_data_entry *comp,
                                                  const char *operation_type) const {
    // Find the request from context to access the completion callback
    nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
    if (req && req->in_use) { // Only process if request is still valid and in use
        // Call completion callback if it exists
        if (req->completion_callback) {
            NIXL_TRACE << "Calling completion callback for " << operation_type << " request "
                       << req->xfer_id;
            req->completion_callback();
            NIXL_TRACE << "Completion callback completed for " << operation_type;
        }
        releaseRequest(req);
    } else if (req && !req->in_use) {
        NIXL_WARN << "Completion received for already released " << operation_type << " request "
                  << req->xfer_id << " on rail " << rail_id << " - skipping to prevent double-free";
    } else {
        NIXL_ERROR << "No request found for " << operation_type << " completion context "
                   << comp->op_context << " on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// Handle remote receive completions (conn_req, conn_ack, notification messages)
nixl_status_t
nixlLibfabricRail::processRecvCompletion(struct fi_cq_data_entry *comp) const {
    // Get the request from context to access the received buffer
    nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
    if (!req) {
        NIXL_ERROR << "No request found for receive completion context on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }
    // Decode the immediate data format
    uint64_t msg_type = NIXL_GET_MSG_TYPE_FROM_IMM(comp->data);
    uint16_t agent_idx = NIXL_GET_AGENT_INDEX_FROM_IMM(comp->data);
    uint32_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(comp->data);
    NIXL_TRACE << "Received control message type " << msg_type << " agent_idx=" << agent_idx
               << " XFER_ID=" << xfer_id << " imm_data=0x" << std::hex << comp->data << std::dec;

    if (msg_type == NIXL_LIBFABRIC_MSG_CONNECT) {
        NIXL_TRACE << "Processing connection request on rail " << rail_id
                   << " Xfer_id :" << xfer_id;
        // Use callback to handle connection request processing
        if (connectionReqCallback) {
            std::string serialized_data(static_cast<char *>(req->buffer), req->buffer_size);
            nixl_status_t callback_status = connectionReqCallback(
                agent_idx, serialized_data, const_cast<nixlLibfabricRail *>(this));
            if (callback_status != NIXL_SUCCESS) {
                NIXL_ERROR << "Connection request callback failed";
                return callback_status;
            }
            NIXL_TRACE << "Connection request processed via callback for rail " << rail_id;
        } else {
            NIXL_ERROR << "No connection request callback set for rail " << rail_id;
            return NIXL_ERR_BACKEND;
        }
    } else if (msg_type == NIXL_LIBFABRIC_MSG_ACK) {
        NIXL_TRACE << "Processing connect request acknowledgement on rail " << rail_id;
        // Notify engine that connection is established via callback
        // TODO: validate the current state before calling callback
        if (connectionAckCallback) {
            connectionAckCallback(agent_idx, nullptr, ConnectionState::CONNECTED);
            NIXL_TRACE << "Connection state updated to CONNECTED via callback for rail " << rail_id;
        } else {
            NIXL_ERROR << "No connection state callback set for rail " << rail_id;
            return NIXL_ERR_BACKEND;
        }
    } else if (msg_type == NIXL_LIBFABRIC_MSG_NOTIFICTION) {
        NIXL_TRACE << "Processing notification request on rail " << rail_id
                   << " Xfer_id :" << xfer_id;

        // Create string from received buffer using the actual received length from completion entry
        std::string message(static_cast<char *>(req->buffer), comp->len);

        NIXL_TRACE << "Adding message: " << message << " to the notification list on rail "
                   << rail_id;

        // Call engine's callback to store notification in central storage (like reference)
        if (notificationCallback) {
            notificationCallback(message);
            NIXL_TRACE << "Notification stored via callback";
        } else {
            NIXL_ERROR << "No notification callback set!";
            return NIXL_ERR_BACKEND;
        }
    } else if (msg_type == NIXL_LIBFABRIC_MSG_DISCONNECT) {
        NIXL_TRACE << "Processing disconnect request from agent " << agent_idx << " on rail "
                   << rail_id
                   << "Currently not tracking the fi_addrs, so no callback for disconnect to clean "
                      "up libfabric AV list";
    } else {
        NIXL_ERROR << "Unknown message type: " << std::hex << msg_type << std::dec;
        return NIXL_ERR_BACKEND;
    }

    // Clear the receive buffer after processing
    memset(req->buffer, 0, req->buffer_size);

    releaseRequest(req);

    // Post a new receive using new resource management system
    nixlLibfabricReq *new_req = allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);
    if (!new_req) {
        NIXL_ERROR << "Failed to allocate request for subsequent receive on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }
    nixl_status_t status = postRecv(new_req);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to post subsequent receive on rail " << rail_id;
        releaseRequest(new_req);
        return status;
    }
    return NIXL_SUCCESS;
}

// Handle remote write completions (data arrival notification)
nixl_status_t
nixlLibfabricRail::processRemoteWriteCompletion(struct fi_cq_data_entry *comp) const {
    // Decode the immediate data format
    uint64_t msg_type = NIXL_GET_MSG_TYPE_FROM_IMM(comp->data);
    uint16_t agent_idx = NIXL_GET_AGENT_INDEX_FROM_IMM(comp->data);
    uint32_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(comp->data);

    // For remote write completions, we don't need to post a new receive
    // The write operation doesn't consume a receive buffer
    if (msg_type == NIXL_LIBFABRIC_MSG_TRANSFER) {
        NIXL_TRACE << "Remote write completion on rail " << rail_id << " - received " << comp->len
                   << " bytes" << " agent_idx=" << agent_idx << " XFER_ID=" << xfer_id
                   << " imm_data=0x" << std::hex << comp->data << std::dec;

        // Call XFER_ID tracking callback to add received XFER_ID to global set
        if (xferIdCallback) {
            xferIdCallback(comp->data);
            NIXL_TRACE << "Called XFER_ID callback for XFER_ID " << xfer_id;
        } else {
            NIXL_ERROR << "No XFER_ID callback set for rail " << rail_id;
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

// Per-Rail Libfabric Operation Wrappers

nixl_status_t
nixlLibfabricRail::postRecv(nixlLibfabricReq *req) const {
    if (!req || !req->buffer) {
        NIXL_ERROR << "Invalid request or buffer for receive on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    struct fi_msg msg = {};
    struct iovec msg_iov;
    void *desc = fi_mr_desc(req->mr); // Use request's MR

    // Setup message structure using request's buffer
    msg_iov.iov_base = req->buffer;
    msg_iov.iov_len = req->buffer_size;

    msg.msg_iov = &msg_iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = FI_ADDR_UNSPEC;
    msg.context = &req->ctx; // Use request's context directly
    msg.data = 0;

    NIXL_TRACE << "Posting receive on endpoint: " << endpoint << " buffer: " << req->buffer
               << " size: " << req->buffer_size << " context: " << &req->ctx;

    int ret = fi_recvmsg(endpoint, &msg, 0);
    if (ret) {
        NIXL_ERROR << "fi_recvmsg failed on rail " << rail_id << ": " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    NIXL_TRACE << "Receive posted successfully";
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRail::postSend(uint64_t immediate_data,
                            fi_addr_t dest_addr,
                            nixlLibfabricReq *req) const {
    if (req->buffer_size == 0 || req->buffer_size > NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE) {
        NIXL_ERROR << "Invalid message size: " << req->buffer_size
                   << " (max: " << NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE << ")";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Prepare descriptor
    void *desc = fi_mr_desc(req->mr);

    NIXL_TRACE << "Sending data on endpoint: " << endpoint << " buffer: " << req->buffer
               << " size: " << req->buffer_size << " immediate_data: " << std::hex << immediate_data
               << " msg_type: " << NIXL_GET_MSG_TYPE_FROM_IMM(immediate_data)
               << " agent_idx: " << NIXL_GET_AGENT_INDEX_FROM_IMM(immediate_data)
               << " XFER_ID: " << NIXL_GET_XFER_ID_FROM_IMM(immediate_data)
               << " dest_addr: " << dest_addr << std::dec << " context: " << &req->ctx;

    // Retry indefinitely until senddata succeeds or fails for all providers
    int ret = -FI_EAGAIN;
    int attempt = 0;

    while (true) {
        // Libfabric fi_senddata call
        ret = fi_senddata(
            endpoint, req->buffer, req->buffer_size, desc, immediate_data, dest_addr, &req->ctx);

        if (ret == 0) {
            // Success
            NIXL_TRACE << "Send posted successfully"
                       << (attempt > 0 ? " after " + std::to_string(attempt + 1) + " attempts" :
                                         "");
            return NIXL_SUCCESS;
        }

        if (ret == -FI_EAGAIN) {
            // Resource temporarily unavailable - retry indefinitely for all providers
            attempt++;

            // Log every N attempts to avoid log spam
            if (attempt % NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS == 0) {
                NIXL_INFO << "fi_senddata still retrying EAGAIN on rail " << rail_id << " after "
                          << attempt << " attempts";
            } else {
                NIXL_TRACE << "fi_senddata returned EAGAIN on rail " << rail_id
                           << ", retrying (attempt " << attempt << ")";
            }

            // Exponential backoff with cap to avoid overwhelming the system
            int delay_us = std::min(NIXL_LIBFABRIC_BASE_RETRY_DELAY_US * (1 + attempt / 10),
                                    NIXL_LIBFABRIC_MAX_RETRY_DELAY_US);

            // Progress completion queue to drain pending completions before retry
            nixl_status_t progress_status = progressCompletionQueue(false);
            if (progress_status == NIXL_SUCCESS) {
                NIXL_TRACE << "Progressed completions on rail " << rail_id << " before retry";
            }

            usleep(delay_us);
            continue;
        } else {
            // Other error - don't retry, fail immediately
            break;
        }
    }

    NIXL_ERROR << "fi_senddata failed on rail " << rail_id << ": " << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlLibfabricRail::postWrite(const void *local_buffer,
                             size_t length,
                             void *local_desc,
                             uint64_t immediate_data,
                             fi_addr_t dest_addr,
                             uint64_t remote_addr,
                             uint64_t remote_key,
                             nixlLibfabricReq *req) const {
    // Validation
    if (!req) {
        NIXL_ERROR << "Invalid request for write on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_TRACE << "Posting RDMA write on endpoint: " << std::hex << endpoint
               << " local_buffer: " << local_buffer << " length: " << length
               << " immediate_data: " << immediate_data << " dest_addr: " << dest_addr
               << " remote_addr: " << (void *)remote_addr << " remote_key: " << remote_key
               << " context: " << &req->ctx;

    // Retry indefinitely until writedata succeeds or fails for all providers
    int ret = -FI_EAGAIN;
    int attempt = 0;

    while (true) {
        // Libfabric fi_writedata call
        ret = fi_writedata(endpoint,
                           local_buffer,
                           length,
                           local_desc,
                           immediate_data,
                           dest_addr,
                           remote_addr,
                           remote_key,
                           &req->ctx);

        if (ret == 0) {
            // Success
            NIXL_TRACE << "RDMA write posted successfully"
                       << (attempt > 0 ? " after " + std::to_string(attempt + 1) + " attempts" :
                                         "");
            return NIXL_SUCCESS;
        }

        if (ret == -FI_EAGAIN) {
            // Resource temporarily unavailable - retry indefinitely for all providers
            attempt++;

            // Log every N attempts to avoid log spam
            if (attempt % NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS == 0) {
                NIXL_INFO << "fi_writedata still retrying EAGAIN on rail " << rail_id << " after "
                          << attempt << " attempts";
            } else {
                NIXL_TRACE << "fi_writedata returned EAGAIN on rail " << rail_id
                           << ", retrying (attempt " << attempt << ")";
            }

            // Exponential backoff with cap to avoid overwhelming the system
            int delay_us = std::min(NIXL_LIBFABRIC_BASE_RETRY_DELAY_US * (1 + attempt / 10),
                                    NIXL_LIBFABRIC_MAX_RETRY_DELAY_US);

            // Progress completion queue to drain pending completions before retry
            nixl_status_t progress_status = progressCompletionQueue(false);
            if (progress_status == NIXL_SUCCESS) {
                NIXL_TRACE << "Progressed completions on rail " << rail_id << " before retry";
            }

            usleep(delay_us);
            continue;
        } else {
            // Other error - don't retry, fail immediately
            break;
        }
    }

    NIXL_ERROR << "fi_writedata failed on rail " << rail_id << ": " << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlLibfabricRail::postRead(void *local_buffer,
                            size_t length,
                            void *local_desc,
                            fi_addr_t dest_addr,
                            uint64_t remote_addr,
                            uint64_t remote_key,
                            nixlLibfabricReq *req) const {
    // Validation
    if (!req) {
        NIXL_ERROR << "Invalid request for read on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    NIXL_TRACE << "Posting RDMA read on endpoint: " << std::hex << endpoint
               << " local_buffer: " << local_buffer << " length: " << length
               << " dest_addr: " << dest_addr << " remote_addr: " << (void *)remote_addr
               << " remote_key: " << remote_key << " context: " << &req->ctx;

    // Retry indefinitely until readdata succeeds or fails for all providers
    int ret = -FI_EAGAIN;
    int attempt = 0;

    while (true) {
        // Libfabric fi_read call
        ret = fi_read(endpoint,
                      local_buffer,
                      length,
                      local_desc,
                      dest_addr,
                      remote_addr,
                      remote_key,
                      &req->ctx);

        if (ret == 0) {
            // Success
            NIXL_TRACE << "RDMA read posted successfully"
                       << (attempt > 0 ? " after " + std::to_string(attempt + 1) + " attempts" :
                                         "");
            return NIXL_SUCCESS;
        }

        if (ret == -FI_EAGAIN) {
            // Resource temporarily unavailable - retry indefinitely for all providers
            attempt++;

            // Log every N attempts to avoid log spam
            if (attempt % NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS == 0) {
                NIXL_INFO << "fi_read still retrying EAGAIN on rail " << rail_id << " after "
                          << attempt << " attempts";
            } else {
                NIXL_TRACE << "fi_read returned EAGAIN on rail " << rail_id
                           << ", retrying (attempt " << attempt << ")";
            }

            // Exponential backoff with cap to avoid overwhelming the system
            int delay_us = std::min(NIXL_LIBFABRIC_BASE_RETRY_DELAY_US * (1 + attempt / 10),
                                    NIXL_LIBFABRIC_MAX_RETRY_DELAY_US);

            // Progress completion queue to drain pending completions before retry
            nixl_status_t progress_status = progressCompletionQueue(false);
            if (progress_status == NIXL_SUCCESS) {
                NIXL_TRACE << "Progressed completions on rail " << rail_id << " before retry";
            }

            usleep(delay_us);
            continue;
        } else {
            // Other error - don't retry, fail immediately
            break;
        }
    }

    NIXL_ERROR << "fi_read failed on rail " << rail_id << ": " << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
}

// Memory Registration Methods

nixl_status_t
nixlLibfabricRail::registerMemory(void *buffer,
                                  size_t length,
                                  nixl_mem_t mem_type,
                                  int gpu_id,
                                  struct fid_mr **mr_out,
                                  uint64_t *key_out) const {
    if (!buffer || !mr_out || !key_out) {
        NIXL_ERROR << "Invalid parameters on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!domain) {
        NIXL_ERROR << "Domain not initialized on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }

    // Determine access flags based on provider capabilities
    uint64_t provider_access_flags;
    if (provider_name == "tcp" || provider_name == "sockets") {
        // TCP provider has more limited memory registration capabilities
        // Use basic flags that are commonly supported
        provider_access_flags = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    } else {
        // EFA and other providers use standard remote access flags
        provider_access_flags = FI_REMOTE_WRITE | FI_REMOTE_READ;
    }

    struct fid_mr *mr;

    // For TCP providers, use a unique key to avoid conflicts
    // TCP provider assigns key 0 by default, but we need unique keys for multiple registrations
    uint64_t requested_key = 0;
    if (provider_name == "tcp" || provider_name == "sockets") {
        // Generate a unique key based on buffer address to avoid collisions
        // Use the lower bits of the buffer address as a simple unique identifier
        requested_key = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;

        NIXL_DEBUG << "TCP provider: using requested key " << requested_key << " for buffer "
                   << buffer << " on rail " << rail_id;
    }

    NIXL_TRACE << "Memory Registration: rail=" << rail_id << " provider=" << provider_name
               << " buffer=" << buffer << " length=" << length << " access_flags=0x" << std::hex
               << provider_access_flags << std::dec << " requested_key=" << requested_key;

    // Use fi_mr_regattr for enhanced memory registration control
    struct fi_mr_attr mr_attr = {};
    mr_attr.access = provider_access_flags;
    mr_attr.offset = 0;
    mr_attr.requested_key = requested_key;
    mr_attr.context = nullptr;
    mr_attr.auth_key_size = 0;
    mr_attr.auth_key = nullptr;

    // Set HMEM interface based on memory type and provider capability
    if (mem_type == VRAM_SEG) {
        if (provider_supports_hmem_) {
            mr_attr.iface = FI_HMEM_CUDA;
            mr_attr.device.cuda = gpu_id;
            NIXL_DEBUG << "CUDA memory registration - iface: FI_HMEM_CUDA, device.cuda: " << gpu_id;
        } else {
            NIXL_WARN << "VRAM memory requested but provider does not support FI_HMEM - falling "
                         "back to system memory registration";
            mr_attr.iface = FI_HMEM_SYSTEM;
        }
    } else {
        mr_attr.iface = FI_HMEM_SYSTEM;
        NIXL_DEBUG << "System memory registration - iface: FI_HMEM_SYSTEM";
    }

    struct iovec iov;
    iov.iov_base = buffer;
    iov.iov_len = length;
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;

    int ret = fi_mr_regattr(domain, &mr_attr, 0, &mr);
    if (ret) {
        NIXL_ERROR << "fi_mr_reg failed on rail " << rail_id << ": " << fi_strerror(-ret)
                   << " (buffer=" << buffer << ", length=" << length
                   << ", requested_key=" << requested_key << ")";
        return NIXL_ERR_BACKEND;
    }

    *mr_out = mr;
    *key_out = fi_mr_key(mr);

    NIXL_TRACE << "Memory Registration SUCCESS: rail=" << rail_id << " provider=" << provider_name
               << " buffer=" << buffer << " length=" << length << " mr=" << mr
               << " key=" << *key_out << " registered_range=[" << buffer << " - "
               << (void *)((char *)buffer + length) << "]";

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRail::deregisterMemory(struct fid_mr *mr) const {
    if (!mr) {
        NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }

    int ret = fi_close(&mr->fid);
    if (ret) {
        NIXL_ERROR << "fi_close failed on rail " << rail_id << ": " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// Address Vector Management Methods

nixl_status_t
nixlLibfabricRail::insertAddress(const void *addr, fi_addr_t *fi_addr_out) const {
    if (!addr || !fi_addr_out) {
        NIXL_ERROR << "Invalid parameters on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!av) {
        NIXL_ERROR << "Address vector not initialized on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }

    int ret = fi_av_insert(av, addr, 1, fi_addr_out, 0, NULL);
    if (ret != 1) {
        NIXL_ERROR << "fi_av_insert failed on rail " << rail_id << ": " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricRail::removeAddress(fi_addr_t fi_addr) const {
    if (fi_addr == FI_ADDR_UNSPEC) {
        NIXL_ERROR << "Invalid fi_addr parameter on rail " << rail_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!av) {
        NIXL_ERROR << "Address vector not initialized on rail " << rail_id;
        return NIXL_ERR_BACKEND;
    }

    int ret = fi_av_remove(av, &fi_addr, 1, 0);
    if (ret != 0) {
        NIXL_ERROR << "fi_av_remove failed on rail " << rail_id << ": " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// Memory Descriptor Helper Methods

void *
nixlLibfabricRail::getMemoryDescriptor(struct fid_mr *mr) const {
    if (!mr) {
        NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
        return nullptr;
    }
    return fi_mr_desc(mr);
}

uint64_t
nixlLibfabricRail::getMemoryKey(struct fid_mr *mr) const {
    if (!mr) {
        NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
        return 0;
    }
    return fi_mr_key(mr);
}

// Optimized Resource Management Methods

nixlLibfabricReq *
nixlLibfabricRail::allocateControlRequest(size_t needed_size) const {
    return const_cast<ControlRequestPool &>(control_request_pool_).allocate(needed_size);
}

nixlLibfabricReq *
nixlLibfabricRail::allocateDataRequest(nixlLibfabricReq::OpType op_type) const {
    return const_cast<DataRequestPool &>(data_request_pool_).allocate(op_type);
}

void
nixlLibfabricRail::releaseRequest(nixlLibfabricReq *req) const {
    if (!req) {
        NIXL_ERROR << "Null request provided to releaseRequest on rail " << rail_id;
        return;
    }

    // Determine which pool to release to based on operation type
    if (req->operation_type == nixlLibfabricReq::SEND ||
        req->operation_type == nixlLibfabricReq::RECV) {
        control_request_pool_.release(req);
    } else {
        data_request_pool_.release(req);
    }
}

nixlLibfabricReq *
nixlLibfabricRail::findRequestFromContext(void *context) const {
    if (!context) {
        NIXL_ERROR << "Null context provided to findRequestFromContext on rail " << rail_id;
        return nullptr;
    }
    // Try control pool first
    nixlLibfabricReq *req = control_request_pool_.findByContext(context);
    if (req) {
        return req;
    }
    // Try data pool
    req = data_request_pool_.findByContext(context);
    if (req) {
        return req;
    }
    NIXL_ERROR << "No request found for context " << context << " on rail " << rail_id;
    return nullptr;
}
