/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <cufile.h>
#include <absl/strings/str_format.h>
#include "gds_backend.h"
#include "gds_utils.h"
#include "common/nixl_log.h"
#include "file/file_path_mode.h"
#include "file/file_utils.h"
#include <unordered_map>
#include <memory>
#include <cuda_runtime.h>

/** Setting the default values to check the batch limit */
#define DEFAULT_BATCH_LIMIT 128
/** Setting the max request size to 16 MB */
#define DEFAULT_MAX_REQUEST_SIZE (16 * 1024 * 1024)  // 16MB
/** Create a batch pool of size 16 */
#define DEFAULT_BATCH_POOL_SIZE 16

nixlGdsEngine::nixlGdsEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
{
    gds_utils = new gdsUtil();

    // Set default values
    batch_pool_size = DEFAULT_BATCH_POOL_SIZE;
    batch_limit = DEFAULT_BATCH_LIMIT;
    max_request_size = DEFAULT_MAX_REQUEST_SIZE;

    // Read custom parameters if available
    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params) {
        // Configure batch_pool_size
        if (custom_params->count("batch_pool_size") > 0) {
            try {
                batch_pool_size = std::stoi((*custom_params)["batch_pool_size"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "Invalid batch_pool_size parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }

        // Configure batch_limit
        if (custom_params->count("batch_limit") > 0) {
            try {
                batch_limit = std::stoi((*custom_params)["batch_limit"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "Invalid batch_limit parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }

        // Configure max_request_size
        if (custom_params->count("max_request_size") > 0) {
            try {
                max_request_size = std::stoul((*custom_params)["max_request_size"]);
            } catch (const std::exception& e) {
                NIXL_ERROR << "Invalid max_request_size parameter: " << e.what();
                this->initErr = true;
                return;
            }
        }
    }

    this->initErr = false;
    if (gds_utils->openGdsDriver() == NIXL_ERR_BACKEND) {
        this->initErr = true;
        return;
    }

    // Initialize the batch pool
    for (unsigned int i = 0; i < batch_pool_size; i++) {
        batch_pool.push_back(new nixlGdsIOBatch(batch_limit));
    }

}

nixl_status_t nixlGdsEngine::registerMem(const nixlBlobDesc &mem,
                                         const nixl_mem_t &nixl_mem,
                                         nixlBackendMD* &out)
{
    switch (nixl_mem) {
        case FILE_SEG: {
            // Build the fdHandle (path-mode opens + owns, fd-mode keeps devId)
            // and look it up in the weak_ptr cache; on miss, construct a new
            // gdsFileHandle (RAII cuFileHandleRegister) from the fdHandle.
            nixl::fdHandle fdh;
            try {
                fdh = nixl::fdHandle(mem.metaInfo, static_cast<int>(mem.devId));
            } catch (const std::system_error &e) {
                NIXL_ERROR << "GDS path-mode open failed: " << e.what();
                return NIXL_ERR_BACKEND;
            }
            int fd = fdh.fd();
            std::shared_ptr<gdsFileHandle> handle;
            if (auto it = gds_file_map.find(fd); it != gds_file_map.end()) {
                handle = it->second.lock();
                // Cache hit: drop the just-built fdh (its dtor closes any
                // duplicate owned fd) and reuse the cached registration.
            }
            if (!handle) {
                try {
                    handle = std::make_shared<gdsFileHandle>(std::move(fdh));
                } catch (const std::exception &e) {
                    NIXL_ERROR << e.what();
                    return NIXL_ERR_BACKEND;
                }
                gds_file_map[fd] = handle;
            }
            out = new nixlGdsMetadata(std::move(handle));
            return NIXL_SUCCESS;
        }

        case VRAM_SEG: {
            const cudaError_t error_id = cudaSetDevice(mem.devId);
            if (error_id != cudaSuccess) {
                NIXL_ERROR << "cudaSetDevice returned " << cudaGetErrorString(error_id)
                          << " for device ID " << mem.devId;
                return NIXL_ERR_BACKEND;
            }
            [[fallthrough]];
        }

        case DRAM_SEG: {
            if (gds_utils->registerBufHandle((void *)mem.addr, mem.len, 0) != NIXL_SUCCESS) {
                return NIXL_ERR_BACKEND;
            }
            out = new nixlGdsMetadata(gdsMemBuf{(void *)mem.addr, mem.len}, nixl_mem);
            return NIXL_SUCCESS;
        }

        default:
            return NIXL_ERR_BACKEND;
    }
}

nixl_status_t nixlGdsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlGdsMetadata *md = (nixlGdsMetadata *)meta;
    if (md->type != FILE_SEG) {
        gds_utils->deregisterBufHandle(md->buf.base);
    }
    // FILE_SEG: dropping md releases the last shared_ptr<gdsFileHandle> if
    // this was the final user; ~gdsFileHandle then deregisters and ~fdHandle
    // closes the fd if owned. weak_ptr entries in gds_file_map become stale.
    delete md;
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    nixlGdsBackendReqH* gds_handle = new nixlGdsBackendReqH();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    // Basic validation
    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "Error in count or operation selection";
        delete gds_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        NIXL_ERROR << "Only support I/O between memory (DRAM/VRAM) and file type";
        delete gds_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Clear any existing requests before populating
    gds_handle->request_list.clear();

    // Determine if local is the file segment
    bool is_local_file = (local.getType() == FILE_SEG);

    // Resolve cuFileHandle from a FILE_SEG descriptor: prefer the shared_ptr
    // on metadataP (set by registerMem); fall back to a weak_ptr lookup keyed
    // on the resolved fd (handles cross-MD references in the same xfer).
    auto resolve_fh = [&](const nixlMetaDesc &d, CUfileHandle_t &out) -> nixl_status_t {
        if (d.metadataP) {
            auto *md = static_cast<const nixlGdsMetadata *>(d.metadataP);
            if (md->type == FILE_SEG && md->handle) {
                out = md->handle->cu_fhandle;
                return NIXL_SUCCESS;
            }
        }
        const auto it = gds_file_map.find(d.resolveFd());
        if (it == gds_file_map.end()) {
            NIXL_ERROR << "File handle not found";
            return NIXL_ERR_NOT_FOUND;
        }
        auto fh = it->second.lock();
        if (!fh) {
            NIXL_ERROR << "File handle has been released";
            return NIXL_ERR_NOT_FOUND;
        }
        out = fh->cu_fhandle;
        return NIXL_SUCCESS;
    };

    // Create list of all transfer requests
    for (size_t i = 0; i < buf_cnt; i++) {
        void* base_addr;
        size_t total_size;
        size_t base_offset;
        CUfileHandle_t fh = nullptr;

        // Get transfer parameters based on whether local is file or memory
        if (is_local_file) {
            base_addr = (void*)remote[i].addr;
            if (!base_addr) {
                delete gds_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = remote[i].len;
            base_offset = (size_t)local[i].addr;

            if (resolve_fh(local[i], fh) != NIXL_SUCCESS) {
                delete gds_handle;
                return NIXL_ERR_NOT_FOUND;
            }
        } else {
            base_addr = (void*)local[i].addr;
            if (!base_addr) {
                delete gds_handle;
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = local[i].len;
            base_offset = (size_t)remote[i].addr;

            if (resolve_fh(remote[i], fh) != NIXL_SUCCESS) {
                delete gds_handle;
                return NIXL_ERR_NOT_FOUND;
            }
        }

        // Split large transfers into multiple requests
        size_t remaining_size = total_size;
        size_t current_offset = 0;

        while (remaining_size > 0) {
            size_t request_size = std::min(remaining_size,
                                       (size_t)max_request_size);

            GdsTransferRequestH req;
            req.addr = (char*)base_addr + current_offset;
            req.size = request_size;
            req.file_offset = base_offset + current_offset;
            req.fh = fh;
            req.op = (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE;

            gds_handle->request_list.push_back(req);

            remaining_size -= request_size;
            current_offset += request_size;
        }
    }

    // Validate that we have requests before proceeding
    if (gds_handle->request_list.empty()) {
        delete gds_handle;
        return NIXL_ERR_INVALID_PARAM;
    }

    gds_handle->needs_prep = false;  // Just prepared, no need for prep
    handle = gds_handle;
    return NIXL_SUCCESS;
}

nixlGdsIOBatch* nixlGdsEngine::getBatchFromPool(unsigned int size) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock);
    // Use a pre-allocated batch if available
    if (!batch_pool.empty()) {
        nixlGdsIOBatch* batch = batch_pool.back();
        batch_pool.pop_back();
        batch->reset();
        return batch;
    }
    // Return nullptr if pool is empty - don't create new batches in the data path
    return nullptr;
}

void nixlGdsEngine::returnBatchToPool(nixlGdsIOBatch* batch) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock);
    // Only keep up to batch_pool_size batches
        batch_pool.push_back(batch);
}

nixl_status_t nixlGdsEngine::postXfer(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH* &handle,
                                      const nixl_opt_b_args_t* opt_args) const
{
    nixlGdsBackendReqH* gds_handle = (nixlGdsBackendReqH*)handle;

    // Validate request_list before proceeding
    if (gds_handle->request_list.empty()) {
        NIXL_ERROR << "Empty request list";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Process requests in batches
    const auto& request_list = gds_handle->request_list;
    size_t current_req = 0;

    while (current_req < request_list.size()) {
        size_t batch_size = std::min(request_list.size() - current_req,
                                     (size_t)batch_limit);
        nixl_status_t status = createAndSubmitBatch(request_list, current_req,
                                                    batch_size, gds_handle->batch_io_list);

        if (status != NIXL_SUCCESS) {
            // Clean up on error
            for (auto* batch : gds_handle->batch_io_list) {
                batch->cancelBatch();
                returnBatchToPool(batch);
            }
            gds_handle->batch_io_list.clear();
            return status;
        }
        current_req += batch_size;
    }

    return NIXL_IN_PROG;
}

nixl_status_t nixlGdsEngine::createAndSubmitBatch(const std::vector<GdsTransferRequestH>& requests,
                                                  size_t start_idx, size_t batch_size,
                                                  std::vector<nixlGdsIOBatch*>& batch_list) const
{
    nixlGdsIOBatch* batch = getBatchFromPool(batch_size);
    if (!batch) {
        NIXL_ERROR << "GDS batch pool exhausted";
        return NIXL_ERR_BACKEND;
    }

    // Add all requests to batch
    for (size_t i = 0; i < batch_size; i++) {
        const auto& req = requests[start_idx + i];
        if (!req.addr || !req.fh) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl_status_t status = batch->addToBatch(req.fh, req.addr, req.size,
                                               req.file_offset, 0, req.op);
        if (status != NIXL_SUCCESS) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixl_status_t status = batch->submitBatch(0);
    if (status != NIXL_SUCCESS) {
        returnBatchToPool(batch);
        return NIXL_ERR_BACKEND;
    }

    batch_list.push_back(batch);
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsEngine::checkXfer(nixlBackendReqH* handle) const
{
    nixlGdsBackendReqH *gds_handle = (nixlGdsBackendReqH *)handle;

    if (gds_handle->batch_io_list.empty()) {
        gds_handle->needs_prep = true;
        return NIXL_SUCCESS;
    }

    nixl_status_t status = NIXL_SUCCESS;
    for (auto* batch : gds_handle->batch_io_list) {
        status = batch->checkStatus();

        if (status == NIXL_IN_PROG) {
            return status;
        }

        if (status < 0) {
            batch->cancelBatch();
        }
        returnBatchToPool(batch);
    }

    gds_handle->batch_io_list.clear();
    gds_handle->needs_prep = true;
    return status;
}

nixl_status_t nixlGdsEngine::releaseReqH(nixlBackendReqH* handle) const
{

    nixlGdsBackendReqH *gds_handle = (nixlGdsBackendReqH *) handle;

    delete gds_handle;
    gds_handle = nullptr;

    return NIXL_SUCCESS;
}

nixlGdsEngine::~nixlGdsEngine() {
    // Clean up the batch pool
    for (auto* batch : batch_pool) {
        if (batch) {
            delete batch;
        }
    }
    batch_pool.clear();

    if (gds_utils) {
        gds_utils->closeGdsDriver();
        delete gds_utils;
    }
}

nixl_status_t
nixlGdsEngine::queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
