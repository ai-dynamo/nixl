/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Serapheim Dimitropoulos, WekaIO Ltd.
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
#include <iostream>
#include <cufile.h>
#include <thread>
#include <memory>
#include "common/nixl_log.h"
#include "gds_mt_backend.h"
#include "common/str_tools.h"
#include "common/nixl_time.h"

namespace {
    // Use half the hardware threads by default for better I/O performance
    // Full hardware concurrency often leads to context switching overhead
    // and resource contention for storage I/O workloads
    const size_t DEFAULT_THREAD_COUNT = std::max(1u, std::thread::hardware_concurrency() / 2);
}

nixlGdsMtEngine::nixlGdsMtEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params)
{
    gds_mt_utils_ = std::make_unique<gdsMtUtil>();

    thread_count_ = DEFAULT_THREAD_COUNT;

    // Read custom parameters if available
    nixl_b_params_t* custom_params = init_params->customParams;
    if (custom_params) {
        // Configure thread_count
        if (custom_params->count("thread_count") > 0) {
            try {
                size_t tcount = std::stoul((*custom_params)["thread_count"]);
                if (tcount != 0) {
                    thread_count_ = tcount;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid thread_count parameter: " << e.what() << std::endl;
                this->initErr = true;
                return;
            }
        }
    }
    executor_ = std::make_unique<tf::Executor>(thread_count_);
    NIXL_DEBUG << "GDS_MIT: thread count=" << thread_count_;

    this->initErr = false;
    if (!gds_mt_utils_->isInitialized()) {
        this->initErr = true;
        return;
    }
}

nixl_status_t nixlGdsMtEngine::registerMem(const nixlBlobDesc &mem,
                                           const nixl_mem_t &nixl_mem,
                                           nixlBackendMD* &out)
{
    nixl_status_t status = NIXL_SUCCESS;
    auto md = std::make_unique<nixlGdsMtMetadata>();
    md->type = nixl_mem;
    cudaError_t error_id;

    switch (nixl_mem) {
        case FILE_SEG: {
            auto it = gds_mt_file_map_.find(mem.devId);
            if (it != gds_mt_file_map_.end()) {
                // Reuse existing registered handle
                md->handle = it->second;
                break;
            }

            // Create and register new handle
            md->handle = std::make_shared<gdsMtFileHandle>(*gds_mt_utils_, mem.devId, mem.len, mem.metaInfo);
            if (!md->handle->isRegistered()) {
                status = NIXL_ERR_BACKEND;
                break;
            }

            // Store in map for future reuse
            gds_mt_file_map_[mem.devId] = md->handle;
            break;
        }

        case VRAM_SEG: {
            error_id = cudaSetDevice(mem.devId);
            if (error_id != cudaSuccess) {
                NIXL_ERROR << "GDS_MT: error: cudaSetDevice returned "
                           << cudaGetErrorString(error_id) << " for device ID " << mem.devId;
                status = NIXL_ERR_BACKEND;
                break;
            }
            md->buf = std::make_unique<gdsMtMemBuf>(*gds_mt_utils_, (void *)mem.addr, mem.len, 0);
            if (!md->buf->isRegistered()) {
                status = NIXL_ERR_BACKEND;
            }
            break;
        }

        case DRAM_SEG: {
            md->buf = std::make_unique<gdsMtMemBuf>(*gds_mt_utils_, (void *)mem.addr, mem.len, 0);
            if (!md->buf->isRegistered()) {
                status = NIXL_ERR_BACKEND;
            }
            break;
        }

        default:
            status = NIXL_ERR_BACKEND;
            break;
    }

    if (status != NIXL_SUCCESS) {
        return status;
    }

    out = (nixlBackendMD*)md.release();
    return status;
}

nixl_status_t nixlGdsMtEngine::deregisterMem (nixlBackendMD* meta)
{
    std::unique_ptr<nixlGdsMtMetadata> md((nixlGdsMtMetadata*)meta);
    if (md->type == FILE_SEG && md->handle) {
        // Remove from map - shared_ptr will handle cleanup when last reference is gone
        gds_mt_file_map_.erase(md->handle->fd);
    }
    // No need to deregister buffer either, it is handled automatically by gdsMtMemBuf destructor
    return NIXL_SUCCESS;
}

void runCuFileOp(GdsMtTransferRequestH* req) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            perror("cuFileRead failed");
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            perror("cuFileWrite failed");
            return;
        }
    } else {
        return;
    }
    if ((size_t)nbytes != req->size) {
        NIXL_ERROR << "GDS_MT: error: short "
                   << ((req->op == CUFILE_READ) ? "read: " : "write: ")
                   << nbytes << " out of " << req->size << "bytes - address=" << req->addr;
        return;
    }
}

nixl_status_t nixlGdsMtEngine::prepXfer (const nixl_xfer_op_t &operation,
                                         const nixl_meta_dlist_t &local,
                                         const nixl_meta_dlist_t &remote,
                                         const std::string &remote_agent,
                                         nixlBackendReqH* &handle,
                                         const nixl_opt_b_args_t* opt_args) const
{
    auto gds_mt_handle = std::make_unique<nixlGdsMtBackendReqH>();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    // Basic validation
    if ((buf_cnt != file_cnt) ||
        ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "GDS_MT: error: incorrect count or operation selection";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        NIXL_ERROR << "GDS_MT: error: backend only supports I/O between memory (DRAM/VRAM_SEG) and files (FILE_SEG)";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Clear any existing requests before populating
    gds_mt_handle->request_list.clear();

    // Determine if local is the file segment
    bool is_local_file = (local.getType() == FILE_SEG);

    // Create list of all transfer requests
    for (size_t i = 0; i < buf_cnt; i++) {
        void* base_addr;
        size_t total_size;
        size_t base_offset;
        CUfileHandle_t cu_fhandle;

        // Get transfer parameters based on whether local is file or memory
        if (is_local_file) {
            base_addr = (void*)remote[i].addr;
            if (!base_addr) {
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = remote[i].len;
            base_offset = (size_t)local[i].addr;

            auto it = gds_mt_file_map_.find(local[i].devId);
            if (it == gds_mt_file_map_.end()) {
                NIXL_ERROR << "GDS_MT: error: file handle not found";
                return NIXL_ERR_NOT_FOUND;
            }
            cu_fhandle = it->second->cu_fhandle;
        } else {
            base_addr = (void*)local[i].addr;
            if (!base_addr) {
                return NIXL_ERR_INVALID_PARAM;
            }
            total_size = local[i].len;
            base_offset = (size_t)remote[i].addr;

            auto it = gds_mt_file_map_.find(remote[i].devId);
            if (it == gds_mt_file_map_.end()) {
                NIXL_ERROR << "GDS_MT: error: file handle not found";
                return NIXL_ERR_NOT_FOUND;
            }
            cu_fhandle = it->second->cu_fhandle;
        }

        gds_mt_handle->request_list.emplace_back(base_addr, total_size, base_offset, cu_fhandle,
            (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE);
    }

    if (gds_mt_handle->request_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    for (GdsMtTransferRequestH& req : gds_mt_handle->request_list) {
        GdsMtTransferRequestH* captured_req = &req;
        gds_mt_handle->taskflow.emplace([captured_req]() {
            runCuFileOp(captured_req);
        });
    }


    handle = gds_mt_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsMtEngine::postXfer(const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        nixlBackendReqH* &handle,
                                        const nixl_opt_b_args_t* opt_args) const
{
    nixlGdsMtBackendReqH* gds_mt_handle = (nixlGdsMtBackendReqH*)handle;
    if (!gds_mt_handle) {
        NIXL_ERROR << "GDS_MT: error: invalid handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (gds_mt_handle->request_list.empty()) {
        NIXL_ERROR << "GDS_MT: error: empty request list for Xfer";
        return NIXL_ERR_INVALID_PARAM;
    }
    gds_mt_handle->running_transfer = executor_->run(gds_mt_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t nixlGdsMtEngine::checkXfer(nixlBackendReqH* handle) const
{
    nixlGdsMtBackendReqH *gds_mt_handle = (nixlGdsMtBackendReqH *)handle;
    if (gds_mt_handle->request_list.empty()) {
        return NIXL_SUCCESS;
    }
    if (gds_mt_handle->running_transfer.wait_for(nixlTime::seconds(0)) != std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    gds_mt_handle->running_transfer.get();
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsMtEngine::releaseReqH(nixlBackendReqH* handle) const
{
    std::unique_ptr<nixlGdsMtBackendReqH> gds_mt_handle((nixlGdsMtBackendReqH*)handle);
    return NIXL_SUCCESS;
}

nixlGdsMtEngine::~nixlGdsMtEngine() {
    // Note: The destructor of the TaskFlow executor runs wait_for_all() to
    // wait for all submitted taskflows to complete and then notifies all worker
    // threads to stop and join these threads.
    // Note: The gds_mt_utils_ destructor automatically handles driver cleanup.
}
