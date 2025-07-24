/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nixl.h>
#include <nixl_types.h>
#include <backend/backend_engine.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <thread>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <exception>
#include <cstring>
#include <variant>
#include <future>
#include <atomic>
#include "common/nixl_log.h"
#include "gds_mt_backend.h"
#include "gds_mt_utils.h"
#include "file/file_utils.h"
#include <taskflow/taskflow.hpp>
#include <unordered_map>

namespace {
const size_t default_thread_count = std::max (1u, std::thread::hardware_concurrency() / 2);

struct FileSegData {
    std::shared_ptr<gdsMtFileHandle> handle;

    FileSegData(std::shared_ptr<gdsMtFileHandle> h) : handle(std::move(h)) {}
};

struct MemSegData {
    std::unique_ptr<gdsMtMemBuf> buf;
    MemSegData (void *addr, size_t size, int flags)
        : buf (std::make_unique<gdsMtMemBuf> (addr, size, flags)) {}
};

struct GdsMtTransferRequestH {
    GdsMtTransferRequestH (void *a,
                           size_t s,
                           size_t offset,
                           CUfileHandle_t handle,
                           CUfileOpcode_t operation)
        : addr{a},
          size{s},
          file_offset{offset},
          fh{handle},
          op{operation} {}

    void *addr;
    size_t size;
    size_t file_offset;
    CUfileHandle_t fh;
    CUfileOpcode_t op;
};

class nixlGdsMtMetadata : public nixlBackendMD {
public:
    explicit nixlGdsMtMetadata (std::shared_ptr<gdsMtFileHandle> file_handle)
        : nixlBackendMD (true),
          data_ (FileSegData{std::move (file_handle)}) {}

    explicit nixlGdsMtMetadata (void *addr, size_t size, int flags)
        : nixlBackendMD (true),
          data_ (MemSegData{addr, size, flags}) {}

    ~nixlGdsMtMetadata() = default;

    nixlGdsMtMetadata (const nixlGdsMtMetadata &) = delete;
    nixlGdsMtMetadata &
    operator= (const nixlGdsMtMetadata &) = delete;

    nixlGdsMtMetadata (nixlGdsMtMetadata &&) = default;
    nixlGdsMtMetadata &
    operator= (nixlGdsMtMetadata &&) = default;

    std::variant<FileSegData, MemSegData> data_;
};

class nixlGdsMtBackendReqH : public nixlBackendReqH {
public:
    ~nixlGdsMtBackendReqH();

    std::vector<GdsMtTransferRequestH> request_list;
    tf::Taskflow taskflow;
    std::future<void> running_transfer;
    std::atomic<nixl_status_t> overall_status;
};

size_t
getThreadCount (const nixlBackendInitParams *init_params) {
    size_t thread_count = default_thread_count;

    nixl_b_params_t *custom_params = init_params->customParams;
    if (custom_params) {
        if (custom_params->count ("thread_count") > 0) {
            try {
                size_t tcount = std::stoul ((*custom_params)["thread_count"]);
                if (tcount != 0) {
                    thread_count = tcount;
                }
            }
            catch (const std::exception &e) {
                throw std::runtime_error ("GDS_MT: Invalid thread_count parameter: " +
                                          std::string (e.what()));
            }
        }
    }
    return thread_count;
}

void
runCuFileOp (GdsMtTransferRequestH *req, std::atomic<nixl_status_t> *overall_status) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead (req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "GDS_MT: cuFileRead failed: " << strerror (errno);
            overall_status->store (NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite (req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            NIXL_ERROR << "GDS_MT: cuFileWrite failed: " << strerror (errno);
            overall_status->store (NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store (NIXL_ERR_INVALID_PARAM);
        return;
    }

    if ((size_t)nbytes != req->size) {
        NIXL_ERROR << "GDS_MT: error: short " << ((req->op == CUFILE_READ) ? "read: " : "write: ")
                   << nbytes << " out of " << req->size << " bytes - address=" << req->addr;
        overall_status->store (NIXL_ERR_BACKEND);
        return;
    }
}

nixl_status_t
extractTransferParams (const nixlMetaDesc &mem_desc,
                       const nixlMetaDesc &file_desc,
                       const std::unordered_map<int, std::weak_ptr<gdsMtFileHandle>> &file_map,
                       void *&base_addr,
                       size_t &total_size,
                       size_t &base_offset,
                       CUfileHandle_t &cu_fhandle) {
    base_addr = (void *)mem_desc.addr;
    total_size = mem_desc.len;
    base_offset = (size_t)file_desc.addr;

    auto it = file_map.find (file_desc.devId);
    if (it == file_map.end()) {
        NIXL_ERROR << "GDS_MT: error: file metadata not found";
        return NIXL_ERR_NOT_FOUND;
    }

    auto handle = it->second.lock();
    NIXL_ASSERT (handle);
    cu_fhandle = handle->cu_fhandle;
    return NIXL_SUCCESS;
}
} // namespace

nixlGdsMtBackendReqH::~nixlGdsMtBackendReqH() {
    if (running_transfer.valid()) {
        running_transfer.wait();
    }
}

nixlGdsMtEngine::nixlGdsMtEngine (const nixlBackendInitParams *init_params)
    : nixlBackendEngine (init_params),
      gds_mt_utils_(),
      thread_count_ (getThreadCount (init_params)),
      executor_ (std::make_unique<tf::Executor> (thread_count_)) {
    NIXL_DEBUG << "GDS_MIT: thread count=" << thread_count_;
}

nixl_status_t
nixlGdsMtEngine::registerMem (const nixlBlobDesc &mem,
                              const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) {
    switch (nixl_mem) {
    case FILE_SEG: {
        auto it = gds_mt_file_map_.find (mem.devId);
        std::shared_ptr<gdsMtFileHandle> handle;
        if (it != gds_mt_file_map_.end()) {
            handle = it->second.lock();
            if (handle) {
                out = new nixlGdsMtMetadata (handle);
                return NIXL_SUCCESS;
            }
            gds_mt_file_map_.erase (it);
        }

        try {
            handle = std::make_shared<gdsMtFileHandle> (mem.devId);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "GDS_MT: failed to create file handle: " << e.what();
            return NIXL_ERR_BACKEND;
        }
        gds_mt_file_map_[mem.devId] = handle;
        out = new nixlGdsMtMetadata (handle);
        return NIXL_SUCCESS;
    }

    case VRAM_SEG: {
        const cudaError_t error_id = cudaSetDevice (mem.devId);
        if (error_id != cudaSuccess) {
            NIXL_ERROR << "GDS_MT: error: cudaSetDevice returned " << cudaGetErrorString (error_id)
                       << " for device ID " << mem.devId;
            return NIXL_ERR_BACKEND;
        }
        [[fallthrough]];
    }

    case DRAM_SEG: {
        try {
            out = new nixlGdsMtMetadata ((void *)mem.addr, mem.len, 0);
            return NIXL_SUCCESS;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "GDS_MT: failed to create memory buffer: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    default:
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlGdsMtEngine::deregisterMem (nixlBackendMD *meta) {
    std::unique_ptr<nixlGdsMtMetadata> md ((nixlGdsMtMetadata *)meta);

    if (auto *file_data = std::get_if<FileSegData> (&md->data_)) {
        if (file_data->handle) {
            int key = file_data->handle->fd;
            md.reset(); // Release metadata first

            auto it = gds_mt_file_map_.find (key);
            if (it != gds_mt_file_map_.end() && it->second.expired()) {
                gds_mt_file_map_.erase (it);
            }

            // No need to close fd since we're not opening files
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsMtEngine::prepXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    auto gds_mt_handle = std::make_unique<nixlGdsMtBackendReqH>();
    size_t buf_cnt = local.descCount();
    size_t file_cnt = remote.descCount();

    if ((buf_cnt != file_cnt) || ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "GDS_MT: error: incorrect count or operation selection";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        NIXL_ERROR << "GDS_MT: error: backend only supports I/O between memory (DRAM/VRAM_SEG) and "
                      "files (FILE_SEG)";
        return NIXL_ERR_INVALID_PARAM;
    }

    gds_mt_handle->request_list.clear();
    bool is_local_file = (local.getType() == FILE_SEG);
    for (size_t i = 0; i < buf_cnt; i++) {
        void *base_addr;
        size_t total_size;
        size_t base_offset;
        CUfileHandle_t cu_fhandle;

        nixl_status_t param_status;
        if (is_local_file) {
            param_status = extractTransferParams (remote[i],
                                                  local[i],
                                                  gds_mt_file_map_,
                                                  base_addr,
                                                  total_size,
                                                  base_offset,
                                                  cu_fhandle);
        } else {
            param_status = extractTransferParams (local[i],
                                                  remote[i],
                                                  gds_mt_file_map_,
                                                  base_addr,
                                                  total_size,
                                                  base_offset,
                                                  cu_fhandle);
        }

        if (param_status != NIXL_SUCCESS) {
            return param_status;
        }

        gds_mt_handle->request_list.emplace_back (base_addr,
                                                  total_size,
                                                  base_offset,
                                                  cu_fhandle,
                                                  (operation == NIXL_READ) ? CUFILE_READ :
                                                                             CUFILE_WRITE);
    }

    if (gds_mt_handle->request_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    for (GdsMtTransferRequestH &req : gds_mt_handle->request_list) {
        GdsMtTransferRequestH *captured_req = &req;
        gds_mt_handle->taskflow.emplace (
            [captured_req, overall_status = &gds_mt_handle->overall_status]() {
                runCuFileOp (captured_req, overall_status);
            });
    }

    handle = gds_mt_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsMtEngine::postXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    nixlGdsMtBackendReqH *gds_mt_handle = (nixlGdsMtBackendReqH *)handle;

    gds_mt_handle->overall_status.store (NIXL_SUCCESS);
    gds_mt_handle->running_transfer = executor_->run (gds_mt_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsMtEngine::checkXfer (nixlBackendReqH *handle) const {
    nixlGdsMtBackendReqH *gds_mt_handle = (nixlGdsMtBackendReqH *)handle;
    if (gds_mt_handle->running_transfer.wait_for (nixlTime::seconds (0)) !=
        std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    gds_mt_handle->running_transfer.get();
    return gds_mt_handle->overall_status.load();
}

nixl_status_t
nixlGdsMtEngine::releaseReqH (nixlBackendReqH *handle) const {
    std::unique_ptr<nixlGdsMtBackendReqH> gds_mt_handle ((nixlGdsMtBackendReqH *)handle);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsMtEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}
