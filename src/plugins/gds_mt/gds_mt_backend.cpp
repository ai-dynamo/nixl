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
#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_mt_backend.h"
#include "gds_mt_utils.h"
#include "file/file_path_mode.h"
#include "file/file_utils.h"
#include <taskflow/taskflow.hpp>
#include <unordered_map>

nixl_status_t
gdsMtResolveRegisteredBuffer(void *registered_base,
                             size_t registered_size,
                             uintptr_t descriptor_addr,
                             size_t descriptor_size,
                             gdsMtResolvedBuffer &resolved) {
    const uintptr_t base = reinterpret_cast<uintptr_t> (registered_base);
    if (descriptor_addr < base || descriptor_size > registered_size) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t descriptor_delta = static_cast<size_t> (descriptor_addr - base);
    if (descriptor_delta > registered_size - descriptor_size) {
        return NIXL_ERR_INVALID_PARAM;
    }

    resolved.devPtrBase = registered_base;
    resolved.devPtrOffset = descriptor_delta;
    return NIXL_SUCCESS;
}

namespace {
const size_t default_thread_count = std::max (1u, std::thread::hardware_concurrency() / 2);

struct FileSegData {
    std::shared_ptr<gdsMtFileHandle> handle;
    uint64_t devId;

    FileSegData(std::shared_ptr<gdsMtFileHandle> h, uint64_t devid)
        : handle(std::move(h)),
          devId(devid) {}
};

struct MemSegData {
    std::unique_ptr<gdsMtMemBuf> buf;
    void *registeredBase;
    size_t registeredSize;

    MemSegData(void *addr, size_t size, int flags)
        : buf (std::make_unique<gdsMtMemBuf>(addr, size, flags)),
          registeredBase (addr),
          registeredSize (size) {}
};

struct GdsMtTransferRequestH {
    GdsMtTransferRequestH(void *base,
                           size_t buffer_offset,
                           size_t s,
                           size_t offset,
                           CUfileHandle_t handle,
                           CUfileOpcode_t operation)
        : devPtrBase{base},
          devPtrOffset{buffer_offset},
          size{s},
          file_offset{offset},
          fh{handle},
          op{operation} {}

    uintptr_t
    effectiveAddr() const {
        return reinterpret_cast<uintptr_t> (devPtrBase) + devPtrOffset;
    }

    void *devPtrBase;
    size_t devPtrOffset;
    size_t size;
    size_t file_offset;
    CUfileHandle_t fh;
    CUfileOpcode_t op;
};

class nixlGdsMtMetadata : public nixlBackendMD {
public:
    nixlGdsMtMetadata(std::shared_ptr<gdsMtFileHandle> file_handle, uint64_t devid)
        : nixlBackendMD(true),
          data_(FileSegData{std::move(file_handle), devid}) {}

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

[[nodiscard]] size_t
getThreadCount(const nixlBackendInitParams *init_params) {
    nixl_b_params_t *params = init_params->customParams;
    const size_t count =
        nixl::getBackendParamDefaulted(params, "thread_count", default_thread_count);
    return (count > 0) ? count : default_thread_count;
}

void
runCuFileOp (GdsMtTransferRequestH *req, std::atomic<nixl_status_t> *overall_status) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead(req->fh,
                             req->devPtrBase,
                             req->size,
                             req->file_offset,
                             req->devPtrOffset);
        if (nbytes < 0) {
            NIXL_ERROR << "GDS_MT: cuFileRead failed: " << strerror (errno);
            overall_status->store (NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite(req->fh,
                              req->devPtrBase,
                              req->size,
                              req->file_offset,
                              req->devPtrOffset);
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
                   << nbytes << " out of " << req->size
                   << " bytes - devPtrBase=" << req->devPtrBase
                   << " devPtrOffset=" << req->devPtrOffset
                   << " effective_address=" << reinterpret_cast<void *> (req->effectiveAddr());
        overall_status->store (NIXL_ERR_BACKEND);
        return;
    }
}

nixl_status_t
extractTransferParams (const nixlMetaDesc &mem_desc,
                       const nixlMetaDesc &file_desc,
                       void *&dev_ptr_base,
                       size_t &dev_ptr_offset,
                       size_t &total_size,
                       size_t &file_offset,
                       CUfileHandle_t &cu_fhandle) {
    total_size = mem_desc.len;
    file_offset = (size_t)file_desc.addr;

    auto *metadata = static_cast<nixlGdsMtMetadata *> (mem_desc.metadataP);
    if (!metadata) {
        NIXL_ERROR << "GDS_MT: error: memory metadata not found";
        return NIXL_ERR_NOT_FOUND;
    }

    auto *mem_data = std::get_if<MemSegData>(&metadata->data_);
    if (!mem_data) {
        NIXL_ERROR << "GDS_MT: error: memory metadata is not a registered memory buffer";
        return NIXL_ERR_INVALID_PARAM;
    }

    gdsMtResolvedBuffer resolved{};
    nixl_status_t status = gdsMtResolveRegisteredBuffer(mem_data->registeredBase,
                                                         mem_data->registeredSize,
                                                         mem_desc.addr,
                                                         mem_desc.len,
                                                         resolved);
    if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "GDS_MT: error: transfer descriptor is outside registered memory"
                   << " descriptor_addr=" << reinterpret_cast<void *> (mem_desc.addr)
                   << " descriptor_size=" << mem_desc.len
                   << " registeredBase=" << mem_data->registeredBase
                   << " registeredSize=" << mem_data->registeredSize;
        return status;
    }

    dev_ptr_base = resolved.devPtrBase;
    dev_ptr_offset = resolved.devPtrOffset;

    const auto *md = static_cast<const nixlGdsMtMetadata *>(file_desc.metadataP);
    if (!md) {
        NIXL_ERROR << "GDS_MT: missing FILE_SEG metadata at xfer time";
        return NIXL_ERR_NOT_FOUND;
    }
    auto *file_data = std::get_if<FileSegData>(&md->data_);
    if (!file_data || !file_data->handle) {
        NIXL_ERROR << "GDS_MT: file metadata is not a FILE_SEG variant";
        return NIXL_ERR_NOT_FOUND;
    }
    cu_fhandle = file_data->handle->cu_fhandle;
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
        auto resv = path_mode_devids_.reserve(mem.devId, mem.metaInfo);
        if (!resv.ok()) {
            NIXL_ERROR << "GDS_MT: path-mode requires a unique devId per file (devId=" << mem.devId
                       << " already registered)";
            return NIXL_ERR_INVALID_PARAM;
        }
        nixl::FileFd file_fd;
        try {
            file_fd = nixl::FileFd(mem.devId, mem.metaInfo);
        }
        catch (const std::system_error &e) {
            NIXL_ERROR << "GDS_MT: path-mode open failed: " << e.what();
            return NIXL_ERR_BACKEND;
        }
        int fd = file_fd.fd();

        std::shared_ptr<gdsMtFileHandle> handle;
        if (auto it = gds_mt_file_map_.find(fd); it != gds_mt_file_map_.end()) {
            handle = it->second.lock();
            if (!handle) {
                gds_mt_file_map_.erase(it);
            }
            // Cache hit: drop file_fd (~FileFd closes any duplicate owned fd).
        }
        if (!handle) {
            try {
                handle = std::make_shared<gdsMtFileHandle>(std::move(file_fd));
            }
            catch (const std::exception &e) {
                NIXL_ERROR << "GDS_MT: failed to create file handle: " << e.what();
                return NIXL_ERR_BACKEND;
            }
            gds_mt_file_map_[fd] = handle;
        }
        out = new nixlGdsMtMetadata(std::move(handle), mem.devId);
        resv.commit();
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
    std::unique_ptr<nixlGdsMtMetadata> md((nixlGdsMtMetadata *)meta);

    if (auto *file_data = std::get_if<FileSegData> (&md->data_)) {
        if (file_data->handle) {
            int key = file_data->handle->file_fd.fd();
            if (!file_data->handle->file_fd.path().empty()) {
                path_mode_devids_.release(file_data->devId);
            }
            md.reset(); // Release metadata first

            auto it = gds_mt_file_map_.find (key);
            if (it != gds_mt_file_map_.end() && it->second.expired()) {
                gds_mt_file_map_.erase (it);
            }

            // owned fds: closed by ~FileFd (RAII) on last shared_ptr drop.
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
        void *dev_ptr_base;
        size_t dev_ptr_offset;
        size_t total_size;
        size_t file_offset;
        CUfileHandle_t cu_fhandle;

        nixl_status_t param_status;
        if (is_local_file) {
            param_status = extractTransferParams (remote[i],
                                                  local[i],
                                                  dev_ptr_base,
                                                  dev_ptr_offset,
                                                  total_size,
                                                  file_offset,
                                                  cu_fhandle);
        } else {
            param_status = extractTransferParams (local[i],
                                                  remote[i],
                                                  dev_ptr_base,
                                                  dev_ptr_offset,
                                                  total_size,
                                                  file_offset,
                                                  cu_fhandle);
        }

        if (param_status != NIXL_SUCCESS) {
            return param_status;
        }

        gds_mt_handle->request_list.emplace_back(dev_ptr_base,
                                                  dev_ptr_offset,
                                                  total_size,
                                                  file_offset,
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
