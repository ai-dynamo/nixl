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
#include <exception>
#include <memory>
#include <system_error>
#include <utility>
#include <variant>

#include <cuda_runtime.h>

#include "common/nixl_log.h"
#include "file/file_utils.h"
#include "gds_backend.h"

namespace {

struct fileSegData {
    std::shared_ptr<gdsFileHandle> handle;
    uint64_t dev_id;

    fileSegData(std::shared_ptr<gdsFileHandle> h, uint64_t id) : handle(std::move(h)), dev_id(id) {}
};

struct memSegData {
    gdsMemBuf buf;

    memSegData(void *addr, size_t size, int flags) : buf(addr, size, flags) {}
};

class nixlGdsMetadata : public nixlBackendMD {
public:
    nixlGdsMetadata(std::shared_ptr<gdsFileHandle> file_handle, uint64_t dev_id)
        : nixlBackendMD(true),
          data_(std::in_place_type<fileSegData>, std::move(file_handle), dev_id) {}

    explicit nixlGdsMetadata(void *addr, size_t size, int flags)
        : nixlBackendMD(true),
          data_(std::in_place_type<memSegData>, addr, size, flags) {}

    ~nixlGdsMetadata() override = default;

    nixlGdsMetadata(const nixlGdsMetadata &) = delete;
    nixlGdsMetadata &
    operator=(const nixlGdsMetadata &) = delete;

    std::variant<fileSegData, memSegData> data_;
};

} // namespace

nixlGdsEngine::nixlGdsEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    try {
        driver_ = std::make_unique<gdsDriverHandle>();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        this->initErr = true;
    }
}

nixl_status_t
nixlGdsEngine::registerMem(const nixlBlobDesc &mem,
                           const nixl_mem_t &nixl_mem,
                           nixlBackendMD *&out) {
    switch (nixl_mem) {
    case FILE_SEG: {
        auto reservation = path_mode_devids_.reserve(mem.devId, mem.metaInfo);
        if (!reservation.ok()) {
            NIXL_ERROR << "GDS: path-mode requires a unique devId per file (devId=" << mem.devId
                       << " already registered)";
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl::FileFd file_fd;
        try {
            file_fd = nixl::FileFd(mem.devId, mem.metaInfo);
        }
        catch (const std::system_error &e) {
            NIXL_ERROR << "GDS: path-mode open failed: " << e.what();
            return NIXL_ERR_BACKEND;
        }
        int fd = file_fd.fd();

        std::shared_ptr<gdsFileHandle> handle;
        if (auto it = gds_file_map_.find(fd); it != gds_file_map_.end()) {
            handle = it->second.lock();
            if (!handle) {
                gds_file_map_.erase(it);
            }
            // Cache hit: drop file_fd (~FileFd closes any duplicate owned fd).
        }
        if (!handle) {
            try {
                handle = std::make_shared<gdsFileHandle>(std::move(file_fd));
            }
            catch (const std::exception &e) {
                NIXL_ERROR << "GDS: failed to create file handle: " << e.what();
                return NIXL_ERR_BACKEND;
            }
            gds_file_map_[fd] = handle;
        }
        out = new nixlGdsMetadata(std::move(handle), mem.devId);
        reservation.commit();
        return NIXL_SUCCESS;
    }

    case VRAM_SEG: {
        const cudaError_t error_id = cudaSetDevice(mem.devId);
        if (error_id != cudaSuccess) {
            NIXL_ERROR << "GDS: error: cudaSetDevice returned " << cudaGetErrorString(error_id)
                       << " for device ID " << mem.devId;
            return NIXL_ERR_BACKEND;
        }
        [[fallthrough]];
    }

    case DRAM_SEG: {
        try {
            out = new nixlGdsMetadata((void *)mem.addr, mem.len, 0);
            return NIXL_SUCCESS;
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "GDS: failed to create memory buffer: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    default:
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlGdsEngine::deregisterMem(nixlBackendMD *meta) {
    std::unique_ptr<nixlGdsMetadata> md((nixlGdsMetadata *)meta);

    if (auto *file_data = std::get_if<fileSegData>(&md->data_)) {
        if (file_data->handle) {
            const int key = file_data->handle->file_fd.fd();
            const bool path_mode = !file_data->handle->file_fd.path().empty();
            const uint64_t dev_id = file_data->dev_id;
            md.reset(); // Release metadata first (drops this registration's ref).

            auto it = gds_file_map_.find(key);
            if (it != gds_file_map_.end() && it->second.expired()) {
                gds_file_map_.erase(it);
            }
            // owned fds: closed by ~FileFd (RAII) on last shared_ptr drop.
            if (path_mode) {
                path_mode_devids_.release(dev_id);
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsEngine::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    const size_t buf_cnt = local.descCount();
    const size_t file_cnt = remote.descCount();

    if ((buf_cnt != file_cnt) || ((operation != NIXL_READ) && (operation != NIXL_WRITE))) {
        NIXL_ERROR << "GDS: error: incorrect count or operation selection";
        return NIXL_ERR_INVALID_PARAM;
    }

    const bool is_local_file = (local.getType() == FILE_SEG);
    if (is_local_file == (remote.getType() == FILE_SEG)) {
        NIXL_ERROR << "GDS: backend only supports I/O between memory and files";
        return NIXL_ERR_INVALID_PARAM;
    }

    std::vector<gdsXferReq> reqs;
    reqs.reserve(buf_cnt);
    for (size_t i = 0; i < buf_cnt; i++) {
        const nixlMetaDesc &mem_desc = is_local_file ? remote[i] : local[i];
        const nixlMetaDesc &file_desc = is_local_file ? local[i] : remote[i];

        void *base_addr = (void *)mem_desc.addr;
        if (!base_addr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const auto *md = static_cast<const nixlGdsMetadata *>(file_desc.metadataP);
        if (!md) {
            NIXL_ERROR << "GDS: missing FILE_SEG metadata at xfer time";
            return NIXL_ERR_NOT_FOUND;
        }
        const auto *file_data = std::get_if<fileSegData>(&md->data_);
        if (!file_data || !file_data->handle) {
            NIXL_ERROR << "GDS: file metadata is not a FILE_SEG variant";
            return NIXL_ERR_NOT_FOUND;
        }

        reqs.push_back(gdsXferReq{base_addr,
                                  mem_desc.len,
                                  (size_t)file_desc.addr,
                                  file_data->handle->cu_fhandle,
                                  (operation == NIXL_READ) ? CUFILE_READ : CUFILE_WRITE});
    }

    if (reqs.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    return finalizePrep(std::move(reqs), handle);
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
