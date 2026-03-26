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

#include "libblkio_backend.h"
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include "common/nixl_log.h"
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <sstream>

// -----------------------------------------------------------------------------
// Request Handle Implementation
// -----------------------------------------------------------------------------

nixlLibblkioBackendReqH::nixlLibblkioBackendReqH(const nixl_xfer_op_t &operation,
                                                 const nixl_meta_dlist_t &local,
                                                 const nixl_meta_dlist_t &remote,
                                                 std::vector<struct blkio *> blkio_handles)
    : operation_(operation),
      local_(local),
      remote_(remote),
      blkio_handles_(std::move(blkio_handles)),
      status_(NIXL_SUCCESS) {}

nixl_status_t
nixlLibblkioBackendReqH::prepXfer() {
    if (blkio_handles_.empty()) {
        NIXL_ERROR << "libblkio: no blkio handles";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local_.descCount() != remote_.descCount()) {
        NIXL_ERROR << absl::StrFormat("libblkio: descriptor count mismatch - local: %d, remote: %d",
                                      local_.descCount(),
                                      remote_.descCount());
        return NIXL_ERR_INVALID_PARAM;
    }

    if (static_cast<int>(blkio_handles_.size()) != remote_.descCount()) {
        NIXL_ERROR << absl::StrFormat("libblkio: handle count (%zu) != descriptor count (%d)",
                                      blkio_handles_.size(),
                                      remote_.descCount());
        return NIXL_ERR_INVALID_PARAM;
    }

    for (size_t i = 0; i < blkio_handles_.size(); i++) {
        if (!blkio_handles_[i]) {
            NIXL_ERROR << absl::StrFormat("libblkio: null blkio handle at index %zu", i);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    status_ = NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibblkioBackendReqH::registerBlkioBuf(struct blkio *handle, void *buf, size_t size) {
    struct blkio_mem_region region;
    region.addr = buf;
    region.len = size;
    region.fd = -1;
    region.fd_offset = 0;

    int ret = blkio_map_mem_region(handle, &region);
    if (ret < 0) {
        NIXL_ERROR << "libblkio: blkio_map_mem_region failed";
        return NIXL_ERR_BACKEND;
    }

    registered_regions_.push_back({handle, region});
    return NIXL_SUCCESS;
}

void
nixlLibblkioBackendReqH::unregisterBlkioBufs() {
    for (auto &[handle, region] : registered_regions_)
        blkio_unmap_mem_region(handle, &region);
    registered_regions_.clear();
}

nixl_status_t
nixlLibblkioBackendReqH::postXfer() {
    for (int i = 0; i < local_.descCount(); i++) {
        const auto &local_desc = local_[i];
        const auto &remote_desc = remote_[i];
        struct blkio *handle = blkio_handles_[i];

        struct blkioq *queue = blkio_get_queue(handle, 0);
        if (!queue) {
            NIXL_ERROR << absl::StrFormat("libblkio: failed to get queue for desc %d", i);
            unregisterBlkioBufs();
            return NIXL_ERR_BACKEND;
        }

        struct blkio_completion comp;
        int ret;

        registerBlkioBuf(handle, (void *)(local_desc.addr), local_desc.len);

        if (operation_ == NIXL_READ) {
            blkioq_read(queue, remote_desc.addr, (void *)local_desc.addr, local_desc.len, &comp, 0);
        } else {
            blkioq_write(
                queue, remote_desc.addr, (void *)local_desc.addr, local_desc.len, &comp, 0);
        }

        ret = blkioq_do_io(queue, &comp, 1, 1, nullptr);
        if (ret < 0) {
            NIXL_ERROR << absl::StrFormat("libblkio: completion failed: %s", strerror(-ret));
            status_ = NIXL_ERR_BACKEND;
            unregisterBlkioBufs();
            return NIXL_ERR_BACKEND;
        }

        if (comp.ret < 0) {
            NIXL_ERROR << absl::StrFormat("libblkio: I/O error: %s", strerror(-comp.ret));
            status_ = NIXL_ERR_BACKEND;
            unregisterBlkioBufs();
            return NIXL_ERR_BACKEND;
        }
    }

    unregisterBlkioBufs();
    status_ = NIXL_SUCCESS;
    return status_;
}

nixl_status_t
nixlLibblkioBackendReqH::checkXfer() {
    return status_;
}

// -----------------------------------------------------------------------------
// Backend Engine Implementation
// -----------------------------------------------------------------------------

nixlLibblkioEngine::nixlLibblkioEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      api_type_("io_uring"),
      num_queues_(1),
      queue_size_(128),
      direct_io_(false),
      io_polling_(false),
      next_blk_reg_idx_(0) {
    const nixl_b_params_t &params = getCustomParams();

    auto api_type_it = params.find("api_type");
    if (api_type_it != params.end()) {
        const std::string &val = api_type_it->second;
        if (val == "IO_URING") {
            api_type_ = "io_uring";
        } else {
            NIXL_WARN << absl::StrFormat(
                "libblkio: unsupported api_type '%s', defaulting to io_uring", val);
            api_type_ = "io_uring";
        }
    }

    auto polling_it = params.find("io_polling");
    if (polling_it != params.end()) {
        io_polling_ = (polling_it->second == "true" || polling_it->second == "1");
    }

    auto direct_io_it = params.find("direct_io");
    if (direct_io_it != params.end()) {
        direct_io_ = (direct_io_it->second == "true" || direct_io_it->second == "1");
    }

    auto rawdev_it = params.find("device_list");
    if (rawdev_it != params.end()) {
        std::stringstream ss(rawdev_it->second);
        std::string entry;
        while (std::getline(ss, entry, ',')) {
            entry.erase(0, entry.find_first_not_of(" \t"));
            entry.erase(entry.find_last_not_of(" \t") + 1);
            if (!entry.empty()) {
                // Parse entry in format "id:type:path"
                std::stringstream entry_ss(entry);
                std::string id_str, type_str, path;
                if (std::getline(entry_ss, id_str, ':') && std::getline(entry_ss, type_str, ':') &&
                    std::getline(entry_ss, path)) {
                    try {
                        uint64_t dev_id = std::stoull(id_str);
                        rawdev_map_[dev_id] = path;
                        rawdevs_.push_back(entry); // Keep for backward compatibility
                    }
                    catch (const std::exception &e) {
                        NIXL_ERROR
                            << absl::StrFormat("libblkio: invalid device id '%s' in entry '%s'",
                                               id_str.c_str(),
                                               entry.c_str());
                    }
                } else {
                    NIXL_ERROR << absl::StrFormat(
                        "libblkio: invalid device entry format '%s', expected id:type:path",
                        entry.c_str());
                }
            }
        }
        NIXL_INFO << absl::StrFormat("libblkio: parsed %zu device(s) from device_list",
                                     rawdev_map_.size());
    }

    NIXL_INFO << absl::StrFormat(
        "libblkio: initialized with api_type=%s, direct_io=%d, io_polling=%d",
        api_type_,
        direct_io_,
        io_polling_);
}

nixlLibblkioEngine::~nixlLibblkioEngine() {
    for (auto &device : devices_) {
        if (device.handle) {
            blkio_destroy(&device.handle);
        }
    }
    devices_.clear();
}

struct blkio *
nixlLibblkioEngine::getBlkioHandle(uint64_t devId) const {
    for (const auto &dev : devices_) {
        if (dev.devId == devId) {
            return dev.handle;
        }
    }
    return nullptr;
}

nixl_status_t
nixlLibblkioEngine::createBlkioDevice(const std::string &path, uint64_t devId) {
    if (getBlkioHandle(devId)) {
        return NIXL_SUCCESS;
    }

    NIXL_INFO << absl::StrFormat(
        "libblkio: creating device path=%s, devId=0x%lx, api_type=%s", path, devId, api_type_);

    BlkioDevice dev;
    dev.devId = devId;
    dev.path = path;
    dev.handle = nullptr;

    int ret = blkio_create(api_type_.c_str(), &dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: blkio_create failed: %s", strerror(-ret));
        return NIXL_ERR_BACKEND;
    }

    ret = blkio_set_str(dev.handle, "path", path.c_str());
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: failed to set path: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    if (api_type_ == "io_uring") {
        if (direct_io_) {
            ret = blkio_set_bool(dev.handle, "direct", true);
            if (ret < 0) {
                NIXL_ERROR << "libblkio: failed to enable direct I/O";
                blkio_destroy(&dev.handle);
                return NIXL_ERR_BACKEND;
            }
        }
    } else {
        NIXL_ERROR << absl::StrFormat("libblkio: api_type '%s' is not supported", api_type_);
        blkio_destroy(&dev.handle);
        return NIXL_ERR_NOT_FOUND;
    }

    ret = blkio_connect(dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: connect failed: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    if (io_polling_) {
        ret = blkio_set_int(dev.handle, "io-polling", 1);
        if (ret < 0) {
            NIXL_WARN << "libblkio: failed to enable I/O polling";
        }
    }

    ret = blkio_start(dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: start failed: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    uint64_t capacity = 0;
    ret = blkio_get_uint64(dev.handle, "capacity", &capacity);
    if (ret < 0) {
        NIXL_WARN << absl::StrFormat("libblkio: failed to get capacity: %s", strerror(-ret));
        capacity = 0;
    }
    dev.capacity = capacity;

    devices_.push_back(dev);

    NIXL_INFO << absl::StrFormat("libblkio: device created successfully, capacity=%lu bytes",
                                 capacity);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibblkioEngine::registerMem(const nixlBlobDesc &mem,
                                const nixl_mem_t &nixl_mem,
                                nixlBackendMD *&out) {
    out = nullptr;

    if (nixl_mem != DRAM_SEG && nixl_mem != BLK_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    NIXL_DEBUG << absl::StrFormat("libblkio: registerMem devId=0x%lx, addr=0x%lx, len=0x%lx, "
                                  "type=%d, meta=%s",
                                  mem.devId,
                                  mem.addr,
                                  mem.len,
                                  nixl_mem,
                                  mem.metaInfo);

    if (nixl_mem == BLK_SEG) {
        std::string devpath;
        if (rawdev_map_.empty()) {
            // Try metaInfo fallback first
            if (!mem.metaInfo.empty()) {
                devpath = mem.metaInfo;
                NIXL_DEBUG << absl::StrFormat("libblkio: using metaInfo device path: %s", devpath);
            } else {
                // Try environment variable fallback
                const char *env_path = std::getenv("NIXL_LIBBLKIO_PATH");
                if (env_path && *env_path != '\0') {
                    devpath = env_path;
                    NIXL_DEBUG << absl::StrFormat("libblkio: using NIXL_LIBBLKIO_PATH: %s",
                                                  devpath);
                } else {
                    NIXL_ERROR << "libblkio: no devices in device_list, metaInfo empty, and "
                                  "NIXL_LIBBLKIO_PATH not set";
                    return NIXL_ERR_INVALID_PARAM;
                }
            }
        } else {
            // Use device_list resolution by devId
            auto it = rawdev_map_.find(mem.devId);
            if (it == rawdev_map_.end()) {
                NIXL_ERROR << absl::StrFormat("libblkio: device id 0x%lx not found in device_list",
                                              mem.devId);
                return NIXL_ERR_NOT_FOUND;
            }
            devpath = it->second;
            NIXL_DEBUG << absl::StrFormat("libblkio: devId=0x%lx -> %s", mem.devId, devpath);
        }

        nixl_status_t status = createBlkioDevice(devpath, mem.devId);
        if (status != NIXL_SUCCESS) {
            return status;
        }
    }

    out = new nixlLibblkioBackendMD(nixl_mem, mem.addr, mem.len, mem.devId);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibblkioEngine::deregisterMem(nixlBackendMD *meta) {
    delete meta;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlLibblkioEngine::prepXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    handle = nullptr;

    if (!remote_agent.empty() && remote_agent != localAgent) {
        NIXL_ERROR << absl::StrFormat(
            "libblkio: remote agent mismatch: expected %s or empty, got %s",
            localAgent,
            remote_agent);
        return NIXL_ERR_INVALID_PARAM;
    }

    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << absl::StrFormat("libblkio: local must be DRAM_SEG, got %d", local.getType());
        return NIXL_ERR_INVALID_PARAM;
    }

    if (remote.getType() != BLK_SEG) {
        NIXL_ERROR << absl::StrFormat("libblkio: remote must be BLK_SEG, got %d", remote.getType());
        return NIXL_ERR_INVALID_PARAM;
    }

    std::vector<struct blkio *> handles;
    for (int i = 0; i < remote.descCount(); i++) {
        struct blkio *h = getBlkioHandle(remote[i].devId);
        if (!h) {
            NIXL_ERROR << absl::StrFormat(
                "libblkio: no handle for devId=0x%lx at index %d", remote[i].devId, i);
            return NIXL_ERR_NOT_FOUND;
        }
        handles.push_back(h);
    }

    try {
        auto req =
            std::make_unique<nixlLibblkioBackendReqH>(operation, local, remote, std::move(handles));
        nixl_status_t status = req->prepXfer();
        if (status != NIXL_SUCCESS) {
            return status;
        }
        handle = req.release();
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << absl::StrFormat("libblkio: exception in prepXfer: %s", e.what());
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlLibblkioEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req = static_cast<nixlLibblkioBackendReqH *>(handle);
    return req->postXfer();
}

nixl_status_t
nixlLibblkioEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req = static_cast<nixlLibblkioBackendReqH *>(handle);
    return req->checkXfer();
}

nixl_status_t
nixlLibblkioEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    delete handle;
    return NIXL_SUCCESS;
}
