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

static constexpr int COMPLETION_BATCH_SIZE = 64;
static constexpr int DRAIN_MAX_ITERS = 100000;

// -----------------------------------------------------------------------------
// Backend Engine – Constructor / Destructor
// -----------------------------------------------------------------------------

nixlLibblkioEngine::nixlLibblkioEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      api_type_("io_uring"),
      num_queues_(1),
      queue_size_(128),
      direct_io_(false),
      io_polling_(false) {
    const nixl_b_params_t &params = getCustomParams();

    auto api_type_it = params.find("api_type");
    if (api_type_it != params.end()) {
        const std::string &val = api_type_it->second;
        if (val == "IO_URING") {
            api_type_ = "io_uring";
        } else if (val == "VHOST_USER") {
            api_type_ = "virtio-blk-vhost-user";
        } else if (val == "VDPA") {
            api_type_ = "virtio-blk-vhost-vdpa";
        } else {
            NIXL_ERROR << absl::StrFormat(
                "libblkio: unsupported api_type '%s' (valid: IO_URING, VHOST_USER, VDPA)", val);
            initErr = true;
            return;
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

    auto num_queues_it = params.find("num_queues");
    if (num_queues_it != params.end()) {
        try {
            num_queues_ = std::stoi(num_queues_it->second);
        }
        catch (...) {
            NIXL_WARN << absl::StrFormat("libblkio: invalid num_queues '%s', using default %d",
                                         num_queues_it->second,
                                         num_queues_);
        }
    }

    auto queue_size_it = params.find("queue_size");
    if (queue_size_it != params.end()) {
        try {
            queue_size_ = std::stoi(queue_size_it->second);
        }
        catch (...) {
            NIXL_WARN << absl::StrFormat("libblkio: invalid queue_size '%s', using default %d",
                                         queue_size_it->second,
                                         queue_size_);
        }
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
                        rawdevs_.push_back(entry);
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
        "libblkio: initialized with driver=%s, direct_io=%d, io_polling=%d, "
        "num_queues=%d, queue_size=%d",
        api_type_,
        direct_io_,
        io_polling_,
        num_queues_,
        queue_size_);
}

nixlLibblkioEngine::~nixlLibblkioEngine() {
    for (auto &device : devices_) {
        if (device.handle) {
            blkio_destroy(&device.handle);
        }
    }
    devices_.clear();
}

// -----------------------------------------------------------------------------
// Device Management
// -----------------------------------------------------------------------------

struct blkio *
nixlLibblkioEngine::getBlkioHandle(uint64_t devId) const {
    for (const auto &dev : devices_) {
        if (dev.devId == devId) {
            return dev.handle;
        }
    }
    return nullptr;
}

struct blkioq *
nixlLibblkioEngine::getBlkioQueue(uint64_t devId) const {
    for (const auto &dev : devices_) {
        if (dev.devId == devId) {
            return dev.queue;
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
        "libblkio: creating device path=%s, devId=0x%lx, driver=%s", path, devId, api_type_);

    BlkioDevice dev;
    dev.devId = devId;
    dev.path = path;

    int ret = blkio_create(api_type_.c_str(), &dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat(
            "libblkio: blkio_create(%s) failed: %s", api_type_, strerror(-ret));
        return NIXL_ERR_BACKEND;
    }

    // Set path (file/device for io_uring, socket for vhost-user, vdpa device for vdpa)
    ret = blkio_set_str(dev.handle, "path", path.c_str());
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: failed to set path: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    // Driver-specific pre-connect configuration
    if (api_type_ == "io_uring") {
        // io_uring: set num-entries (queue depth) and direct before connect.
        // num-entries is optional — not all device types support it (e.g.
        // loop devices), so a failure here is only a debug note.
        ret = blkio_set_int(dev.handle, "num-entries", queue_size_);
        if (ret < 0) {
            NIXL_DEBUG << absl::StrFormat(
                "libblkio: num-entries=%d not supported: %s (using default)",
                queue_size_,
                strerror(-ret));
        }

        if (direct_io_) {
            ret = blkio_set_bool(dev.handle, "direct", true);
            if (ret < 0) {
                NIXL_ERROR << "libblkio: failed to enable direct I/O";
                blkio_destroy(&dev.handle);
                return NIXL_ERR_BACKEND;
            }
        }

        if (io_polling_) {
            if (!direct_io_) {
                NIXL_WARN << "libblkio: io_polling requires direct_io for io_uring, skipping";
            } else {
                ret = blkio_set_bool(dev.handle, "poll", true);
                if (ret < 0) {
                    NIXL_WARN << "libblkio: failed to enable IOPOLL mode";
                }
            }
        }
    }
    // vhost-user and vdpa: no pre-connect properties beyond path

    // Connect to the block device
    ret = blkio_connect(dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: connect failed: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    // Post-connect, pre-start configuration: num-queues and queue-size
    if (num_queues_ > 1) {
        ret = blkio_set_int(dev.handle, "num-queues", num_queues_);
        if (ret < 0) {
            NIXL_WARN << absl::StrFormat(
                "libblkio: failed to set num-queues=%d: %s", num_queues_, strerror(-ret));
        }
    }

    // For virtio drivers, set queue-size between connect and start
    if (api_type_ != "io_uring") {
        ret = blkio_set_int(dev.handle, "queue-size", queue_size_);
        if (ret < 0) {
            NIXL_WARN << absl::StrFormat(
                "libblkio: failed to set queue-size=%d: %s", queue_size_, strerror(-ret));
        }

        if (io_polling_) {
            ret = blkio_set_int(dev.handle, "num-poll-queues", 1);
            if (ret < 0) {
                NIXL_WARN << "libblkio: failed to set num-poll-queues";
            }
        }
    }

    // Start the device
    ret = blkio_start(dev.handle);
    if (ret < 0) {
        NIXL_ERROR << absl::StrFormat("libblkio: start failed: %s", strerror(-ret));
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    // Query device properties after start
    uint64_t capacity = 0;
    ret = blkio_get_uint64(dev.handle, "capacity", &capacity);
    if (ret < 0) {
        NIXL_WARN << absl::StrFormat("libblkio: failed to get capacity: %s", strerror(-ret));
    }
    dev.capacity = capacity;

    // Query memory region requirements (consistent across all devices using same driver)
    bool nmr = false;
    ret = blkio_get_bool(dev.handle, "needs-mem-regions", &nmr);
    if (ret == 0) {
        needs_mem_regions_ = nmr;
    }

    bool nmrf = false;
    ret = blkio_get_bool(dev.handle, "needs-mem-region-fd", &nmrf);
    if (ret == 0) {
        needs_mem_region_fd_ = nmrf;
    }

    NIXL_INFO << absl::StrFormat(
        "libblkio: device ready, capacity=%lu, needs-mem-regions=%d, needs-mem-region-fd=%d",
        capacity,
        needs_mem_regions_,
        needs_mem_region_fd_);

    // Cache queue 0 for this device
    dev.queue = blkio_get_queue(dev.handle, 0);
    if (!dev.queue) {
        NIXL_ERROR << "libblkio: failed to get queue 0";
        blkio_destroy(&dev.handle);
        return NIXL_ERR_BACKEND;
    }

    devices_.push_back(dev);
    return NIXL_SUCCESS;
}

// -----------------------------------------------------------------------------
// Memory Registration
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Transfer Operations
// -----------------------------------------------------------------------------

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
            "libblkio: remote agent mismatch: expected '%s' or empty, got '%s'",
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

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat("libblkio: descriptor count mismatch: local=%d, remote=%d",
                                      local.descCount(),
                                      remote.descCount());
        return NIXL_ERR_INVALID_PARAM;
    }

    int count = remote.descCount();

    // Resolve blkio handles and queues for each descriptor
    std::vector<struct blkio *> handles;
    std::vector<struct blkioq *> queues;
    std::set<struct blkioq *> unique_queues;

    for (int i = 0; i < count; i++) {
        struct blkio *bh = getBlkioHandle(remote[i].devId);
        if (!bh) {
            NIXL_ERROR << absl::StrFormat(
                "libblkio: no handle for devId=0x%lx at index %d", remote[i].devId, i);
            return NIXL_ERR_NOT_FOUND;
        }
        struct blkioq *q = getBlkioQueue(remote[i].devId);
        if (!q) {
            NIXL_ERROR << absl::StrFormat(
                "libblkio: no queue for devId=0x%lx at index %d", remote[i].devId, i);
            return NIXL_ERR_NOT_FOUND;
        }
        handles.push_back(bh);
        queues.push_back(q);
        unique_queues.insert(q);
    }

    auto req = std::make_unique<nixlLibblkioBackendReqH>(operation, count);
    req->handles_ = std::move(handles);
    req->queues_ = std::move(queues);
    req->unique_queues_ = std::move(unique_queues);

    handle = req.release();
    return NIXL_SUCCESS;
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

    std::lock_guard<std::mutex> lock(queue_mutex_);

    for (int i = 0; i < req->desc_count_; i++) {
        const auto &local_desc = local[i];
        const auto &remote_desc = remote[i];
        struct blkio *bh = req->handles_[i];
        struct blkioq *q = req->queues_[i];

        void *io_buf = reinterpret_cast<void *>(local_desc.addr);
        size_t io_len = local_desc.len;
        uint64_t blk_offset = remote_desc.addr;

        if (needs_mem_region_fd_) {
            // Allocate fd-backed bounce buffer for vhost-user / vDPA
            nixlBlkioBounce bounce;
            bounce.handle = bh;
            bounce.user_addr = io_buf;
            bounce.size = io_len;
            bounce.needs_copy_back = (req->operation_ == NIXL_READ);

            int ret = blkio_alloc_mem_region(bh, &bounce.region, io_len);
            if (ret < 0) {
                NIXL_ERROR << absl::StrFormat(
                    "libblkio: blkio_alloc_mem_region(%zu) failed: %s", io_len, strerror(-ret));
                cleanupReqResources(req);
                return NIXL_ERR_BACKEND;
            }

            ret = blkio_map_mem_region(bh, &bounce.region);
            if (ret < 0) {
                NIXL_ERROR << absl::StrFormat("libblkio: blkio_map_mem_region (bounce) failed: %s",
                                              strerror(-ret));
                blkio_free_mem_region(bh, &bounce.region);
                cleanupReqResources(req);
                return NIXL_ERR_BACKEND;
            }

            // For writes, copy user data to bounce buffer before submission
            if (req->operation_ == NIXL_WRITE) {
                memcpy(bounce.region.addr, io_buf, io_len);
            }

            io_buf = bounce.region.addr;
            req->bounce_bufs_.push_back(bounce);

        } else if (needs_mem_regions_) {
            // Map user buffer directly (driver needs regions but not fd-backed)
            struct blkio_mem_region region = {};
            region.addr = io_buf;
            region.len = io_len;
            region.fd = -1;

            int ret = blkio_map_mem_region(bh, &region);
            if (ret < 0) {
                NIXL_ERROR << absl::StrFormat("libblkio: blkio_map_mem_region failed: %s",
                                              strerror(-ret));
                cleanupReqResources(req);
                return NIXL_ERR_BACKEND;
            }

            req->mapped_regions_.push_back({bh, region});
        }
        // io_uring: no memory region mapping needed

        // Enqueue the I/O request
        if (req->operation_ == NIXL_READ) {
            blkioq_read(q, blk_offset, io_buf, io_len, &req->io_slots_[i], 0);
        } else {
            blkioq_write(q, blk_offset, io_buf, io_len, &req->io_slots_[i], 0);
        }
    }

    // Submit all pending I/Os by flushing each unique queue
    for (auto *q : req->unique_queues_) {
        int ret = blkioq_do_io(q, nullptr, 0, 0, nullptr);
        if (ret < 0) {
            NIXL_WARN << absl::StrFormat("libblkio: do_io submit flush failed: %s", strerror(-ret));
        }
    }

    req->submitted_ = true;
    return NIXL_IN_PROG;
}

nixl_status_t
nixlLibblkioEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req = static_cast<nixlLibblkioBackendReqH *>(handle);

    // Fast path: already determined complete
    if (req->completed_) {
        return req->had_error_ ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(queue_mutex_);

    // Poll all queues that have pending I/Os
    pollQueues(req->unique_queues_);

    // Check if all this request's I/Os are done
    bool all_done = true;
    bool had_error = false;
    for (int i = 0; i < req->desc_count_; i++) {
        if (!req->io_slots_[i].completed) {
            all_done = false;
        } else if (req->io_slots_[i].ret != 0) {
            had_error = true;
        }
    }

    if (!all_done) {
        return NIXL_IN_PROG;
    }

    // All I/Os complete – copy back bounce buffers for read operations
    if (!had_error) {
        for (auto &bounce : req->bounce_bufs_) {
            if (bounce.needs_copy_back) {
                memcpy(bounce.user_addr, bounce.region.addr, bounce.size);
            }
        }
    }

    req->completed_ = true;
    req->had_error_ = had_error;
    return had_error ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

nixl_status_t
nixlLibblkioEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req = static_cast<nixlLibblkioBackendReqH *>(handle);

    if (req->submitted_ && !req->completed_) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        drainReqCompletions(req);

        // Copy back bounce buffers after drain (if not already done)
        if (!req->had_error_) {
            for (auto &bounce : req->bounce_bufs_) {
                if (bounce.needs_copy_back) {
                    memcpy(bounce.user_addr, bounce.region.addr, bounce.size);
                }
            }
        }
    }

    cleanupReqResources(req);
    delete req;
    return NIXL_SUCCESS;
}

// -----------------------------------------------------------------------------
// Helper Methods
// -----------------------------------------------------------------------------

void
nixlLibblkioEngine::pollQueues(const std::set<struct blkioq *> &queues) const {
    for (auto *q : queues) {
        struct blkio_completion comps[COMPLETION_BATCH_SIZE];
        int n = blkioq_do_io(q, comps, 0, COMPLETION_BATCH_SIZE, nullptr);
        if (n < 0) {
            NIXL_WARN << absl::StrFormat("libblkio: blkioq_do_io poll failed: %s", strerror(-n));
            continue;
        }
        for (int j = 0; j < n; j++) {
            auto *slot = static_cast<nixlBlkioIoSlot *>(comps[j].user_data);
            slot->completed = true;
            slot->ret = comps[j].ret;
        }
    }
}

void
nixlLibblkioEngine::drainReqCompletions(nixlLibblkioBackendReqH *req) const {
    for (int iter = 0; iter < DRAIN_MAX_ITERS; iter++) {
        bool all_done = true;
        for (int i = 0; i < req->desc_count_; i++) {
            if (!req->io_slots_[i].completed) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            // Check for errors
            for (int i = 0; i < req->desc_count_; i++) {
                if (req->io_slots_[i].ret != 0) {
                    req->had_error_ = true;
                    break;
                }
            }
            req->completed_ = true;
            return;
        }

        pollQueues(req->unique_queues_);
    }
    NIXL_WARN << "libblkio: drain timed out with incomplete I/Os";
    req->completed_ = true;
    req->had_error_ = true;
}

void
nixlLibblkioEngine::cleanupReqResources(nixlLibblkioBackendReqH *req) const {
    for (auto &[h, r] : req->mapped_regions_) {
        blkio_unmap_mem_region(h, &r);
    }
    req->mapped_regions_.clear();

    for (auto &bounce : req->bounce_bufs_) {
        blkio_unmap_mem_region(bounce.handle, &bounce.region);
        blkio_free_mem_region(bounce.handle, &bounce.region);
    }
    req->bounce_bufs_.clear();
}
