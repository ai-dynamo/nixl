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

#ifndef NIXL_SRC_PLUGINS_LIBBLKIO_LIBBLKIO_BACKEND_H
#define NIXL_SRC_PLUGINS_LIBBLKIO_LIBBLKIO_BACKEND_H

#include <blkio.h>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <map>
#include "backend/backend_engine.h"

/** Per-I/O completion slot, referenced via blkioq user_data for dispatch. */
struct nixlBlkioIoSlot {
    bool completed = false;
    int ret = 0;
};

/** Bounce buffer tracking for drivers requiring fd-backed memory regions. */
struct nixlBlkioBounce {
    struct blkio *handle;
    struct blkio_mem_region region;
    void *user_addr;
    size_t size;
    bool needs_copy_back;
};

/** Per-registration metadata stored by the libblkio backend for each memory descriptor. */
class nixlLibblkioBackendMD : public nixlBackendMD {
public:
    nixl_mem_t memType;
    uint64_t addr;
    size_t len;
    uint64_t devId;

    nixlLibblkioBackendMD(nixl_mem_t type, uint64_t addr, size_t len, uint64_t devId)
        : nixlBackendMD(true),
          memType(type),
          addr(addr),
          len(len),
          devId(devId) {}

    ~nixlLibblkioBackendMD() override = default;
};

/**
 * Async request handle for libblkio transfers.
 *
 * Tracks submitted I/Os via per-descriptor completion slots.  When a queue is
 * polled (under the engine's queue_mutex_), completions are dispatched to the
 * correct slot via the user_data pointer passed to blkioq_read/blkioq_write.
 */
class nixlLibblkioBackendReqH : public nixlBackendReqH {
public:
    nixlLibblkioBackendReqH(nixl_xfer_op_t op, int desc_count)
        : operation_(op),
          desc_count_(desc_count),
          io_slots_(desc_count) {}

    ~nixlLibblkioBackendReqH() override = default;

    nixl_xfer_op_t operation_;
    int desc_count_;
    bool submitted_ = false;
    bool completed_ = false;
    bool had_error_ = false;

    std::vector<struct blkio *> handles_;
    std::vector<struct blkioq *> queues_;
    std::set<struct blkioq *> unique_queues_;
    std::vector<nixlBlkioIoSlot> io_slots_;

    std::vector<std::pair<struct blkio *, struct blkio_mem_region>> mapped_regions_;
    std::vector<nixlBlkioBounce> bounce_bufs_;
};

/**
 * NIXL backend engine for block device I/O via the libblkio library.
 *
 * Supports three libblkio driver types:
 *   - io_uring:                Linux io_uring for local block devices
 *   - virtio-blk-vhost-user:   vhost-user-blk protocol over Unix sockets
 *   - virtio-blk-vhost-vdpa:   vhost-vdpa kernel interface (HW or SW vDPA)
 *
 * The backend automatically handles memory region requirements: io_uring uses
 * direct buffer access, while vhost-user and vDPA drivers use bounce buffers
 * allocated via blkio_alloc_mem_region() to satisfy the fd-backed memory
 * constraint imposed by the vhost protocol.
 *
 * I/O is asynchronous: postXfer() submits requests and returns NIXL_IN_PROG,
 * checkXfer() polls for completions non-blockingly.
 */
class nixlLibblkioEngine : public nixlBackendEngine {
public:
    /**
     * Construct the backend engine and parse backend parameters.
     *
     * Recognised parameters:
     *   - api_type:    IO_URING (default), VHOST_USER, or VDPA
     *   - device_list: Comma-separated "id:type:path" entries
     *   - direct_io:   Enable O_DIRECT for io_uring (0 or 1)
     *   - io_polling:  Enable IOPOLL / poll queues (0 or 1)
     *   - num_queues:  Number of I/O queues (default 1)
     *   - queue_size:  Queue depth (io_uring num-entries or virtio queue-size)
     */
    nixlLibblkioEngine(const nixlBackendInitParams *init_params);
    ~nixlLibblkioEngine() override;

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {BLK_SEG, DRAM_SEG};
    }

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    /**
     * Register a DRAM or BLK_SEG memory region with the backend.
     *
     * For BLK_SEG, creates and configures a blkio device instance using the
     * device_list or metaInfo path.
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /**
     * Validate and prepare a transfer request.
     *
     * Local descriptors must be DRAM_SEG; remote descriptors must be BLK_SEG.
     */
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * Submit prepared I/O requests asynchronously.
     *
     * Enqueues blkioq_read/write for each descriptor pair and returns
     * NIXL_IN_PROG.  For vhost-user/vDPA drivers, allocates fd-backed bounce
     * buffers and copies write data before submission.
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * Poll for I/O completions non-blockingly.
     *
     * Returns NIXL_IN_PROG while I/Os are pending, NIXL_SUCCESS when all
     * complete successfully, or NIXL_ERR_BACKEND on I/O error.
     */
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    /**
     * Release a request handle and free associated resources.
     *
     * Drains any in-flight completions, copies back bounce buffers for
     * pending reads, unmaps memory regions, and frees bounce allocations.
     */
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    std::string api_type_;
    int num_queues_;
    int queue_size_;
    bool direct_io_;
    bool io_polling_;
    bool needs_mem_regions_ = false;
    bool needs_mem_region_fd_ = false;

    struct BlkioDevice {
        struct blkio *handle = nullptr;
        struct blkioq *queue = nullptr;
        std::string path;
        uint64_t capacity = 0;
        uint64_t devId = 0;
    };

    std::vector<std::string> rawdevs_;
    std::map<uint64_t, std::string> rawdev_map_;
    std::vector<BlkioDevice> devices_;
    size_t next_blk_reg_idx_ = 0;

    mutable std::mutex queue_mutex_;

    struct blkio *
    getBlkioHandle(uint64_t devId) const;
    struct blkioq *
    getBlkioQueue(uint64_t devId) const;
    nixl_status_t
    createBlkioDevice(const std::string &path, uint64_t devId);

    /** Poll all given queues for completions, dispatching via user_data. */
    void
    pollQueues(const std::set<struct blkioq *> &queues) const;

    /** Drain all completions for a request (blocking). Must hold queue_mutex_. */
    void
    drainReqCompletions(nixlLibblkioBackendReqH *req) const;

    /** Unmap regions and free bounce buffers for a request. */
    void
    cleanupReqResources(nixlLibblkioBackendReqH *req) const;
};

#endif // NIXL_SRC_PLUGINS_LIBBLKIO_LIBBLKIO_BACKEND_H
