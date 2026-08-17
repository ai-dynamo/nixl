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
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "backend/backend_engine.h"

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

/** Internal request handle for a single libblkio transfer operation. */
class nixlLibblkioBackendReqH : public nixlBackendReqH {
public:
    nixlLibblkioBackendReqH(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            std::vector<struct blkio *> blkio_handles);
    ~nixlLibblkioBackendReqH() override = default;

    /** Validate descriptor counts and blkio handle availability. */
    nixl_status_t
    prepXfer();

    /** Submit and synchronously complete the I/O operation. */
    nixl_status_t
    postXfer();

    /** Return the cached completion status of the last postXfer call. */
    nixl_status_t
    checkXfer();

    /**
     * Map a DRAM buffer into the blkio memory region table.
     *
     * @param handle  blkio instance owning the queue.
     * @param buf     Host virtual address of the buffer to map.
     * @param size    Size of the buffer in bytes.
     * @return NIXL_SUCCESS on success, NIXL_ERR_BACKEND on failure.
     */
    nixl_status_t
    registerBlkioBuf(struct blkio *handle, void *buf, size_t size);

    /** Unmap all previously registered DRAM buffers. */
    void
    unregisterBlkioBufs();

private:
    const nixl_xfer_op_t &operation_;
    const nixl_meta_dlist_t &local_;
    const nixl_meta_dlist_t &remote_;
    std::vector<struct blkio *> blkio_handles_;
    nixl_status_t status_;
    std::vector<std::pair<struct blkio *, blkio_mem_region>> registered_regions_;
};

/** NIXL backend engine for block device I/O via the libblkio library. */
class nixlLibblkioEngine : public nixlBackendEngine {
public:
    /**
     * Construct the backend engine and parse backend parameters.
     *
     * Recognised parameters: api_type, device_list, direct_io, io_polling,
     * num_queues, queue_size.
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
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
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
     * For BLK_SEG, creates and connects a blkio device instance using the
     * next entry from the device_list parameter.
     *
     * @param mem       Descriptor of the memory region to register.
     * @param nixl_mem  Memory segment type (DRAM_SEG or BLK_SEG).
     * @param out       Output: allocated metadata handle on success.
     * @return NIXL_SUCCESS on success, error code on failure.
     */
    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    /**
     * Deregister a memory region and free its metadata.
     *
     * @param meta  Metadata handle returned by registerMem.
     * @return NIXL_SUCCESS always.
     */
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    /**
     * Validate and prepare a transfer request.
     *
     * Local descriptors must be DRAM_SEG; remote descriptors must be BLK_SEG.
     *
     * @param operation    NIXL_READ or NIXL_WRITE.
     * @param local        Local (DRAM) descriptor list.
     * @param remote       Remote (BLK) descriptor list.
     * @param remote_agent Remote agent name (must be local agent or empty).
     * @param handle       Output: allocated request handle on success.
     * @param opt_args     Optional backend arguments.
     * @return NIXL_SUCCESS on success, error code on failure.
     */
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * Submit the prepared transfer and block until completion.
     *
     * @param handle  Request handle returned by prepXfer.
     * @return NIXL_SUCCESS on success, NIXL_ERR_BACKEND on I/O failure.
     */
    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    /**
     * Poll the completion status of a submitted transfer.
     *
     * @param handle  Request handle returned by prepXfer.
     * @return Cached completion status from the last postXfer call.
     */
    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    /**
     * Release a request handle and free associated resources.
     *
     * @param handle  Request handle to release.
     * @return NIXL_SUCCESS always.
     */
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    std::string api_type_;
    int num_queues_;
    int queue_size_;
    bool direct_io_;
    bool io_polling_;

    struct BlkioDevice {
        struct blkio *handle;
        std::string path;
        uint64_t capacity;
        uint64_t devId;
    };

    std::vector<std::string> rawdevs_;
    std::map<uint64_t, std::string> rawdev_map_;
    std::vector<BlkioDevice> devices_;
    size_t next_blk_reg_idx_;

    /** Look up the blkio handle associated with @p devId, or nullptr if not found. */
    struct blkio *
    getBlkioHandle(uint64_t devId) const;

    /**
     * Create, connect, and start a blkio instance for the given block device.
     *
     * @param path   Path to the block device (e.g. /dev/loop0).
     * @param devId  Logical device identifier used to index the handle.
     * @return NIXL_SUCCESS on success, error code on failure.
     */
    nixl_status_t
    createBlkioDevice(const std::string &path, uint64_t devId);
};

#endif // NIXL_SRC_PLUGINS_LIBBLKIO_LIBBLKIO_BACKEND_H
