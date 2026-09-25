/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_BACKEND_H
#define NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <queue>
#include <list>

#include <nixl.h>
#include <nixl_types.h>
#include "backend/backend_engine.h"
#include "odm_ioctl.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef HAVE_ODM_IO_URING
#include <liburing.h>
#endif

/**
 * @brief Memory types supported by the ODM backend.
 *
 * VRAM_SEG <-> DRAM_SEG: dma-buf path (GPU VRAM exported via
 * cuMemGetHandleForAddressRange; remote DRAM_SEG is Structera device IOVA).
 *
 * DRAM_SEG <-> DRAM_SEG: host-VA path (local DRAM_SEG = host userspace VA,
 * remote DRAM_SEG = Structera device IOVA). WRITE moves host -> device; READ
 * moves device -> host via MRVL_CXL_DMA_WRITE/READ_COMMAND.
 *
 * NOTE: io_uring spray and pre-pinned pages for the host-VA path are not
 * implemented. Those optimizations would require kernel module changes (a host-VA
 * spray ioctl and register-time pin_user_pages support).
 */
inline nixl_mem_list_t
odmSupportedMems() {
#ifdef HAVE_CUDA
    return {VRAM_SEG, DRAM_SEG};
#else
    return {DRAM_SEG};
#endif
}

/**
 * @class nixlOdmMetadata
 * @brief Metadata for a registered ODM memory region.
 */
class nixlOdmMetadata : public nixlBackendMD {
public:
    nixl_mem_t type;
    uint64_t addr;
    uint64_t size;
    uint32_t dev_id;
    uint64_t dma_addr; /* ODM device-local base (DRAM_SEG) */
    bool iova_allocated; /* true when dma_addr came from GET_IOVA (addr==0 at register) */
#ifdef HAVE_CUDA
    std::vector<std::pair<uint64_t, uint64_t>> vram_preexport_chunks;
#endif

    nixlOdmMetadata()
        : nixlBackendMD(true),
          type(VRAM_SEG),
          addr(0),
          size(0),
          dev_id(0),
          dma_addr(0),
          iova_allocated(false) {}

    ~nixlOdmMetadata() override = default;
};

/** One ODM transfer segment (VRAM dma-buf or host-VA DRAM path). */
struct OdmSegment {
    uint64_t host_addr; /* GPU VRAM VA or host DRAM userspace VA */
    uint64_t dma_dev_addr; /* Structera device-local IOVA */
    uint64_t len;
};

/**
 * @class nixlOdmBackendReqH
 * @brief Per-transfer request handle.
 *
 * ODM/dma-buf transfers complete synchronously inside the FD ioctl
 * (posted=true on return).
 */
class nixlOdmBackendReqH : public nixlBackendReqH {
public:
    std::vector<OdmSegment> segments;
    int direction; /* ODM_DIR_OUTBOUND / ODM_DIR_INBOUND */
    uint32_t gpu_id; /* CUDA device of the VRAM buffer (for ctx binding) */
    bool is_dram_to_dram; /* host DRAM <-> device DRAM (host-VA ioctl path) */
    bool posted;

    nixlOdmBackendReqH()
        : direction(ODM_DIR_OUTBOUND),
          gpu_id(0),
          is_dram_to_dram(false),
          posted(false) {}

    ~nixlOdmBackendReqH() override = default;
};

/**
 * @class nixlOdmEngine
 * @brief NIXL backend: ODM controller + GPU VRAM dma-buf (both directions).
 */
class nixlOdmEngine : public nixlBackendEngine {
public:
    explicit nixlOdmEngine(const nixlBackendInitParams *init_params);
    ~nixlOdmEngine() override;

    bool
    supportsNotif() const override {
        return false;
    }

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    struct OdmRegKey {
        uint64_t addr;
        uint64_t len;
        uint32_t dev_id;

        bool
        operator==(const OdmRegKey &o) const {
            return addr == o.addr && len == o.len && dev_id == o.dev_id;
        }
    };

    struct OdmRegKeyHash {
        size_t
        operator()(const OdmRegKey &k) const {
            return std::hash<uint64_t>{}(k.addr) ^ (std::hash<uint64_t>{}(k.len) << 1) ^
                (std::hash<uint32_t>{}(k.dev_id) << 2);
        }
    };

    int dma_fd_;
    std::string device_path_;
    mutable std::mutex auto_iova_lock_;
    std::unordered_map<OdmRegKey, uint64_t, OdmRegKeyHash> auto_iova_map_;
    uint16_t qid_; /* Backward-compat single queue id (odm_qid). */
    uint16_t qid_start_; /* ODM queue range start (inclusive). */
    uint16_t qid_end_; /* ODM queue range end (inclusive). */

    /*
     * Multi-queue spraying: a single ODM hardware queue cannot be driven
     * concurrently from multiple threads. nextQid() pins each calling thread to a
     * fixed queue, assigned round-robin across [qid_start_, qid_end_].
     */
    mutable std::mutex thread_assign_lock_;
    mutable std::unordered_map<std::thread::id, uint16_t> thread_qid_;
    mutable uint32_t next_thread_slot_{0};
    uint16_t
    nextQid() const;
#ifdef HAVE_CUDA
    bool cuda_available_;
    bool dmabuf_supported_;
#endif

    nixl_status_t
    openDevice();
    void
    closeDevice();

    nixl_status_t
    postOdmDmabuf(nixlOdmBackendReqH *req) const;

    /*
     * Host-VA ioctl path for DRAM_SEG <-> DRAM_SEG (local host, remote device).
     * NOTE: io_uring spray and pre-pinned pages are not implemented here; kernel
     * module changes would be required for those optimizations.
     */
    nixl_status_t
    postOdmHostVa(nixlOdmBackendReqH *req) const;

#ifdef HAVE_ODM_IO_URING
    nixl_status_t
    postOdmDmabufUring(nixlOdmBackendReqH *req) const;
    void
    odmUringInit();
    void
    odmUringDestroy();

    mutable struct io_uring uring_;
    bool use_io_uring_{false};
    /*
     * liburing rings are not thread-safe. nixlbench posts from many threads on
     * one engine; without this gate concurrent submit/wait corrupts the ring
     * and surfaces as uring_cmd EBADF under multi-thread block-size sweeps.
     */
    mutable std::mutex uring_mtx_;
#endif

#ifdef HAVE_CUDA
    /*
     * Internal multi-queue worker pool: splits each transfer across queues so a
     * single caller thread can saturate PCIe.
     */
    struct OdmBatch {
        std::atomic<size_t> remaining{0};
        std::atomic<int> status{0}; /* nixl_status_t of first error */
        std::mutex m;
        std::condition_variable cv;
    };

    struct OdmWork {
        uint64_t gpu_va;
        uint64_t iova;
        uint64_t len;
        uint16_t qid;
        unsigned int ioctl_cmd;
        uint32_t xfer_type;
        uint32_t gpu_id;
        OdmBatch *batch;
    };

    struct OdmQueueWorker {
        std::queue<OdmWork> q;
        std::mutex m;
        std::condition_variable cv;
        std::thread th;
    };

    nixl_status_t
    odmDoWork(const OdmWork &w) const;
    void
    odmWorkerLoop(OdmQueueWorker *qw) const;
    void
    odmPoolStart();
    void
    odmPoolStop();

    mutable std::vector<std::unique_ptr<OdmQueueWorker>> odm_workers_;
    mutable std::atomic<bool> odm_stop_{false};
    bool odm_pool_started_{false};
#endif

    int odm_num_queues_{1};

    /*
     * Per-hardware-queue submit gate. A single ODM hardware queue's driver-side
     * pipelined-submit/completion state is not safe for concurrent submitters.
     * The worker pool already serializes each queue, but the inline fast paths
     * in postOdmDmabuf() (single-piece and no-pool) run on the caller thread and
     * can otherwise collide on the same qid when many nixlbench threads post at
     * once. Holding qid_locks_[qid - qid_start_] around the ioctl in odmDoWork()
     * caps concurrent in-flight submissions to at most one per queue (i.e. the
     * queue count) while preserving cross-queue parallelism.
     */
    mutable std::vector<std::mutex> qid_locks_;
#ifdef HAVE_CUDA
    nixl_status_t
    ensureCudaContext(uint32_t dev) const;
    nixl_status_t
    exportVramDmabuf(uint64_t gpu_va, uint64_t len, int &out_fd) const;
    void
    releaseVramDmabuf(uint64_t gpu_va, uint64_t len) const;
    void
    evictDmabufRange(uint64_t gpu_va, uint64_t len) const;
    void
    evictDmabufLocked() const;
    void
    closeDmabufCache();

    using DmabufKey = std::pair<uint64_t, uint64_t>;

    struct DmabufNode {
        int fd;
        uint32_t pin;
        std::list<DmabufKey>::iterator lru_it;
    };

    mutable std::mutex dmabuf_cache_mtx_;
    mutable std::list<DmabufKey> dmabuf_lru_;
    mutable std::map<DmabufKey, DmabufNode> dmabuf_cache_;
    size_t dmabuf_cache_max_{512};
#endif
};

#endif /* NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_BACKEND_H */
