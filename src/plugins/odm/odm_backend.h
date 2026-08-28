/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_ODM_ODM_BACKEND_H
#define NIXL_SRC_PLUGINS_ODM_ODM_BACKEND_H

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

/**
 * @brief Memory types supported by the ODM backend.
 *
 * VRAM_SEG <-> ODM_MEM_SEG: ODM DMA controller via dma-buf for both directions
 * (Iliad DRAM <-> GPU VRAM). VRAM is exported with cuMemGetHandleForAddressRange;
 * CUDA is required only for that export, not for a separate DAX path.
 */
inline nixl_mem_list_t
odmSupportedMems() {
#ifdef HAVE_CUDA
    return {VRAM_SEG, ODM_MEM_SEG};
#else
    return {ODM_MEM_SEG};
#endif
}

/**
 * @class nixlOdmMetadata
 * @brief Metadata for a registered ODM memory region.
 */
class nixlOdmMetadata : public nixlBackendMD {
public:
    nixl_mem_t type;
    uint64_t   addr;
    uint64_t   size;
    uint32_t   dev_id;
    uint64_t   dma_addr;    /* ODM device-local base (ODM_MEM_SEG) */
#ifdef HAVE_CUDA
    std::vector<std::pair<uint64_t, uint64_t>> vram_preexport_chunks;
#endif

    nixlOdmMetadata()
        : nixlBackendMD(true), type(VRAM_SEG), addr(0), size(0),
          dev_id(0), dma_addr(0) {}
    ~nixlOdmMetadata() override = default;
};

/** One ODM/dma-buf transfer segment. */
struct OdmSegment {
    uint64_t gpu_va;        /* GPU VRAM virtual address to export as dma-buf */
    uint64_t dma_dev_addr;  /* Iliad/Structera device-local IOVA */
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
    int      direction;  /* ODM_DIR_TO_GPU / ODM_DIR_FROM_GPU */
    uint32_t gpu_id;     /* CUDA device of the VRAM buffer (for ctx binding) */
    bool     posted;

    nixlOdmBackendReqH()
        : direction(ODM_DIR_TO_GPU), gpu_id(0), posted(false)
    {}
    ~nixlOdmBackendReqH() override = default;
};

/**
 * @class nixlOdmEngine
 * @brief NIXL backend: ODM controller + GPU VRAM dma-buf (both directions).
 */
class nixlOdmEngine : public nixlBackendEngine {
public:
    explicit nixlOdmEngine(const nixlBackendInitParams* init_params);
    ~nixlOdmEngine() override;

    bool supportsNotif() const override { return false; }
    bool supportsRemote() const override { return false; }
    bool supportsLocal() const override { return true; }

    nixl_mem_list_t getSupportedMems() const override { return odmSupportedMems(); }

    nixl_status_t connect(const std::string& remote_agent) override { return NIXL_SUCCESS; }
    nixl_status_t disconnect(const std::string& remote_agent) override { return NIXL_SUCCESS; }
    nixl_status_t loadLocalMD(nixlBackendMD* input, nixlBackendMD*& output) override {
        output = input;
        return NIXL_SUCCESS;
    }
    nixl_status_t unloadMD(nixlBackendMD* input) override { return NIXL_SUCCESS; }

    nixl_status_t registerMem(const nixlBlobDesc& mem,
                              const nixl_mem_t& nixl_mem,
                              nixlBackendMD*& out) override;
    nixl_status_t deregisterMem(nixlBackendMD* meta) override;

    nixl_status_t prepXfer(const nixl_xfer_op_t& operation,
                           const nixl_meta_dlist_t& local,
                           const nixl_meta_dlist_t& remote,
                           const std::string& remote_agent,
                           nixlBackendReqH*& handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;

    nixl_status_t postXfer(const nixl_xfer_op_t& operation,
                           const nixl_meta_dlist_t& local,
                           const nixl_meta_dlist_t& remote,
                           const std::string& remote_agent,
                           nixlBackendReqH*& handle,
                           const nixl_opt_b_args_t* opt_args = nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

private:
    int         dma_fd_;
    std::string device_path_;
    uint16_t    qid_;       /* Backward-compat single queue id (odm_qid). */
    uint16_t    qid_start_; /* ODM queue range start (inclusive). */
    uint16_t    qid_end_;   /* ODM queue range end (inclusive). */

    /*
     * Multi-queue spraying: a single ODM hardware queue cannot be driven
     * concurrently from multiple threads. nextQid() pins each calling thread to a
     * fixed queue, assigned round-robin across [qid_start_, qid_end_].
     */
    mutable std::mutex thread_assign_lock_;
    mutable std::unordered_map<std::thread::id, uint16_t> thread_qid_;
    mutable uint32_t next_thread_slot_{0};
    uint16_t nextQid() const;
#ifdef HAVE_CUDA
    bool        cuda_available_;
    bool        dmabuf_supported_;
#endif

    nixl_status_t openDevice();
    void          closeDevice();

    nixl_status_t postOdmDmabuf(nixlOdmBackendReqH* req) const;

    /*
     * Internal multi-queue worker pool: splits each transfer across queues so a
     * single caller thread can saturate PCIe.
     */
    struct OdmBatch {
        std::atomic<size_t> remaining{0};
        std::atomic<int>    status{0};   /* nixl_status_t of first error */
        std::mutex          m;
        std::condition_variable cv;
    };
    struct OdmWork {
        uint64_t      gpu_va;
        uint64_t      iova;
        uint64_t      len;
        uint16_t      qid;
        unsigned int  ioctl_cmd;
        uint32_t      xfer_type;
        uint32_t      gpu_id;
        OdmBatch*     batch;
    };
    struct OdmQueueWorker {
        std::queue<OdmWork>     q;
        std::mutex              m;
        std::condition_variable cv;
        std::thread             th;
    };

    nixl_status_t odmDoWork(const OdmWork& w) const;
    void          odmWorkerLoop(OdmQueueWorker* qw) const;
    void          odmPoolStart();
    void          odmPoolStop();

    int  odm_num_queues_{1};
    mutable std::vector<std::unique_ptr<OdmQueueWorker>> odm_workers_;
    mutable std::atomic<bool> odm_stop_{false};
    bool odm_pool_started_{false};

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
    nixl_status_t ensureCudaContext(uint32_t dev) const;
    nixl_status_t exportVramDmabuf(uint64_t gpu_va, uint64_t len, int& out_fd) const;
    void          releaseVramDmabuf(uint64_t gpu_va, uint64_t len) const;
    void          evictDmabufLocked() const;
    void          closeDmabufCache();

    using DmabufKey = std::pair<uint64_t, uint64_t>;
    struct DmabufNode {
        int      fd;
        uint32_t pin;
        std::list<DmabufKey>::iterator lru_it;
    };
    mutable std::mutex dmabuf_cache_mtx_;
    mutable std::list<DmabufKey> dmabuf_lru_;
    mutable std::map<DmabufKey, DmabufNode> dmabuf_cache_;
    size_t dmabuf_cache_max_{512};
#endif
};

#endif /* NIXL_SRC_PLUGINS_ODM_ODM_BACKEND_H */
