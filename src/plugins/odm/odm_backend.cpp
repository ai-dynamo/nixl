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

#include <cstring>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "odm_backend.h"
#include "common/nixl_log.h"

#define ODM_DEFAULT_DMA_DEVICE "odm0"

/* 64KB-aligned, < UINT32_MAX: the FD ioctl carries a u32 transfer size and the
 * GPU dma-buf export wants 64KB-aligned ranges. */
static constexpr uint64_t ODM_MAX_FD_CHUNK = 0xFFFF0000ULL;

// ---------------------------------------------------------------------------
// Construction / device handling
// ---------------------------------------------------------------------------

nixlOdmEngine::nixlOdmEngine(const nixlBackendInitParams* init_params)
    : nixlBackendEngine(init_params),
      dma_fd_(-1),
      qid_(0),
      qid_start_(0),
      qid_end_(0)
#ifdef HAVE_CUDA
    , cuda_available_(false),
      dmabuf_supported_(false)
#endif
{
    std::string dev_name = ODM_DEFAULT_DMA_DEVICE;
    if (init_params->customParams) {
        auto it = init_params->customParams->find("dmadev_param");
        if (it != init_params->customParams->end() && !it->second.empty())
            dev_name = it->second;
        it = init_params->customParams->find("odm_qid");
        if (it != init_params->customParams->end()) {
            try {
                qid_ = static_cast<uint16_t>(std::stoul(it->second));
            } catch (...) { /* keep 0 */ }
        }
        /* Multi-queue spraying range; defaults to the single odm_qid. */
        qid_start_ = qid_;
        qid_end_ = qid_;
        it = init_params->customParams->find("odm_qid_start");
        if (it != init_params->customParams->end()) {
            try {
                qid_start_ = static_cast<uint16_t>(std::stoul(it->second));
            } catch (...) { /* keep */ }
        }
        it = init_params->customParams->find("odm_qid_end");
        if (it != init_params->customParams->end()) {
            try {
                qid_end_ = static_cast<uint16_t>(std::stoul(it->second));
            } catch (...) { /* keep */ }
        }
        /* Max dma-buf exports kept open (LRU cap); bounds open-fd growth. */
        it = init_params->customParams->find("dmabuf_cache_max");
        if (it != init_params->customParams->end()) {
            try {
                dmabuf_cache_max_ =
                    std::max<size_t>(1, static_cast<size_t>(std::stoull(it->second)));
            } catch (...) { /* keep default */ }
        }
        /* Clamp to the ODM driver's 0..15 queue range and normalize order. */
        if (qid_start_ > 15) qid_start_ = 15;
        if (qid_end_ > 15) qid_end_ = 15;
        if (qid_start_ > qid_end_) std::swap(qid_start_, qid_end_);
    }

    /* Accept either a bare device name (resolved under /dev) or an absolute
     * path (useful when the ODM char device lives elsewhere, or for testing). */
    device_path_ = (!dev_name.empty() && dev_name[0] == '/') ? dev_name
                                                             : ("/dev/" + dev_name);

    /*
     * Hard requirements: ODM has NO fallback path. If CUDA with GPUDirect is
     * unavailable, or the platform cannot export GPU VRAM as a dma-buf, we
     * refuse to initialize and report a clear message instead of silently
     * degrading to an alternate transfer path.
     */
#ifndef HAVE_CUDA
    NIXL_ERROR << "ODM: built without CUDA support (HAVE_CUDA). ODM requires CUDA "
                  "with GPUDirect (dma-buf export of GPU VRAM); refusing to "
                  "initialize. No fallback is provided.";
    initErr = true;
    return;
#else
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        NIXL_ERROR << "ODM: CUDA with GPUDirect is NOT available ("
                   << (err != cudaSuccess ? cudaGetErrorString(err) : "0 GPUs detected")
                   << "). ODM requires a CUDA GPU; refusing to initialize. "
                      "No fallback is provided.";
        initErr = true;
        return;
    }
    cuda_available_ = true;

    /* Confirm the platform can export VRAM as a dma-buf (Flow 2 step 1). */
    int supported = 0;
    CUresult cr =
        cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, 0);
    dmabuf_supported_ = (cr == CUDA_SUCCESS && supported != 0);
    if (!dmabuf_supported_) {
        const char* es = nullptr;
        cuGetErrorString(cr, &es);
        NIXL_ERROR << "ODM: GPU dma-buf export is NOT supported on this platform"
                   << (cr != CUDA_SUCCESS ? std::string(" (") + (es ? es : "?") + ")"
                                          : std::string(""))
                   << ". dma-buf is required for VRAM<->ODM transfers; refusing to "
                      "initialize. No fallback is provided.";
        initErr = true;
        return;
    }
    NIXL_INFO << "ODM: CUDA with GPUDirect available (" << deviceCount
              << " GPU(s)); dma-buf export supported";
#endif

    if (openDevice() != NIXL_SUCCESS) {
        initErr = true;
        return;
    }

    odm_num_queues_ = static_cast<int>(qid_end_ - qid_start_ + 1);
    if (odm_num_queues_ < 1)
        odm_num_queues_ = 1;
    /* One submit gate per hardware queue (see qid_locks_ in the header). */
    qid_locks_ = std::vector<std::mutex>(static_cast<size_t>(odm_num_queues_));
    /* With more than one queue, start the internal worker pool so a single
     * caller thread fans each transfer across all queues in parallel. */
    if (odm_num_queues_ > 1)
        odmPoolStart();

    NIXL_INFO << "ODM backend initialized: dma_device=" << device_path_
              << " qid=" << qid_
              << " qid_range=[" << qid_start_ << "," << qid_end_ << "]"
              << " internal_queues=" << odm_num_queues_
              << " engine=ODM/dma-buf (both directions)";
    initErr = false;
}

nixlOdmEngine::~nixlOdmEngine() {
    odmPoolStop();
#ifdef HAVE_CUDA
    closeDmabufCache();
#endif
    closeDevice();
}

nixl_status_t nixlOdmEngine::openDevice() {
    if (dma_fd_ >= 0)
        return NIXL_SUCCESS;
    dma_fd_ = open(device_path_.c_str(), O_RDWR);
    if (dma_fd_ < 0) {
        NIXL_ERROR << "ODM: open(" << device_path_ << ") failed: " << strerror(errno);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void nixlOdmEngine::closeDevice() {
    if (dma_fd_ >= 0) {
        close(dma_fd_);
        dma_fd_ = -1;
    }
}

uint16_t nixlOdmEngine::nextQid() const {
    const std::thread::id tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(thread_assign_lock_);
    auto it = thread_qid_.find(tid);
    if (it != thread_qid_.end())
        return it->second;

    const uint32_t span = static_cast<uint32_t>(qid_end_ - qid_start_ + 1);
    const uint16_t assigned =
        static_cast<uint16_t>(qid_start_ + (next_thread_slot_ % span));
    next_thread_slot_++;
    thread_qid_[tid] = assigned;
    NIXL_INFO << "ODM: assigned thread to ODM qid " << assigned;
    return assigned;
}

// ---------------------------------------------------------------------------
// Memory registration
// ---------------------------------------------------------------------------

nixl_status_t nixlOdmEngine::registerMem(const nixlBlobDesc& mem,
                                         const nixl_mem_t& nixl_mem,
                                         nixlBackendMD*& out) {
    out = nullptr;
    auto supported = odmSupportedMems();
    if (std::find(supported.begin(), supported.end(), nixl_mem) == supported.end()) {
        NIXL_ERROR << "ODM: unsupported memory type: " << nixl_mem;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    auto* md = new nixlOdmMetadata();
    md->type = nixl_mem;
    md->addr = mem.addr;
    md->size = mem.len;
    md->dev_id = static_cast<uint32_t>(mem.devId);

    if (nixl_mem == ODM_MEM_SEG) {
        md->dma_addr = mem.addr;  /* device-local IOVA */
    } else if (nixl_mem == VRAM_SEG) {
#ifdef HAVE_CUDA
        /* SYNC_MEMOPS orders CUDA ops on this buffer against the ODM DMA
         * (Flow 2 step 6); required for the dma-buf export to be safe. */
        unsigned int sync = 1;
        CUresult cr = cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                            static_cast<CUdeviceptr>(mem.addr));
        if (cr != CUDA_SUCCESS) {
            const char* es = nullptr;
            cuGetErrorString(cr, &es);
            NIXL_WARN << "ODM: cuPointerSetAttribute(SYNC_MEMOPS) failed for VRAM 0x"
                      << std::hex << mem.addr << ": " << (es ? es : "?");
        }
        {
            nixl_status_t cst = ensureCudaContext(md->dev_id);
            if (cst != NIXL_SUCCESS) {
                delete md;
                return cst;
            }
            uint64_t base = mem.addr;
            uint64_t left = mem.len;
            uint64_t off = 0;
            while (left > 0) {
                const uint64_t chunk = std::min<uint64_t>(left, ODM_MAX_FD_CHUNK);
                int fd = -1;
                cst = exportVramDmabuf(base + off, chunk, fd);
                if (cst != NIXL_SUCCESS) {
                    for (const auto& pr : md->vram_preexport_chunks)
                        releaseVramDmabuf(pr.first, pr.second);
                    md->vram_preexport_chunks.clear();
                    delete md;
                    return cst;
                }
                md->vram_preexport_chunks.emplace_back(base + off, chunk);
                off += chunk;
                left -= chunk;
            }
            NIXL_DEBUG << "ODM: registerMem VRAM pre-exported " << md->vram_preexport_chunks.size()
                       << " dma-buf chunk(s)";
        }
#endif
    }

    out = md;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOdmEngine::deregisterMem(nixlBackendMD* meta) {
    auto* md = static_cast<nixlOdmMetadata*>(meta);
    if (!md)
        return NIXL_SUCCESS;
#ifdef HAVE_CUDA
    if (md->type == VRAM_SEG) {
        for (const auto& pr : md->vram_preexport_chunks)
            releaseVramDmabuf(pr.first, pr.second);
        md->vram_preexport_chunks.clear();
    }
#endif
    delete md;
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// Transfer preparation / routing
// ---------------------------------------------------------------------------

nixl_status_t nixlOdmEngine::prepXfer(const nixl_xfer_op_t& operation,
                                      const nixl_meta_dlist_t& local,
                                      const nixl_meta_dlist_t& remote,
                                      const std::string&,
                                      nixlBackendReqH*& handle,
                                      const nixl_opt_b_args_t*) const {
    int n = local.descCount();
    if (n <= 0 || n != remote.descCount())
        return NIXL_ERR_INVALID_PARAM;

    nixl_mem_t lt = local.getType();
    nixl_mem_t rt = remote.getType();

    bool vram_odm = (lt == VRAM_SEG && rt == ODM_MEM_SEG) ||
                    (lt == ODM_MEM_SEG && rt == VRAM_SEG);

    if (!vram_odm) {
        NIXL_ERROR << "ODM: only VRAM_SEG<->ODM_MEM_SEG (ODM/dma-buf) is supported; "
                   << "got local=" << lt << " remote=" << rt;
        return NIXL_ERR_INVALID_PARAM;
    }

    auto* req = new nixlOdmBackendReqH();

    /* Destination side decides direction (mirrors the ODM plugin):
     *   data into VRAM (Iliad -> VRAM)  => ODM_DIR_TO_GPU  (READ_FD)
     *   data into ODM  (VRAM -> Iliad)  => ODM_DIR_FROM_GPU (WRITE_FD) */
    bool dest_is_vram = (operation == NIXL_READ) ? (lt == VRAM_SEG) : (rt == VRAM_SEG);
    req->direction = dest_is_vram ? ODM_DIR_TO_GPU : ODM_DIR_FROM_GPU;
    req->gpu_id = static_cast<uint32_t>((lt == VRAM_SEG) ? local[0].devId
                                                         : remote[0].devId);

    NIXL_DEBUG << "ODM prepXfer: VRAM<->ODM op="
               << (operation == NIXL_READ ? "READ" : "WRITE") << " -> "
               << (req->direction == ODM_DIR_TO_GPU ? "READ_FD (Iliad->VRAM)"
                                                    : "WRITE_FD (VRAM->Iliad)")
               << " segments=" << n;

    /*
     * Coalesce contiguous descriptors. A batched transfer (e.g. nixlbench
     * --batch_size 64) arrives as N separate descriptors; when the blocks tile
     * the registered buffers they are contiguous in BOTH the GPU VA space and
     * the ODM device address space. Merging such adjacent descriptors into one
     * larger segment turns many tiny DMAs into a few large ones: it cuts the
     * per-descriptor control-plane overhead (one dma-buf export, one ioctl, one
     * doorbell, one completion wait per merged segment instead of per block) and
     * lets the multi-queue split hand each queue a large, well-amortized piece.
     * This is always correct — only truly adjacent [addr, addr+len) ranges are
     * joined — and is a no-op for non-contiguous batches.
     */
    for (int i = 0; i < n; i++) {
        auto* lmd = static_cast<nixlOdmMetadata*>(local[i].metadataP);
        auto* rmd = static_cast<nixlOdmMetadata*>(remote[i].metadataP);
        if (!lmd || !rmd || local[i].len != remote[i].len) {
            delete req;
            return NIXL_ERR_INVALID_PARAM;
        }
        OdmSegment seg;
        seg.len = local[i].len;
        if (lt == VRAM_SEG) {
            seg.gpu_va = local[i].addr;
            uint64_t off = remote[i].addr - rmd->addr;
            seg.dma_dev_addr = rmd->dma_addr + off;
        } else {
            uint64_t off = local[i].addr - lmd->addr;
            seg.dma_dev_addr = lmd->dma_addr + off;
            seg.gpu_va = remote[i].addr;
        }
        if (!req->segments.empty()) {
            OdmSegment& prev = req->segments.back();
            if (prev.gpu_va + prev.len == seg.gpu_va &&
                prev.dma_dev_addr + prev.len == seg.dma_dev_addr) {
                prev.len += seg.len;  /* extend the previous contiguous run */
                continue;
            }
        }
        req->segments.push_back(seg);
    }

    NIXL_DEBUG << "ODM prepXfer: coalesced " << n << " descriptors into "
               << req->segments.size() << " contiguous segment(s)";

    handle = req;
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// ODM / dma-buf path (Flow 2)
// ---------------------------------------------------------------------------

#ifdef HAVE_CUDA
nixl_status_t nixlOdmEngine::ensureCudaContext(uint32_t dev) const {
    CUcontext cur = nullptr;
    if (cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur != nullptr)
        return NIXL_SUCCESS;  /* already bound on this thread */

    CUdevice cudev;
    CUresult cr = cuDeviceGet(&cudev, static_cast<int>(dev));
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr;
        cuGetErrorString(cr, &es);
        NIXL_ERROR << "ODM: cuDeviceGet(" << dev << ") failed: " << (es ? es : "?");
        return NIXL_ERR_BACKEND;
    }
    CUcontext pctx = nullptr;
    cr = cuDevicePrimaryCtxRetain(&pctx, cudev);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr;
        cuGetErrorString(cr, &es);
        NIXL_ERROR << "ODM: cuDevicePrimaryCtxRetain(" << dev << ") failed: "
                   << (es ? es : "?");
        return NIXL_ERR_BACKEND;
    }
    cuCtxSetCurrent(pctx);  /* bind primary ctx for driver + runtime APIs */
    return NIXL_SUCCESS;
}

nixl_status_t nixlOdmEngine::exportVramDmabuf(uint64_t gpu_va, uint64_t len,
                                              int& out_fd) const {
    out_fd = -1;
    if (!dmabuf_supported_) {
        NIXL_ERROR << "ODM: GPU dma-buf export not supported on this platform";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const DmabufKey key(gpu_va, len);
    {
        std::lock_guard<std::mutex> lk(dmabuf_cache_mtx_);
        auto it = dmabuf_cache_.find(key);
        if (it != dmabuf_cache_.end()) {
            /* Hit: promote to MRU and pin so eviction can't close it under us. */
            dmabuf_lru_.splice(dmabuf_lru_.begin(), dmabuf_lru_, it->second.lru_it);
            it->second.pin++;
            out_fd = it->second.fd;
            NIXL_DEBUG << "ODM: reused cached dma-buf fd=" << out_fd << " for VRAM 0x"
                       << std::hex << gpu_va << " len=0x" << len << std::dec;
            return NIXL_SUCCESS;
        }
    }

    /* Re-assert SYNC_MEMOPS on the exact range (Flow 2 step 6). */
    unsigned int sync = 1;
    cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                          static_cast<CUdeviceptr>(gpu_va));

    int fd = -1;
    CUresult cr = cuMemGetHandleForAddressRange(
        &fd, static_cast<CUdeviceptr>(gpu_va), static_cast<size_t>(len),
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (cr != CUDA_SUCCESS) {
        const char* es = nullptr;
        cuGetErrorString(cr, &es);
        NIXL_ERROR << "ODM: cuMemGetHandleForAddressRange(DMA_BUF_FD) failed for VRAM 0x"
                   << std::hex << gpu_va << " len=0x" << len << ": "
                   << (es ? es : "?");
        return NIXL_ERR_BACKEND;
    }

    {
        std::lock_guard<std::mutex> lk(dmabuf_cache_mtx_);
        /* Another thread may have raced us; keep one fd and close the loser. */
        auto it = dmabuf_cache_.find(key);
        if (it != dmabuf_cache_.end()) {
            close(fd);
            dmabuf_lru_.splice(dmabuf_lru_.begin(), dmabuf_lru_, it->second.lru_it);
            it->second.pin++;
            out_fd = it->second.fd;
        } else {
            dmabuf_lru_.push_front(key);
            dmabuf_cache_.emplace(key, DmabufNode{fd, 1, dmabuf_lru_.begin()});
            out_fd = fd;
            /* Trim least-recently-used (unpinned) entries back toward the cap. */
            evictDmabufLocked();
        }
    }
    NIXL_DEBUG << "ODM: exported VRAM 0x" << std::hex << gpu_va << " len=0x" << len
               << " as dma-buf fd=" << std::dec << out_fd << " (cached)";
    return NIXL_SUCCESS;
}

/* Drop the pin taken by exportVramDmabuf; the entry becomes evictable once no
 * worker is using it. Safe to call even if the entry was already evicted. */
void nixlOdmEngine::releaseVramDmabuf(uint64_t gpu_va, uint64_t len) const {
    std::lock_guard<std::mutex> lk(dmabuf_cache_mtx_);
    auto it = dmabuf_cache_.find(DmabufKey(gpu_va, len));
    if (it != dmabuf_cache_.end() && it->second.pin > 0)
        it->second.pin--;
}

/* Close least-recently-used, unpinned exports until at/under the cap. Caller
 * must hold dmabuf_cache_mtx_. If every over-cap entry is still pinned (the
 * live working set exceeds the cap) we stop without blocking; the open-fd count
 * is then bounded by that working set, which is itself bounded per transfer. */
void nixlOdmEngine::evictDmabufLocked() const {
    /* Walk from the LRU (back) toward the MRU (front), closing unpinned fds. */
    auto it = dmabuf_lru_.end();
    while (dmabuf_cache_.size() > dmabuf_cache_max_ && it != dmabuf_lru_.begin()) {
        --it;  /* now points at a real element (back-most not yet examined) */
        auto cit = dmabuf_cache_.find(*it);
        if (cit == dmabuf_cache_.end() || cit->second.pin > 0)
            continue;  /* skip pinned (or stale) entries; scan further forward */
        if (cit->second.fd >= 0)
            close(cit->second.fd);
        dmabuf_cache_.erase(cit);
        it = dmabuf_lru_.erase(it);  /* returns the element toward the back */
    }
}

void nixlOdmEngine::closeDmabufCache() {
    std::lock_guard<std::mutex> lk(dmabuf_cache_mtx_);
    for (auto& kv : dmabuf_cache_) {
        if (kv.second.fd >= 0)
            close(kv.second.fd);
    }
    dmabuf_cache_.clear();
    dmabuf_lru_.clear();
}
#endif

/* 64KB granularity for splitting a segment across queues: the GPU dma-buf
 * export requires 64KB-aligned ranges, so partition boundaries must be too. */
static constexpr uint64_t ODM_QSPLIT_ALIGN = 0x10000ULL;
/* Don't split a transfer into pieces smaller than this. Each ODM ioctl re-runs
 * the dma-buf attach/map/detach lifecycle, so tiny pieces multiply that
 * per-ioctl overhead and lose more than the extra queue parallelism gains.
 * Only fan across more queues once each piece stays large enough to amortize it. */
static constexpr uint64_t ODM_QSPLIT_MIN_PIECE_READ = 0x400000ULL;  /* 4 MiB */
static constexpr uint64_t ODM_QSPLIT_MIN_PIECE_WRITE = 0x400000ULL; /* 4 MiB */
static constexpr uint64_t ODM_QSPLIT_WRITE_SINGLE_QUEUE_BELOW = 0x800000ULL; /* 8 MiB */

#ifdef HAVE_CUDA
/* Export one VRAM sub-range as a dma-buf and run the FD ioctl on w.qid,
 * sub-chunking ranges larger than the ioctl's u32 size limit. */
nixl_status_t nixlOdmEngine::odmDoWork(const OdmWork& w) const {
    uint64_t remaining = w.len;
    uint64_t gpu_va = w.gpu_va;
    uint64_t iova = w.iova;
    /*
     * Serialize all submissions to this hardware queue. Distinct queues use
     * distinct mutexes, so cross-queue parallelism is preserved; this only
     * prevents two caller threads (or a caller thread and a pool worker) from
     * driving the same qid's driver-side pipelined ring concurrently, which
     * otherwise corrupts completion accounting and times out under 8/16 threads.
     */
    const int lock_slot = static_cast<int>(w.qid) - static_cast<int>(qid_start_);
    std::lock_guard<std::mutex> qlk(
        qid_locks_[(lock_slot >= 0 && lock_slot < odm_num_queues_)
                       ? static_cast<size_t>(lock_slot)
                       : 0]);
    while (remaining > 0) {
        const uint64_t chunk = std::min<uint64_t>(remaining, ODM_MAX_FD_CHUNK);
        int fd = -1;
        nixl_status_t st = exportVramDmabuf(gpu_va, chunk, fd);
        if (st != NIXL_SUCCESS)
            return st;

        struct mrvl_dma_xfer_commands_fd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.dmabuf_fd = static_cast<uint32_t>(fd);
        cmd.target_iova_addr = iova;
        cmd.tranfer_size = static_cast<uint32_t>(chunk);
        cmd.tranfer_type = w.xfer_type;
        cmd.qid = w.qid;

        NIXL_DEBUG << "ODM ioctl "
                   << (w.ioctl_cmd == MRVL_CXL_DMA_READ_COMMAND_FD
                           ? "READ_FD(Iliad->VRAM)" : "WRITE_FD(VRAM->Iliad)")
                   << " fd=" << fd << " iova=0x" << std::hex << iova
                   << " size=" << std::dec << chunk << " qid=" << w.qid;

        int rc = ioctl(dma_fd_, w.ioctl_cmd, &cmd);
        int saved = errno;
        /* fd owned by the export cache; the kernel took its own dma_buf_get
         * reference for the duration of the ioctl, so the cached entry can now
         * be unpinned (and become eligible for LRU eviction). */
        releaseVramDmabuf(gpu_va, chunk);
        if (rc < 0) {
            NIXL_ERROR << "ODM: FD ioctl failed: " << strerror(saved);
            return NIXL_ERR_BACKEND;
        }
        remaining -= chunk;
        gpu_va += chunk;
        iova += chunk;
    }
    return NIXL_SUCCESS;
}

void nixlOdmEngine::odmWorkerLoop(OdmQueueWorker* qw) const {
    for (;;) {
        OdmWork w;
        {
            std::unique_lock<std::mutex> lk(qw->m);
            qw->cv.wait(lk, [&] {
                return odm_stop_.load(std::memory_order_acquire) || !qw->q.empty();
            });
            if (odm_stop_.load(std::memory_order_acquire) && qw->q.empty())
                return;
            w = qw->q.front();
            qw->q.pop();
        }
        nixl_status_t cst = ensureCudaContext(w.gpu_id);
        nixl_status_t st = (cst != NIXL_SUCCESS) ? cst : odmDoWork(w);
        OdmBatch* b = w.batch;
        if (st != NIXL_SUCCESS)
            b->status.store(st, std::memory_order_relaxed);
        if (b->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> blk(b->m);
            b->cv.notify_one();
        }
    }
}

void nixlOdmEngine::odmPoolStart() {
    if (odm_pool_started_ || odm_num_queues_ <= 1)
        return;
    odm_stop_ = false;
    odm_workers_.reserve(odm_num_queues_);
    for (int k = 0; k < odm_num_queues_; ++k) {
        auto qw = std::make_unique<OdmQueueWorker>();
        OdmQueueWorker* raw = qw.get();
        qw->th = std::thread([this, raw] { odmWorkerLoop(raw); });
        odm_workers_.push_back(std::move(qw));
    }
    odm_pool_started_ = true;
    NIXL_INFO << "ODM: started internal ODM worker pool (" << odm_num_queues_
              << " queues, qid " << qid_start_ << ".." << qid_end_ << ")";
}

void nixlOdmEngine::odmPoolStop() {
    if (!odm_pool_started_)
        return;
    odm_stop_.store(true, std::memory_order_release);
    /* Wake each worker (lock its mutex so we don't race its wait predicate). */
    for (auto& qw : odm_workers_) {
        std::lock_guard<std::mutex> lk(qw->m);
        qw->cv.notify_all();
    }
    for (auto& qw : odm_workers_) {
        if (qw->th.joinable())
            qw->th.join();
    }
    odm_workers_.clear();
    odm_pool_started_ = false;
}
#endif

nixl_status_t nixlOdmEngine::postOdmDmabuf(nixlOdmBackendReqH* req) const {
#ifndef HAVE_CUDA
    (void)req;
    NIXL_ERROR << "ODM: ODM/dma-buf path requires CUDA to export VRAM";
    return NIXL_ERR_NOT_SUPPORTED;
#else
    unsigned int ioctl_cmd = (req->direction == ODM_DIR_TO_GPU)
                                 ? MRVL_CXL_DMA_READ_COMMAND_FD
                                 : MRVL_CXL_DMA_WRITE_COMMAND_FD;
    uint32_t xfer_type = (req->direction == ODM_DIR_TO_GPU) ? ODM_XTYPE_OUTBOUND
                                                            : ODM_XTYPE_INBOUND;

    /*
     * Single queue (no pool): run inline on the caller thread, pinned to one
     * queue via nextQid() (preserves the original behavior / multi-thread spray).
     */
    if (!odm_pool_started_) {
        nixl_status_t cst = ensureCudaContext(req->gpu_id);
        if (cst != NIXL_SUCCESS)
            return cst;
        const uint16_t req_qid = nextQid();
        for (const auto& seg : req->segments) {
            OdmWork w{seg.gpu_va, seg.dma_dev_addr, seg.len, req_qid,
                      ioctl_cmd, xfer_type, req->gpu_id, nullptr};
            nixl_status_t st = odmDoWork(w);
            if (st != NIXL_SUCCESS)
                return st;
        }
        return NIXL_SUCCESS;
    }

    /*
     * Multi-queue: fan this transfer across all ODM queues in parallel. Each
     * segment is split into up to odm_num_queues_ 64KB-aligned partitions; a
     * global round-robin index spreads pieces across queues so that a BATCH of
     * many small segments (e.g. batched 256K blocks) also lands on distinct
     * queues instead of all piling onto queue 0. The caller blocks until all
     * pieces complete.
     */
    OdmBatch batch;
    std::vector<OdmWork> items;
    items.reserve(req->segments.size() * odm_num_queues_);
    int g = 0;  /* global round-robin queue index across all segments+pieces */
    for (const auto& seg : req->segments) {
        if (seg.len == 0)
            continue;
        uint64_t nq;
        if (req->direction == ODM_DIR_FROM_GPU &&
            seg.len < ODM_QSPLIT_WRITE_SINGLE_QUEUE_BELOW) {
            nq = 1;
        } else {
            const uint64_t min_piece = (req->direction == ODM_DIR_FROM_GPU)
                                           ? ODM_QSPLIT_MIN_PIECE_WRITE
                                           : ODM_QSPLIT_MIN_PIECE_READ;
            nq = seg.len / min_piece;
            if (nq < 1)
                nq = 1;
            if (nq > static_cast<uint64_t>(odm_num_queues_))
                nq = static_cast<uint64_t>(odm_num_queues_);
        }
        /* Aligned partition size so every piece starts on a 64KB boundary. */
        uint64_t part = (seg.len + nq - 1) / nq;
        part = ((part + ODM_QSPLIT_ALIGN - 1) / ODM_QSPLIT_ALIGN) * ODM_QSPLIT_ALIGN;
        if (part == 0)
            part = seg.len;
        uint64_t off = 0;
        while (off < seg.len) {
            const uint64_t len = std::min<uint64_t>(part, seg.len - off);
            items.push_back(OdmWork{seg.gpu_va + off, seg.dma_dev_addr + off, len,
                                    static_cast<uint16_t>(qid_start_ + (g % odm_num_queues_)),
                                    ioctl_cmd, xfer_type, req->gpu_id, &batch});
            off += len;
            ++g;
        }
    }
    if (items.empty())
        return NIXL_SUCCESS;

    /* Single piece (small transfer): run inline on the caller thread to avoid
     * the worker dispatch + condvar round-trip. */
    if (items.size() == 1) {
        nixl_status_t cst = ensureCudaContext(req->gpu_id);
        if (cst != NIXL_SUCCESS)
            return cst;
        OdmWork w = items[0];
        w.batch = nullptr;
        return odmDoWork(w);
    }

    batch.remaining.store(items.size(), std::memory_order_relaxed);
    batch.status.store(NIXL_SUCCESS, std::memory_order_relaxed);

    /* Enqueue each item to its queue's worker. */
    for (const auto& w : items) {
        const int k = static_cast<int>(w.qid - qid_start_);
        OdmQueueWorker* qw = odm_workers_[k].get();
        std::lock_guard<std::mutex> lk(qw->m);
        qw->q.push(w);
        qw->cv.notify_one();
    }

    /* Wait for all partitions. */
    {
        std::unique_lock<std::mutex> blk(batch.m);
        batch.cv.wait(blk, [&] {
            return batch.remaining.load(std::memory_order_acquire) == 0;
        });
    }
    return static_cast<nixl_status_t>(batch.status.load(std::memory_order_relaxed));
#endif
}

// ---------------------------------------------------------------------------
// postXfer / checkXfer / releaseReqH
// ---------------------------------------------------------------------------

nixl_status_t nixlOdmEngine::postXfer(const nixl_xfer_op_t& operation,
                                      const nixl_meta_dlist_t& local,
                                      const nixl_meta_dlist_t& remote,
                                      const std::string&,
                                      nixlBackendReqH*& handle,
                                      const nixl_opt_b_args_t*) const {
    (void)operation;
    (void)local;
    (void)remote;
    auto* req = static_cast<nixlOdmBackendReqH*>(handle);
    if (!req)
        return NIXL_ERR_INVALID_PARAM;

    if (req->posted) {
        NIXL_ERROR << "ODM: postXfer called twice on same handle";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixl_status_t st = postOdmDmabuf(req);
    if (st != NIXL_SUCCESS)
        return st;
    req->posted = true;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOdmEngine::checkXfer(nixlBackendReqH* handle) const {
    auto* req = static_cast<nixlOdmBackendReqH*>(handle);
    if (!req)
        return NIXL_ERR_INVALID_PARAM;
    return req->posted ? NIXL_SUCCESS : NIXL_ERR_NOT_POSTED;
}

nixl_status_t nixlOdmEngine::releaseReqH(nixlBackendReqH* handle) const {
    auto* req = static_cast<nixlOdmBackendReqH*>(handle);
    if (!req)
        return NIXL_SUCCESS;
    delete req;
    return NIXL_SUCCESS;
}
