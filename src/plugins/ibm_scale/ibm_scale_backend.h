/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 IBM Corporation. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_IBM_SCALE_IBM_SCALE_BACKEND_H
#define NIXL_SRC_PLUGINS_IBM_SCALE_IBM_SCALE_BACKEND_H

#include <atomic>
#include <string>
#include <vector>

#include <liburing.h>
#include <sys/types.h>
#include <sys/vfs.h>

#include "backend/backend_engine.h"
#include "file/file_path_mode.h"
#include "nixl_types.h"

// IBM Storage Scale plugin version information (single source of truth)
inline constexpr const char *IBM_SCALE_PLUGIN_NAME = "IBM_SCALE";
inline constexpr const char *IBM_SCALE_PLUGIN_VERSION = "0.2.0";

// ---------------------------------------------------------------------------
// nixlScaleFileMD — FILE_SEG metadata for IBM_SCALE backend
//
// Extends nixlFilePathMD (which owns the fd via nixl::FileFd) with the
// byte range passed to registerMem and the filesystem block size sampled at
// registration time via fstatfs().  blksize is used in prepXfer to coalesce
// consecutive descriptors at filesystem block boundaries, minimising the
// number of io_uring SQEs submitted per transfer.
// ---------------------------------------------------------------------------
struct nixlScaleFileMD : public nixlFilePathMD {
    long long reg_offset;
    long long reg_length;
    // Filesystem block size (fstatfs f_bsize).  On IBM Storage Scale this
    // matches the NSD block size (e.g. 4 MiB or 8 MiB).  Falls back to 4 MiB
    // if fstatfs() fails.
    long long blksize;

    // Constructor: opens the file via nixlFilePathMD(devid, metaInfo), then
    // samples the filesystem block size via fstatfs() for coalescing.
    nixlScaleFileMD(uint64_t devid, const std::string &metaInfo, long long offset, long long length)
        : nixlFilePathMD(devid, metaInfo),
          reg_offset(offset),
          reg_length(length),
          blksize(sampleBlksize(file_fd.fd())) {}

private:
    static long long
    sampleBlksize(int fd) noexcept {
        if (fd < 0) {
            return 4194304LL;
        }

        struct statfs sfs{};

        if (fstatfs(fd, &sfs) == 0 && sfs.f_bsize > 0) {
            return (long long)sfs.f_bsize;
        }
        return 4194304LL; // 4 MiB fallback
    }
};

// ---------------------------------------------------------------------------
// nixlScaleIODesc — flattened I/O descriptor captured in prepXfer
//
// Raw fd/buf/len/offset are copied from the descriptor lists at prepXfer time
// so postXfer/checkXfer never dereference metadataP after the transfer starts.
// This eliminates the UAF risk when deregisterMem() races with checkXfer().
// ---------------------------------------------------------------------------
struct nixlScaleIODesc {
    int fd;
    void *buf;
    size_t len;
    off_t offset;
    long long blksize; // filesystem block size for coalescing arithmetic
    size_t done; // bytes completed so far (for short-I/O retry)
};

// ---------------------------------------------------------------------------
// nixlScaleBackendReqH — request handle for one prepXfer/postXfer/checkXfer
//
// Owns a private io_uring ring initialised in prepXfer.  postXfer submits
// SQEs tagged with their descriptor index; checkXfer peeks CQEs and handles
// short I/O by re-issuing remaining bytes.  No engine-level mutex is needed
// — each request is fully independent.
// ---------------------------------------------------------------------------
class nixlScaleBackendReqH : public nixlBackendReqH {
public:
    explicit nixlScaleBackendReqH(nixl_xfer_op_t op, size_t desc_count, unsigned ring_size)
        : operation_(op),
          expected_((int)desc_count),
          completed_(0),
          error_(false),
          ringOk_(false) {
        descs_.reserve(desc_count);
        // Cap at the io_uring hard limit of 32768 entries.
        unsigned depth = (ring_size > 32768u) ? 32768u : ring_size;
        int ret = io_uring_queue_init(depth, &ring_, 0);
        if (ret == 0) {
            ringOk_ = true;
        } else {
            ringInitErr_ = -ret;
        }
    }

    ~nixlScaleBackendReqH() override {
        if (ringOk_) {
            io_uring_queue_exit(&ring_);
        }
    }

    nixl_xfer_op_t
    operation() const noexcept {
        return operation_;
    }

    std::vector<nixlScaleIODesc> &
    descs() noexcept {
        return descs_;
    }

    const std::vector<nixlScaleIODesc> &
    descs() const noexcept {
        return descs_;
    }

    bool
    ringOk() const noexcept {
        return ringOk_;
    }

    int
    ringInitErr() const noexcept {
        return ringInitErr_;
    }

    struct io_uring *
    ring() noexcept {
        return &ring_;
    }

    bool
    hasError() const noexcept {
        return error_.load(std::memory_order_relaxed);
    }

    bool
    allDone() const noexcept {
        return completed_.load(std::memory_order_relaxed) >= (int)expected_;
    }

    void
    markCompleted(int n = 1) {
        completed_.fetch_add(n, std::memory_order_relaxed);
    }

    void
    markError() {
        error_.store(true, std::memory_order_relaxed);
    }

private:
    nixl_xfer_op_t operation_;
    std::atomic<int> expected_;
    std::atomic<int> completed_;
    std::atomic<bool> error_;
    bool ringOk_;
    int ringInitErr_ = 0;

    struct io_uring ring_{};

    std::vector<nixlScaleIODesc> descs_;
};

// ---------------------------------------------------------------------------
// nixlScaleEngine — IBM Storage Scale NIXL backend
//
// Design: per-request io_uring ring (allocated in prepXfer, freed in the
// handle destructor).  Concurrent requests each get their own independent
// ring with no mutex contention between them.
//
// Descriptor coalescing (prepXfer):
//   LMCache passes l1_align_bytes=4096, so a 10 MiB file produces 2560 x 4 KiB
//   descriptors that are contiguous in both memory and file.  prepXfer merges
//   consecutive contiguous descriptors that do not cross an fstatfs block
//   boundary.  For a 10 MiB file on 8 MiB GPFS blocks this collapses 2560
//   SQEs to 2 (one per block), reducing io_uring_queue_init overhead ~13x.
//
// Parameters (via nixl_b_params_t / env vars):
//   "nixl_scale_ring_size" / NIXL_SCALE_RING_SIZE
//       io_uring queue depth per request (default 128, capped at 32768)
// ---------------------------------------------------------------------------
class nixlScaleEngine : public nixlBackendEngine {
public:
    explicit nixlScaleEngine(const nixlBackendInitParams *init_params);
    ~nixlScaleEngine() override;

    nixlScaleEngine(const nixlScaleEngine &) = delete;
    nixlScaleEngine &
    operator=(const nixlScaleEngine &) = delete;

    // ---- capability flags --------------------------------------------------
    [[nodiscard]] bool
    supportsRemote() const noexcept override {
        return false;
    }

    [[nodiscard]] bool
    supportsLocal() const noexcept override {
        return true;
    }

    [[nodiscard]] bool
    supportsNotif() const noexcept override {
        return false;
    }

    [[nodiscard]] nixl_mem_list_t
    getSupportedMems() const noexcept override {
        return {FILE_SEG, DRAM_SEG};
    }

    // ---- memory management -------------------------------------------------
    [[nodiscard]] nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    [[nodiscard]] nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    // ---- connection management (no-op, local-only) -------------------------
    [[nodiscard]] nixl_status_t
    connect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_SUCCESS;
    }

    // ---- transfer operations -----------------------------------------------
    [[nodiscard]] nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    [[nodiscard]] nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    [[nodiscard]] nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    // ---- local metadata (pass-through) -------------------------------------
    [[nodiscard]] nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

private:
    bool initialized_ = false;

    // io_uring queue depth used when allocating each per-request ring.
    // Default 128: covers batch-64 with 2x headroom for short-I/O retries.
    // Hard cap: 32768 (io_uring limit).
    unsigned ring_size_ = 128;
};

#endif // NIXL_SRC_PLUGINS_IBM_SCALE_IBM_SCALE_BACKEND_H
