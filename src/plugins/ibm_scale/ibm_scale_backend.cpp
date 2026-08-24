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
 *
 * IBM Storage Scale NIXL backend implementation.
 *
 * io_uring ring ownership (per-request ring):
 *   Each nixlScaleBackendReqH owns one io_uring ring allocated in prepXfer
 *   and freed in the handle destructor.  Concurrent requests each get their
 *   own independent ring with no mutex contention between them.
 *
 * Descriptor coalescing (prepXfer):
 *   LMCache passes l1_align_bytes=4096, so a 10 MiB file produces 2560 x 4 KiB
 *   descriptors contiguous in both memory and file.  prepXfer merges consecutive
 *   contiguous descriptors whose merged range stays within one filesystem block
 *   (fstatfs f_bsize).  For a 10 MiB file on 8 MiB GPFS blocks this collapses
 *   2560 SQEs to 2, reducing io_uring_queue_init overhead approximately 13x.
 *
 * Short I/O:
 *   io_uring read/write ops on regular files can return partial byte counts.
 *   checkXfer detects this, advances done, and re-submits a follow-up SQE for
 *   the remaining bytes.  The retry reuses the same descriptor slot so the
 *   expected completion count is unchanged.
 *
 * Synchronous fallback:
 *   If io_uring_queue_init() fails (e.g. RLIMIT_MEMLOCK), postXfer falls
 *   back to pread()/pwrite() transparently.
 */

#include "ibm_scale_backend.h"

#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "common/nixl_log.h"

namespace {

long long
scaleParamLl(const nixl_b_params_t *params, const char *key, long long def) {
    if (!params) {
        return def;
    }
    auto it = params->find(key);
    if (it == params->end()) {
        return def;
    }
    char *end = nullptr;
    long long v = strtoll(it->second.c_str(), &end, 10);
    return (end && *end == '\0') ? v : def;
}

nixlScaleBackendReqH &
castScaleHandle(nixlBackendReqH *handle) {
    if (!handle) {
        throw std::invalid_argument("IBM_SCALE: received null handle");
    }
    return static_cast<nixlScaleBackendReqH &>(*handle);
}

unsigned
nextPow2(unsigned n) {
    if (n == 0) {
        return 1;
    }
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

nixlScaleEngine::nixlScaleEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    if (!init_params) {
        initErr = true;
        NIXL_ERROR << "IBM_SCALE: null init_params";
        return;
    }

    auto envLl = [](const char *var, long long def) -> long long {
        const char *v = getenv(var);
        if (!v || !*v) {
            return def;
        }
        char *end = nullptr;
        long long r = strtoll(v, &end, 10);
        return (end && *end == '\0') ? r : def;
    };

    ring_size_ = nextPow2((unsigned)envLl("NIXL_SCALE_RING_SIZE", 128));
    if (ring_size_ > 32768u) {
        ring_size_ = 32768u;
    }

    // customParams override env vars.
    if (init_params->customParams) {
        const nixl_b_params_t *p = init_params->customParams;
        long long rs = scaleParamLl(p, "nixl_scale_ring_size", (long long)ring_size_);
        unsigned clamped = nextPow2((unsigned)(rs > 1 ? rs : ring_size_));
        ring_size_ = (clamped > 32768u) ? 32768u : clamped;
    }

    NIXL_INFO << "IBM_SCALE: init ring_size=" << ring_size_ << " (per-request)";
    initialized_ = true;
    NIXL_INFO << "IBM_SCALE: backend initialized";
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

nixlScaleEngine::~nixlScaleEngine() {
    NIXL_INFO << "IBM_SCALE: backend destroyed";
}

// ---------------------------------------------------------------------------
// registerMem
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    out = nullptr;

    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (nixl_mem == DRAM_SEG) {
        return NIXL_SUCCESS;
    }

    if (nixl_mem != FILE_SEG) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixlScaleFileMD *fmd = nullptr;
    try {
        long long offset = static_cast<long long>(mem.addr);
        long long length = static_cast<long long>(mem.len);
        fmd = new nixlScaleFileMD(static_cast<uint64_t>(mem.devId), mem.metaInfo, offset, length);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "IBM_SCALE: registerMem failed: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    out = fmd;
    NIXL_DEBUG << "IBM_SCALE: registerMem fd=" << fmd->file_fd.fd() << " off=" << fmd->reg_offset
               << " len=" << fmd->reg_length << " blksize=" << fmd->blksize;
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// deregisterMem
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::deregisterMem(nixlBackendMD *meta) {
    if (meta == nullptr) {
        return NIXL_SUCCESS;
    }
    delete static_cast<nixlScaleFileMD *>(meta);
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// prepXfer — validate, coalesce, capture descriptors into handle, alloc ring.
//
// Descriptor coalescing:
//   Merge consecutive descriptors that satisfy all of:
//     1. same fd
//     2. local buf is contiguous: prev.buf + prev.len == cur.buf
//     3. file offset is contiguous: prev_end == cur.offset
//     4. the merged end does NOT cross a filesystem block boundary
//        i.e. (prev.offset + merged_len) <= blk_end
//
//   Condition 4 prevents cross-block single reads that some filesystems
//   may handle less efficiently than two independent block-aligned reads.
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (remote_agent != localAgent) {
        NIXL_ERROR << "IBM_SCALE: prepXfer remote_agent '" << remote_agent << "' != localAgent '"
                   << localAgent << "'";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.getType() != DRAM_SEG) {
        NIXL_ERROR << "IBM_SCALE: prepXfer local must be DRAM_SEG, got " << (int)local.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (remote.getType() != FILE_SEG) {
        NIXL_ERROR << "IBM_SCALE: prepXfer remote must be FILE_SEG, got " << (int)remote.getType();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "IBM_SCALE: prepXfer descriptor count mismatch local=" << local.descCount()
                   << " remote=" << remote.descCount();
        return NIXL_ERR_INVALID_PARAM;
    }
    if (local.descCount() == 0) {
        NIXL_ERROR << "IBM_SCALE: prepXfer empty descriptor lists";
        return NIXL_ERR_INVALID_PARAM;
    }

    // Pass 1: capture all descriptors verbatim, validating metadata pointers
    // and length consistency.
    std::vector<nixlScaleIODesc> raw;
    raw.reserve((size_t)local.descCount());

    for (auto [l_it, r_it] = std::make_pair(local.begin(), remote.begin());
         l_it != local.end() && r_it != remote.end();
         ++l_it, ++r_it) {
        if (r_it->metadataP == nullptr) {
            NIXL_ERROR << "IBM_SCALE: prepXfer null metadataP in remote descriptor";
            return NIXL_ERR_INVALID_PARAM;
        }
        if (l_it->len < r_it->len) {
            NIXL_ERROR << "IBM_SCALE: prepXfer local descriptor length " << l_it->len
                       << " < remote descriptor length " << r_it->len;
            return NIXL_ERR_INVALID_PARAM;
        }
        auto *fmd = static_cast<nixlScaleFileMD *>(r_it->metadataP);
        nixlScaleIODesc d{};
        d.fd = fmd->file_fd.fd();
        d.buf = reinterpret_cast<void *>(l_it->addr);
        d.len = r_it->len;
        d.offset = static_cast<off_t>(r_it->addr);
        d.blksize = fmd->blksize;
        d.done = 0;
        raw.push_back(d);
    }

    // Pass 2: coalesce consecutive descriptors that are contiguous in both
    // memory and file, keeping the merged range within one filesystem block.
    std::vector<nixlScaleIODesc> coalesced;
    coalesced.reserve(raw.size());

    for (const nixlScaleIODesc &cur : raw) {
        if (!coalesced.empty()) {
            nixlScaleIODesc &prev = coalesced.back();
            long long blksz = (prev.blksize > 0) ? prev.blksize : 4194304LL;
            long long prevFileEnd = prev.offset + static_cast<long long>(prev.len);
            long long blkEnd = (prev.offset / blksz + 1) * blksz;
            long long mergedEnd = prevFileEnd + static_cast<long long>(cur.len);

            if (prev.fd == cur.fd &&
                static_cast<char *>(prev.buf) + prev.len == static_cast<char *>(cur.buf) &&
                prevFileEnd == cur.offset && mergedEnd <= blkEnd) {
                prev.len += cur.len;
                continue;
            }
        }
        coalesced.push_back(cur);
    }

    NIXL_DEBUG << "IBM_SCALE: prepXfer op=" << (operation == NIXL_READ ? "READ" : "WRITE")
               << " raw=" << raw.size() << " coalesced=" << coalesced.size();

    unsigned ringDepth = nextPow2((unsigned)coalesced.size() * 2);
    if (ringDepth < ring_size_) {
        ringDepth = ring_size_;
    }
    if (ringDepth > 32768u) {
        ringDepth = 32768u;
    }

    auto *req = new nixlScaleBackendReqH(operation, coalesced.size(), ringDepth);
    if (!req->ringOk()) {
        NIXL_WARN << "IBM_SCALE: ring init failed (depth=" << ringDepth
                  << " err=" << req->ringInitErr() << ") — using sync fallback";
    }
    req->descs() = std::move(coalesced);

    handle = req;
    return NIXL_SUCCESS;
}

// ---------------------------------------------------------------------------
// postXfer — submit all SQEs to the request's own ring.
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (!initialized_) {
        return NIXL_ERR_BACKEND;
    }

    if (!handle) {
        NIXL_ERROR << "IBM_SCALE: postXfer null handle";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlScaleBackendReqH &req = castScaleHandle(handle);
    const bool isRead = (req.operation() == NIXL_READ);

    // ── io_uring path ─────────────────────────────────────────────────────
    if (req.ringOk()) {
        struct io_uring *ring = req.ring();
        size_t queued = 0;

        for (size_t idx = 0; idx < req.descs().size(); ++idx) {
            const nixlScaleIODesc &d = req.descs()[idx];

            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (!sqe) {
                // Ring is full — flush what we have, then retry once.
                int flushed = io_uring_submit(ring);
                if (flushed > 0) {
                    queued -= (size_t)flushed;
                }
                sqe = io_uring_get_sqe(ring);
                if (!sqe) {
                    NIXL_ERROR << "IBM_SCALE: postXfer SQ full even after flush"
                               << " idx=" << idx << " descs=" << req.descs().size();
                    req.markError();
                    return NIXL_ERR_BACKEND;
                }
            }

            if (isRead) {
                io_uring_prep_read(sqe, d.fd, d.buf, (unsigned)d.len, d.offset);
            } else {
                io_uring_prep_write(sqe, d.fd, d.buf, (unsigned)d.len, d.offset);
            }
            io_uring_sqe_set_data64(sqe, (uint64_t)idx);
            ++queued;
        }

        // Submit any remaining queued SQEs.
        while (queued > 0) {
            int ret = io_uring_submit(ring);
            if (ret < 0) {
                if (ret == -EINTR) {
                    continue;
                }
                NIXL_ERROR << "IBM_SCALE: postXfer io_uring_submit failed: " << -ret << " ("
                           << strerror(-ret) << ")";
                req.markError();
                return NIXL_ERR_BACKEND;
            }
            queued -= (size_t)ret;
        }

        NIXL_DEBUG << "IBM_SCALE: postXfer submitted " << req.descs().size()
                   << " SQEs op=" << (isRead ? "READ" : "WRITE");
        return NIXL_IN_PROG;
    }

    // ── Synchronous fallback (io_uring unavailable) ────────────────────────
    for (nixlScaleIODesc &d : req.descs()) {
        while (d.done < d.len) {
            char *ptr = static_cast<char *>(d.buf) + d.done;
            size_t remain = d.len - d.done;
            off_t off = d.offset + (off_t)d.done;

            ssize_t n = isRead ? pread(d.fd, ptr, remain, off) : pwrite(d.fd, ptr, remain, off);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                NIXL_ERROR << "IBM_SCALE: postXfer sync " << (isRead ? "pread" : "pwrite")
                           << " fd=" << d.fd << " off=" << (long long)off << " len=" << remain
                           << " errno=" << errno << " (" << strerror(errno) << ")";
                req.markError();
                return NIXL_ERR_BACKEND;
            }
            if (n == 0) {
                NIXL_ERROR << "IBM_SCALE: postXfer sync unexpected EOF fd=" << d.fd
                           << " off=" << (long long)off << " done=" << d.done << " total=" << d.len;
                req.markError();
                return NIXL_ERR_BACKEND;
            }
            d.done += (size_t)n;
        }
    }

    req.markCompleted((int)req.descs().size());
    return NIXL_IN_PROG;
}

// ---------------------------------------------------------------------------
// checkXfer — harvest CQEs from the request's own ring; handle short I/O.
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlScaleBackendReqH &req = castScaleHandle(handle);

    if (req.hasError()) {
        return NIXL_ERR_BACKEND;
    }

    // Synchronous fallback completed everything in postXfer.
    if (!req.ringOk()) {
        return req.allDone() ? NIXL_SUCCESS : NIXL_IN_PROG;
    }

    const bool isRead = (req.operation() == NIXL_READ);
    struct io_uring *ring = req.ring();
    struct io_uring_cqe *cqe = nullptr;

    // Drain all available CQEs without blocking.
    while (!req.allDone()) {
        int ret = io_uring_peek_cqe(ring, &cqe);
        if (ret == -EAGAIN || cqe == nullptr) {
            break;
        }

        if (ret < 0) {
            NIXL_ERROR << "IBM_SCALE: checkXfer io_uring_peek_cqe error: " << -ret << " ("
                       << strerror(-ret) << ")";
            req.markError();
            io_uring_cqe_seen(ring, cqe);
            return NIXL_ERR_BACKEND;
        }

        uint64_t idx = io_uring_cqe_get_data64(cqe);
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);

        if (res < 0) {
            NIXL_ERROR << "IBM_SCALE: checkXfer I/O error op=" << (isRead ? "READ" : "WRITE")
                       << " idx=" << idx << " res=" << res << " (" << strerror(-res) << ")";
            req.markError();
            return NIXL_ERR_BACKEND;
        }

        if (idx >= req.descs().size()) {
            NIXL_WARN << "IBM_SCALE: checkXfer CQE idx=" << idx
                      << " out of range (descs=" << req.descs().size() << ") — ignored";
            continue;
        }

        nixlScaleIODesc &d = req.descs()[idx];
        d.done += (size_t)res;

        if (d.done < d.len) {
            // Short I/O — re-submit for the remaining bytes.
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (!sqe) {
                io_uring_submit(ring);
                sqe = io_uring_get_sqe(ring);
                if (!sqe) {
                    NIXL_ERROR << "IBM_SCALE: checkXfer SQ full during short-I/O"
                               << " re-submit fd=" << d.fd;
                    req.markError();
                    return NIXL_ERR_BACKEND;
                }
            }

            char *ptr = static_cast<char *>(d.buf) + d.done;
            size_t remain = d.len - d.done;
            off_t off = d.offset + (off_t)d.done;

            if (isRead) {
                io_uring_prep_read(sqe, d.fd, ptr, (unsigned)remain, off);
            } else {
                io_uring_prep_write(sqe, d.fd, ptr, (unsigned)remain, off);
            }
            io_uring_sqe_set_data64(sqe, idx);

            int sub = io_uring_submit(ring);
            if (sub < 0) {
                NIXL_ERROR << "IBM_SCALE: checkXfer short-I/O retry submit failed: " << -sub << " ("
                           << strerror(-sub) << ")";
                req.markError();
                return NIXL_ERR_BACKEND;
            }

            continue;
        }

        req.markCompleted();
    }

    return req.allDone() ? NIXL_SUCCESS : NIXL_IN_PROG;
}

// ---------------------------------------------------------------------------
// releaseReqH
// ---------------------------------------------------------------------------

nixl_status_t
nixlScaleEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_ERR_INVALID_PARAM;
    }
    delete handle;
    return NIXL_SUCCESS;
}
