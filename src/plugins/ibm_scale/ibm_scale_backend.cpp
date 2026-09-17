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

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>

#ifdef HAVE_GPFS_FCNTL
#include <gpfs_fcntl.h>
#endif

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

    ringSize_ = nextPow2((unsigned)envLl("NIXL_SCALE_RING_SIZE", 128));
    if (ringSize_ > 32768u) {
        ringSize_ = 32768u;
    }

    disableMarHints_ = envLl("NIXL_SCALE_DISABLE_MAR_HINTS", 1) != 0;

    // customParams override env vars.
    if (init_params->customParams) {
        const nixl_b_params_t *p = init_params->customParams;
        long long rs = scaleParamLl(p, "nixl_scale_ring_size", (long long)ringSize_);
        unsigned clamped = nextPow2((unsigned)(rs > 1 ? rs : ringSize_));
        ringSize_ = (clamped > 32768u) ? 32768u : clamped;

        disableMarHints_ =
            scaleParamLl(p, "nixl_scale_disable_mar_hints", disableMarHints_ ? 1 : 0) != 0;
    }

    NIXL_INFO << "IBM_SCALE: init ring_size=" << ringSize_
              << " disable_mar_hints=" << disableMarHints_ << " (per-request)";
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
#ifdef HAVE_GPFS_FCNTL
        if (fmd->file_fd.fd() >= 0) {
            struct {
                gpfsFcntlHeader_t header;
                gpfsAccessRange_t accessRange;
            } arg{};

            arg.header.totalLength = sizeof(arg);
            arg.header.fcntlVersion = GPFS_FCNTL_CURRENT_VERSION;
            arg.header.fcntlReserved = 0;

            arg.accessRange.structLen = sizeof(gpfsAccessRange_t);
            arg.accessRange.structType = GPFS_FCNTL_ACCESS_RANGE;
            arg.accessRange.start = offset;
            arg.accessRange.length = length;
            arg.accessRange.accuracy = GPFS_ACCESS_SEQUENTIAL;

            int ret = gpfs_fcntl(fmd->file_fd.fd(), &arg);
            if (ret != 0) {
                NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl registration access hint failed for fd="
                           << fmd->file_fd.fd() << " off=" << offset << " len=" << length
                           << " ret=" << ret << " errno=" << errno << " (" << strerror(errno)
                           << ")";
            } else {
                NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl registration access hint registered "
                              "successfully for fd="
                           << fmd->file_fd.fd() << " off=" << offset << " len=" << length;
            }
        }
#endif
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "IBM_SCALE: registerMem failed: " << e.what();
        return NIXL_ERR_BACKEND;
    }

    out = fmd;
    NIXL_DEBUG << "IBM_SCALE: registerMem fd=" << fmd->file_fd.fd() << " off=" << fmd->regOffset
               << " len=" << fmd->regLength << " blksize=" << fmd->blksize;
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
    auto *fmd = static_cast<nixlScaleFileMD *>(meta);
#ifdef HAVE_GPFS_FCNTL
    if (fmd && fmd->file_fd.fd() >= 0) {
        struct {
            gpfsFcntlHeader_t header;
            gpfsFreeRange_t freeRange;
        } arg{};

        arg.header.totalLength = sizeof(arg);
        arg.header.fcntlVersion = GPFS_FCNTL_CURRENT_VERSION;
        arg.header.fcntlReserved = 0;

        arg.freeRange.structLen = sizeof(gpfsFreeRange_t);
        arg.freeRange.structType = GPFS_FCNTL_FREE_RANGE;
        arg.freeRange.start = fmd->regOffset;
        arg.freeRange.length = fmd->regLength;

        int ret = gpfs_fcntl(fmd->file_fd.fd(), &arg);
        if (ret != 0) {
            NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl free range hint failed: " << errno << " ("
                       << strerror(errno) << ")";
        } else {
            NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl free range hint released successfully";
        }
    }
#endif
    delete fmd;
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
    if (ringDepth < ringSize_) {
        ringDepth = ringSize_;
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

    // ── Quiesce any outstanding operations from a prior run on this handle ──
    if (req.ringOk()) {
        while (req.inFlight() > 0) {
            struct io_uring_cqe *cqe = nullptr;
            int ret = io_uring_wait_cqe(req.ring(), &cqe);
            if (ret < 0) {
                if (ret == -EINTR) {
                    continue;
                }
                NIXL_ERROR << "IBM_SCALE: postXfer quiescing wait_cqe failed: " << -ret;
                break;
            }
            if (cqe) {
                io_uring_cqe_seen(req.ring(), cqe);
                req.decInFlight();
            }
        }
    }

    req.resetState();

    const bool isRead = (req.operation() == NIXL_READ);

#ifdef HAVE_GPFS_FCNTL
    if (!disableMarHints_) {
        const bool isWrite = !isRead;
        for (const nixlScaleIODesc &d : req.descs()) {
            if (d.fd >= 0) {
                struct {
                    gpfsFcntlHeader_t header;
                    gpfsAccessRange_t accessRange;
                } arg{};

                arg.header.totalLength = sizeof(arg);
                arg.header.fcntlVersion = GPFS_FCNTL_CURRENT_VERSION;
                arg.header.fcntlReserved = 0;

                arg.accessRange.structLen = sizeof(gpfsAccessRange_t);
                arg.accessRange.structType = GPFS_FCNTL_ACCESS_RANGE;
                arg.accessRange.start = d.offset;
                arg.accessRange.length = d.len;
                arg.accessRange.accuracy = isWrite ? GPFS_ACCESS_WRITE : GPFS_ACCESS_SEQUENTIAL;

                int ret = gpfs_fcntl(d.fd, &arg);
                if (ret != 0) {
                    NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl transfer access hint failed for fd="
                               << d.fd << " off=" << d.offset << " len=" << d.len << " ret=" << ret
                               << " errno=" << errno << " (" << strerror(errno) << ")";
                } else {
                    NIXL_DEBUG << "IBM_SCALE: gpfs_fcntl transfer access hint sent for fd=" << d.fd
                               << " off=" << d.offset << " len=" << d.len;
                }
            }
        }
    }
#endif

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
                    req.addInFlight(flushed);
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

            const auto submitLen = static_cast<unsigned>(
                std::min(d.len, static_cast<size_t>(std::numeric_limits<unsigned>::max())));
            if (isRead) {
                io_uring_prep_read(sqe, d.fd, d.buf, submitLen, d.offset);
            } else {
                io_uring_prep_write(sqe, d.fd, d.buf, submitLen, d.offset);
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
            req.addInFlight(ret);
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

    // Synchronous fallback completed everything in postXfer.
    if (!req.ringOk()) {
        if (req.hasError()) {
            return NIXL_ERR_BACKEND;
        }
        return req.allDone() ? NIXL_SUCCESS : NIXL_IN_PROG;
    }

    const bool isRead = (req.operation() == NIXL_READ);
    struct io_uring *ring = req.ring();
    struct io_uring_cqe *cqe = nullptr;

    // Drain all available CQEs without blocking.
    while (true) {
        int ret = io_uring_peek_cqe(ring, &cqe);
        if (ret == -EAGAIN || cqe == nullptr) {
            break;
        }

        if (ret < 0) {
            NIXL_ERROR << "IBM_SCALE: checkXfer io_uring_peek_cqe error: " << -ret << " ("
                       << strerror(-ret) << ")";
            req.markError();
            break;
        }

        uint64_t idx = io_uring_cqe_get_data64(cqe);
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        req.decInFlight();

        if (req.hasError()) {
            // Already in error state, just discard/drain this CQE.
            continue;
        }

        if (res < 0) {
            NIXL_ERROR << "IBM_SCALE: checkXfer I/O error op=" << (isRead ? "READ" : "WRITE")
                       << " idx=" << idx << " res=" << res << " (" << strerror(-res) << ")";
            req.markError();
            continue;
        }

        if (idx >= req.descs().size()) {
            NIXL_WARN << "IBM_SCALE: checkXfer CQE idx=" << idx
                      << " out of range (descs=" << req.descs().size() << ") — ignored";
            continue;
        }

        nixlScaleIODesc &d = req.descs()[idx];
        if (res == 0 && d.done < d.len) {
            NIXL_ERROR << "IBM_SCALE: checkXfer zero-byte " << (isRead ? "read" : "write")
                       << " idx=" << idx << " fd=" << d.fd << " done=" << d.done
                       << " total=" << d.len;
            req.markError();
            continue;
        }
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
                    continue;
                }
            }

            char *ptr = static_cast<char *>(d.buf) + d.done;
            size_t remain = d.len - d.done;
            off_t off = d.offset + (off_t)d.done;

            const auto submitLen = static_cast<unsigned>(
                std::min(remain, static_cast<size_t>(std::numeric_limits<unsigned>::max())));
            if (isRead) {
                io_uring_prep_read(sqe, d.fd, ptr, submitLen, off);
            } else {
                io_uring_prep_write(sqe, d.fd, ptr, submitLen, off);
            }
            io_uring_sqe_set_data64(sqe, idx);

            int sub = io_uring_submit(ring);
            if (sub < 0) {
                NIXL_ERROR << "IBM_SCALE: checkXfer short-I/O retry submit failed: " << -sub << " ("
                           << strerror(-sub) << ")";
                req.markError();
                continue;
            }
            req.addInFlight(sub);
            continue;
        }

        req.markCompleted();
    }

    if (req.hasError()) {
        return (req.inFlight() > 0) ? NIXL_IN_PROG : NIXL_ERR_BACKEND;
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
    nixlScaleBackendReqH &req = castScaleHandle(handle);
    if (req.inFlight() > 0) {
        NIXL_ERROR << "IBM_SCALE: releaseReqH failed because " << req.inFlight()
                   << " SQEs are still in flight";
        return NIXL_ERR_BACKEND;
    }
    delete handle;
    return NIXL_SUCCESS;
}
