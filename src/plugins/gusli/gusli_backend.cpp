/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gusli_backend.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#define __LOG_ERR(format, ...)                                                                    \
    {                                                                                             \
        NIXL_ERROR << absl::StrFormat (                                                           \
            "GUSLI: %s() %s[%d]" format, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
    }                                                                                             \
    while (0)
#define __LOG_DBG(format, ...)                                           \
    { NIXL_DEBUG << absl::StrFormat ("GUSLI: " format, ##__VA_ARGS__); } \
    while (0)
#define __LOG_TRC(format, ...)                                           \
    { NIXL_TRACE << absl::StrFormat ("GUSLI: " format, ##__VA_ARGS__); } \
    while (0)
#define __LOG_RETERR(rv, format, ...)                               \
    do {                                                            \
        __LOG_ERR ("nixl_err=%d, " format, (int)rv, ##__VA_ARGS__); \
        return rv;                                                  \
    } while (0)

namespace {
[[nodiscard]] nixl_status_t
conErrConv (const gusli::connect_rv rv) {
    if (rv == gusli::connect_rv::C_OK) return NIXL_SUCCESS;
    if (rv == gusli::connect_rv::C_NO_DEVICE) return NIXL_ERR_NOT_FOUND;
    if (rv == gusli::connect_rv::C_WRONG_ARGUMENTS) return NIXL_ERR_INVALID_PARAM;
    return NIXL_ERR_BACKEND;
}

[[nodiscard]] bool
isEntireIOto1Bdev (const nixl_meta_dlist_t &remote) {
    const uint64_t devId = remote[0].devId;
    const unsigned nRanges = remote.descCount();
    for (unsigned i = 1; i < nRanges; i++)
        if (devId != remote[i].devId) return false;
    return true;
}
}; // namespace

nixlGusliEngine::nixlGusliEngine (const nixlBackendInitParams *nixlInit)
    : nixlBackendEngine (nixlInit) {
    lib_ = &gusli::global_clnt_context::get();
    gusli::global_clnt_context::init_params gusli_params; // Convert nixl params to lib params
    gusli_params.log =
        stdout; // Redirect gusli logs to stdout, important errors will be printed by the plugin
    if (nixlInit && nixlInit->customParams) {
        const nixl_b_params_t *backParams = nixlInit->customParams;
        if (backParams->count ("client_name") > 0)
            gusli_params.client_name = backParams->at ("client_name").c_str();
        if (backParams->count ("max_num_simultaneous_requests") > 0)
            gusli_params.max_num_simultaneous_requests =
                std::stoi (backParams->at ("max_num_simultaneous_requests"));
        if (backParams->count ("config_file") > 0)
            gusli_params.config_file = backParams->at ("config_file").c_str();
    }
    const int rv = lib_->init (gusli_params);
    this->initErr = (rv != 0);
    if (this->initErr) {
        __LOG_ERR ("Error opening driver rv=%d", rv);
        lib_ = nullptr;
    }
}

nixlGusliEngine::~nixlGusliEngine() {
    if (lib_) {
        const int rv = lib_->destroy();
        lib_ = nullptr;
        if (rv) __LOG_ERR ("Error closing driver rv=%d", rv);
    }
}

nixl_status_t
nixlGusliEngine::bdevOpen (uint64_t devId) {
    auto it = bdevs_.find (devId);
    if (it != bdevs_.end()) {
        bdevRefcountT &v = it->second;
        v.ref_count++;
        const gusli::bdev_info &i = v.bi;
        __LOG_DBG ("Open: 0x%lx already exists[ref=%d]: fd=%d, name=%s",
                   devId,
                   v.ref_count,
                   i.bdev_descriptor,
                   i.name);
    } else {
        gusli::backend_bdev_id bdev;
        bdev.set_from (devId);
        const gusli::connect_rv rv = lib_->bdev_connect (bdev);
        if (rv != gusli::connect_rv::C_OK)
            __LOG_RETERR (conErrConv (rv), "connect uuid=%.16s rv=%d", bdev.uuid, (int)rv);
        bdevRefcountT v;
        v.ref_count = 1;
        const gusli::bdev_info &i = v.bi;
        lib_->bdev_get_info (bdev, &v.bi);
        __LOG_DBG ("Open: 0x%lx {bdev uuid=%.16s, fd=%d name=%s, block_size=%u[B], #blocks=0x%lx}",
                   devId,
                   bdev.uuid,
                   i.bdev_descriptor,
                   i.name,
                   i.block_size,
                   i.num_total_blocks);
        bdevs_[devId] = v;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGusliEngine::bdevClose (uint64_t devId) {
    auto it = bdevs_.find (devId);
    if (it == bdevs_.end()) {
        __LOG_DBG ("Close: 0x%lx not oppened", devId);
    } else {
        bdevRefcountT &v = it->second;
        const gusli::bdev_info &i = v.bi;
        v.ref_count--;
        if (v.ref_count > 0) {
            __LOG_DBG ("Close: 0x%lx still used[ref=%d]: fd=%d, name=%s",
                       devId,
                       v.ref_count,
                       i.bdev_descriptor,
                       i.name);
            return NIXL_SUCCESS;
        }
        gusli::backend_bdev_id bdev;
        bdev.set_from (devId);
        const gusli::connect_rv rv = lib_->bdev_disconnect (bdev);
        if (rv != gusli::connect_rv::C_OK) {
            v.ref_count++; // Device will not be erased from the hash so next open() / close() will
                           // be able to pick it up and possibly close later
            __LOG_RETERR (conErrConv (rv), "disconnect uuid=%.16s rv=%d", bdev.uuid, (int)rv);
        }
        __LOG_DBG ("Close: 0x%lx {bdev uuid=%.16s, fd=%d name=%s, block_size=%u[B], #blocks=0x%lx}",
                   devId,
                   bdev.uuid,
                   i.bdev_descriptor,
                   i.name,
                   i.block_size,
                   i.num_total_blocks);
        bdevs_.erase (devId);
    }
    return NIXL_SUCCESS;
}

class nixlGusliMemReq : public nixlBackendMD { // Register/Unregister request
public:
    gusli::backend_bdev_id bdev; // Gusli bdev uuid
    uint64_t devId; // Nixl bdev uuid
    std::vector<gusli::io_buffer_t> ioBufs;
    // std::string metadata; // Just for future, currently unused
    nixl_mem_t mem_type;
    nixlGusliMemReq (const nixlBlobDesc &mem, nixl_mem_t _mem_type) : nixlBackendMD (true) {
        bdev.set_from (mem.devId);
        devId = mem.devId;
        // metadata = mem.metaInfo;
        mem_type = _mem_type;
    }
};

nixl_status_t
nixlGusliEngine::registerMem (const nixlBlobDesc &mem,
                              const nixl_mem_t &mem_type,
                              nixlBackendMD *&out) {
    out = nullptr;
    if ((mem_type != DRAM_SEG) && (mem_type != BLK_SEG))
        __LOG_RETERR (
            NIXL_ERR_NOT_SUPPORTED, "type not supported %d!=%d", (int)mem_type, (int)DRAM_SEG);
    std::unique_ptr<nixlGusliMemReq> md = std::make_unique<nixlGusliMemReq> (mem, mem_type);
    __LOG_DBG ("register dev[0x%lx].ram_lba[%p].len=0x%lx, mem_type=%u, md=%s",
               mem.devId,
               (void *)mem.addr,
               mem.len,
               mem_type,
               mem.metaInfo.c_str());
    md->ioBufs.emplace_back (gusli::io_buffer_t{.ptr = (void *)mem.addr, .byte_len = mem.len});
    if (mem_type == BLK_SEG) {
        // Todo: LBA of block devices, verify size, extend volume
    } else {
        const nixl_status_t openRv = bdevOpen (md->devId);
        if (openRv != NIXL_SUCCESS)
            return openRv;
        const gusli::connect_rv rv = lib_->bdev_bufs_register (md->bdev, md->ioBufs);
        if (rv != gusli::connect_rv::C_OK) {
            const nixl_status_t closeRv = bdevClose (md->devId);
            // Even if close fails, nothing todo with its error code
            __LOG_RETERR (conErrConv (rv),
                          "register buf rv=%d, closeRV=%d, [%p,0x%lx]",
                          (int)rv,
                          (int)closeRv,
                          (void *)mem.addr,
                          mem.len);
        }
    }
    out = (nixlBackendMD *)md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGusliEngine::deregisterMem (nixlBackendMD *_md) {
    nixlGusliMemReq *md = (nixlGusliMemReq *)_md;
    if (!md) __LOG_RETERR (NIXL_ERR_INVALID_PARAM, "md==null");
    std::unique_ptr<nixlGusliMemReq> auto_deleter =
        std::unique_ptr<nixlGusliMemReq> (md); // Regardless of the outcome: md should be deleted
    __LOG_DBG ("unregister dev[0x%lx].ram_lba[%p].len=0x%lx, mem_type=%u",
               md->devId,
               (void *)md->ioBufs[0].ptr,
               md->ioBufs[0].byte_len,
               md->mem_type);
    if (md->mem_type == BLK_SEG) {
        // Nothing to do
    } else {
        const gusli::connect_rv rv = lib_->bdev_bufs_unregist (md->bdev, md->ioBufs);
        if (rv != gusli::connect_rv::C_OK)
            __LOG_RETERR (conErrConv (rv),
                          "unregister buf rv=%d, [%p,0x%lx]",
                          (int)rv,
                          (void *)md->ioBufs[0].ptr,
                          md->ioBufs[0].byte_len);
        return bdevClose (md->devId);
    }
    return NIXL_SUCCESS;
}

/********************************** IO ***************************************/
#define __LOG_IO(o, fmt, ...) __LOG_TRC ("IO[%c%p]" fmt, (o)->op, (o), ##__VA_ARGS__)

class nixlGusliBackendReqHbase : public nixlBackendReqH {
public:
    enum gusli::io_error_codes
        pollableAsyncRV; // NIXL actively polls on rv instead of waiting for completion.
    enum gusli::io_type op; // USed for prints
    [[nodiscard]] virtual nixl_status_t
    exec (void) = 0;
    [[nodiscard]] virtual nixl_status_t
    pollStatus (void) = 0;
    nixlGusliBackendReqHbase (const nixl_xfer_op_t _op)
        : op ((_op == NIXL_WRITE) ? gusli::G_WRITE : gusli::G_READ) {
        __LOG_IO (this, "_prep");
    }
    virtual ~nixlGusliBackendReqHbase() {
        __LOG_IO (this, "_free");
    }

protected:
    [[nodiscard]] nixl_status_t
    getCompStatus (void) const {
        const enum gusli::io_error_codes rv = pollableAsyncRV;
        if (rv == gusli::io_error_codes::E_OK) return NIXL_SUCCESS;
        if (rv == gusli::io_error_codes::E_IN_TRANSFER) return NIXL_IN_PROG;
        if (rv == gusli::io_error_codes::E_INVAL_PARAMS) return NIXL_ERR_INVALID_PARAM;
        __LOG_RETERR (NIXL_ERR_BACKEND, "IO[%c%p], io exec error rv=%d", op, this, (int)rv);
    }
};

class nixlGusliBackendReqHSingleBdev : public nixlGusliBackendReqHbase {
    gusli::io_request io; // gusli executor of 1 io
    static void
    completionCallback (nixlGusliBackendReqHSingleBdev *c) {
        __LOG_IO (c, "_doneCB, rv=%d", c->io.get_error());
        c->pollableAsyncRV =
            c->io.get_error(); // Must be last line because once set, class can be destroyed
    }

public:
    nixlGusliBackendReqHSingleBdev (const nixl_xfer_op_t _op) : nixlGusliBackendReqHbase (_op) {
        io.params.set_completion (this, completionCallback);
        io.params.op = op;
    }
    ~nixlGusliBackendReqHSingleBdev() {
        (void)io.try_cancel(); // If io was completed - meaningless, otherwise if io is in air,
                               // cancel it so 'io' field can be free. dont care about return value
                               // because io will get free anyways
    }
    void
    set1Buf (int32_t gid, const nixlMetaDesc &local, const nixlMetaDesc &remote) {
        io.params.init_1_rng (
            op, gid, (uint64_t)remote.addr, (uint64_t)local.len, (void *)local.addr);
        __LOG_IO (this,
                  ".RNG1: dev=%d, %p, 0x%lx[b], lba=0x%lx, gid=%d",
                  remote.devId,
                  (void *)local.addr,
                  (uint64_t)local.len,
                  (uint64_t)remote.addr,
                  gid);
    }
    [[nodiscard]] nixl_status_t
    setBufs (int32_t gid, const nixl_meta_dlist_t &local, const nixl_meta_dlist_t &remote) {
        const int nRanges = remote.descCount();
        gusli::io_multi_map_t *mio =
            (gusli::io_multi_map_t *)local[0].addr; // Allocate scatter gather in the first entry
        mio->n_entries = (nRanges - 1); // First entry is the scatter gather
        if (mio->my_size() > local[0].len) {
            __LOG_ERR ("mmap of sg=0x%lx[b] > is too short=0x%lx[b], Enlarge mapping or use "
                       "shorter transfer list",
                       mio->my_size(),
                       local[0].len);
            return NIXL_ERR_INVALID_PARAM;
        }
        __LOG_IO (this,
                  ".SGL: dev=%d, %p, 0x%lx[b], lba=0x%lx, gid=%d",
                  remote[0].devId,
                  mio,
                  (uint64_t)local[0].len,
                  remote[0].addr,
                  gid);
        for (int i = 1; i < nRanges; i++) { // Skip first range
            mio->entries[i - 1] = (gusli::io_map_t){.data =
                                                        {
                                                            .ptr = (void *)local[i].addr,
                                                            .byte_len = (uint64_t)local[i].len,
                                                        },
                                                    .offset_lba_bytes = (uint64_t)remote[i].addr};
            __LOG_IO (this,
                      ".RNG: dev=%d, %p, 0x%lx[b], lba=0x%lx, idx=%u",
                      remote[i].devId,
                      (void *)local[i].addr,
                      (uint64_t)local[i].len,
                      remote[i].addr,
                      i);
        }
        if (false && (mio->n_entries > 64)) { // I did not measure this number to opimize it, Not sure why getting (Operation not permitted) in NIXL container
            io.params.try_using_uring_api = true; // More efficient for long range io's.
            io.params.set_async_pollable(); // Uring support is faster as polling without callback
            __LOG_IO (this, ".URING");
        }
        io.params.init_multi (op, gid, *mio);
        return NIXL_SUCCESS;
    }
    [[nodiscard]] nixl_status_t
    exec (void) override {
        pollableAsyncRV = gusli::io_error_codes::E_IN_TRANSFER;
        __LOG_IO (this,
                  "start, nRanges=%u, size=%lu[KB]",
                  io.params.num_ranges(),
                  ((long)io.params.buf_size() >> 10));
        io.submit_io();
        return NIXL_IN_PROG;
    }
    [[nodiscard]] nixl_status_t
    pollStatus (void) override {
        if (!io.has_callback()) // Callback will update pollable rv
            pollableAsyncRV = io.get_error();
        return getCompStatus();
    }
};
class nixlGusliBackendReqHCompound : public nixlGusliBackendReqHbase {
    std::vector<class nixlGusliBackendReqHSingleBdev>
        child; // Array of sub completions.
public:
    nixlGusliBackendReqHCompound (const nixl_xfer_op_t _op, unsigned nSubIOs)
        : nixlGusliBackendReqHbase (_op) {
        child.reserve (nSubIOs);
    }
    ~nixlGusliBackendReqHCompound() = default;      // Will cancle all child io
    void
    addSubIO (const nixl_xfer_op_t _op,
              int32_t gid,
              const nixlMetaDesc &local,
              const nixlMetaDesc &remote) {
        auto &sub = child.emplace_back (_op);
        sub.set1Buf (gid, local, remote);
    }
    [[nodiscard]] nixl_status_t
    exec (void) override {
        pollableAsyncRV = gusli::io_error_codes::E_IN_TRANSFER;
        __LOG_IO (this, "start, nSubIOs=%zu", child.size());
        for (auto &sub : child)
            (void)sub.exec(); // We know that return value is in progress
        return NIXL_IN_PROG;
    }
    [[nodiscard]] nixl_status_t
    pollStatus (void) override {
        if (pollableAsyncRV != gusli::io_error_codes::E_IN_TRANSFER)
            return getCompStatus(); // All sub ios returned and already updated this compound op
        for (auto &sub : child) {
            if (sub.pollStatus() == NIXL_IN_PROG)
                return NIXL_IN_PROG; // At least 1 sub-io is in air, still wait
        }
        for (auto &sub : child) { // All sub-ios completed find out if at least 1 failed
            if (sub.pollStatus() != NIXL_SUCCESS) {
                __LOG_IO (this,
                          "_done_all_sub, inherit_sub_io[%zu].rv=%d",
                          (&sub - &child[0]),
                          sub.pollableAsyncRV);
                pollableAsyncRV = sub.pollableAsyncRV; // Propagate error up the tree
                return getCompStatus(); // Dont care about success/failure of the rest of children
            }
        }
        __LOG_IO (this, "_done_all_sub, success");
        pollableAsyncRV = gusli::io_error_codes::E_OK;
        return getCompStatus();
    }
};

nixl_status_t
nixlGusliEngine::prepXfer (const nixl_xfer_op_t &op,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    handle = nullptr;
    // Verify params
    // if (strcmp(remote_agent.c_str(), gusli_params.client_name))
    // __LOG_RETERR(NIXL_ERR_INVALID_PARAM, "Remote(%s) != localAgent(%s)", remote_agent.c_str(),
    // gusli_params.client_name);
    if (local.getType() != DRAM_SEG)
        __LOG_RETERR (
            NIXL_ERR_INVALID_PARAM, "Local memory type must be DRAM_SEG, got %d", local.getType());
    if (remote.getType() != BLK_SEG)
        __LOG_RETERR (
            NIXL_ERR_INVALID_PARAM, "Remote memory type must be BLK_SEG, got %d", remote.getType());
    if (local.descCount() != remote.descCount())
        __LOG_RETERR (NIXL_ERR_INVALID_PARAM,
                      "Mismatch in descriptor counts - local[%d] != remote[%d]",
                      local.descCount(),
                      remote.descCount());

    const int32_t gid = getGidOfBDev (remote[0].devId); // First bdev for IO
    const unsigned nRanges = remote.descCount();
    const bool is_single_range_io = (nRanges == 1);
    const bool has_sgl_mem =
        (opt_args && (opt_args->customParam.find ("-sgl") != std::string::npos));
    const bool entire_io_1_bdev = isEntireIOto1Bdev (remote);
    const bool can_use_multi_range_optimization = (entire_io_1_bdev && has_sgl_mem);
    if (is_single_range_io) {
        std::unique_ptr<nixlGusliBackendReqHSingleBdev> req =
            std::make_unique<nixlGusliBackendReqHSingleBdev> (op);
        req->set1Buf (gid, local[0], remote[0]);
        handle = (nixlBackendReqH *)req.release();
    } else if (can_use_multi_range_optimization) {
        std::unique_ptr<nixlGusliBackendReqHSingleBdev> req =
            std::make_unique<nixlGusliBackendReqHSingleBdev> (op);
        const nixl_status_t rv = req->setBufs (gid, local, remote);
        if (rv != NIXL_SUCCESS)
            __LOG_RETERR (rv, "missing SGL, or SGL too small 0x%lx[b]", local[0].len);
        handle = (nixlBackendReqH *)req.release();
    } else {
        std::unique_ptr<nixlGusliBackendReqHCompound> req =
            std::make_unique<nixlGusliBackendReqHCompound> (op, nRanges);
        unsigned i = (has_sgl_mem ? 1 : 0); // If supplied sgl, can't use it for now, just ignore it
        __LOG_IO (req.get(),
                  "_Compound IO, 1-bdev=%d, has_sgl=%d, nSubIOs=%u",
                  entire_io_1_bdev,
                  has_sgl_mem,
                  (nRanges - i));
        for (; i < nRanges; i++)
            req->addSubIO (op, getGidOfBDev (remote[i].devId), local[i], remote[i]);
        handle = (nixlBackendReqH *)req.release();
    }
    __LOG_IO ((nixlGusliBackendReqHbase *)handle,
              "HDR: 1-gio=%d, 1-bdev=%d, has_sgl=%d, vec_size=%u, opt=%p cust=%s",
              (is_single_range_io || can_use_multi_range_optimization),
              entire_io_1_bdev,
              has_sgl_mem,
              nRanges,
              opt_args,
              opt_args->customParam.c_str());
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGusliEngine::postXfer (const nixl_xfer_op_t &operation,
                           const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote,
                           const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args) const {
    (void)operation;
    (void)local;
    (void)remote;
    (void)remote_agent;
    (void)opt_args;
    nixlGusliBackendReqHbase *req = (nixlGusliBackendReqHbase *)handle;
    return req->exec();
}

nixl_status_t
nixlGusliEngine::checkXfer (nixlBackendReqH *handle) const {
    nixlGusliBackendReqHbase *req = (nixlGusliBackendReqHbase *)handle;
    return req->pollStatus();
}

nixl_status_t
nixlGusliEngine::releaseReqH (nixlBackendReqH *handle) const {
    delete ((nixlGusliBackendReqHbase *)handle);
    return NIXL_SUCCESS;
}
