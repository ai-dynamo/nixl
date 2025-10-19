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
    do {                                                                                          \
        NIXL_ERROR << absl::StrFormat(                                                            \
            "GUSLI: %s() %s[%d]" format, __PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)
#define __LOG_DBG(format, ...)                                          \
    do {                                                                \
        NIXL_DEBUG << absl::StrFormat("GUSLI: " format, ##__VA_ARGS__); \
    } while (0)
#define __LOG_TRC(format, ...)                                          \
    do {                                                                \
        NIXL_TRACE << absl::StrFormat("GUSLI: " format, ##__VA_ARGS__); \
    } while (0)
#define __LOG_RETERR(rv, format, ...)                              \
    do {                                                           \
        __LOG_ERR("nixl_err=%d, " format, (int)rv, ##__VA_ARGS__); \
        return rv;                                                 \
    } while (0)

namespace {
[[nodiscard]] nixl_status_t
conErrConv(const gusli::connect_rv rv) {
    switch (rv) {
    case gusli::connect_rv::C_OK:
        return NIXL_SUCCESS;
    case gusli::connect_rv::C_NO_DEVICE:
        return NIXL_ERR_NOT_FOUND;
    case gusli::connect_rv::C_WRONG_ARGUMENTS:
        return NIXL_ERR_INVALID_PARAM;
    default:
        return NIXL_ERR_BACKEND;
    }
}

[[nodiscard]] bool
isEntireIOto1Bdev(const nixl_meta_dlist_t &remote) {
    const uint64_t devId = remote[0].devId;
    const unsigned num_ranges = remote.descCount();
    for (unsigned i = 1; i < num_ranges; i++)
        if (devId != remote[i].devId) return false;
    return true;
}

class nixlGusliMemReq : public nixlBackendMD { // Register/Unregister request
public:
    gusli::backend_bdev_id bdev; // Gusli bdev uuid
    uint64_t devId; // Nixl bdev uuid
    std::vector<gusli::io_buffer_t> ioBufs;
    nixl_mem_t memType;

    nixlGusliMemReq(const nixlBlobDesc &mem, nixl_mem_t mem_type) : nixlBackendMD(true) {
        bdev.set_from(mem.devId);
        devId = mem.devId;
        memType = mem_type;
    }
};

nixl_status_t
verifyRequestParams(const nixl_xfer_op_t &op,
                    const nixl_meta_dlist_t &local,
                    const nixl_meta_dlist_t &remote) {
    if (local.getType() != DRAM_SEG)
        __LOG_RETERR(
            NIXL_ERR_INVALID_PARAM, "Local memory type must be DRAM_SEG, got %d", local.getType());
    if (remote.getType() != BLK_SEG)
        __LOG_RETERR(
            NIXL_ERR_INVALID_PARAM, "Remote memory type must be BLK_SEG, got %d", remote.getType());
    if (local.descCount() != remote.descCount())
        __LOG_RETERR(NIXL_ERR_INVALID_PARAM,
                     "Mismatch in descriptor counts - local[%d] != remote[%d]",
                     local.descCount(),
                     remote.descCount());
    return NIXL_SUCCESS;
}
}; // namespace

void
nixlGusliEngine::parseInitParams(const nixlBackendInitParams *nixl_init,
                                 gusli::global_clnt_context::init_params &gusli_params) {
    // Convert nixl params to lib params
    gusli_params.log =
        stdout; // Redirect gusli logs to stdout, important errors will be printed by the plugin
    if (nixl_init && nixl_init->customParams) {
        const nixl_b_params_t *backParams = nixl_init->customParams;
        if (backParams->count("client_name") > 0)
            gusli_params.client_name = backParams->at("client_name").c_str();
        if (backParams->count("max_num_simultaneous_requests") > 0)
            gusli_params.max_num_simultaneous_requests =
                std::stoi(backParams->at("max_num_simultaneous_requests"));
        if (backParams->count("config_file") > 0)
            gusli_params.config_file = backParams->at("config_file").c_str();
    }
}

nixlGusliEngine::nixlGusliEngine(const nixlBackendInitParams *nixl_init)
    : nixlBackendEngine(nixl_init) {
    gusli::global_clnt_context::init_params gusli_params;
    parseInitParams(nixl_init, gusli_params);
    lib_ = std::make_unique<gusli::global_clnt_context>(gusli_params);
    NIXL_ASSERT_ALWAYS(lib_->BREAKING_VERSION == 1);
}

nixl_status_t
nixlGusliEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &mem_type,
                             nixlBackendMD *&out) {
    out = nullptr;
    if ((mem_type != DRAM_SEG) && (mem_type != BLK_SEG)) return NIXL_ERR_NOT_SUPPORTED;

    std::unique_ptr<nixlGusliMemReq> md = std::make_unique<nixlGusliMemReq>(mem, mem_type);

    __LOG_DBG("register dev[0x%lx].ram_lba[%p].len=0x%lx, type=%u, md=%s",
              mem.devId,
              (void *)mem.addr,
              mem.len,
              mem_type,
              mem.metaInfo.c_str());

    md->ioBufs.emplace_back(gusli::io_buffer_t{.ptr = (void *)mem.addr, .byte_len = mem.len});

    if (mem_type == BLK_SEG) {
        // Todo: LBA of block devices, verify size, extend volume
    } else {
        const gusli::connect_rv rv = lib_->open__bufs_register(md->bdev, md->ioBufs);
        if (rv != gusli::connect_rv::C_OK) {
            __LOG_RETERR(conErrConv(rv),
                         "register buf rv=%d, [%p,0x%lx]",
                         (int)rv,
                         (void *)mem.addr,
                         mem.len);
        }
    }

    out = (nixlBackendMD *)md.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGusliEngine::deregisterMem(nixlBackendMD *_md) {
    if (!_md) __LOG_RETERR(NIXL_ERR_INVALID_PARAM, "md==null");

    // Take ownership of the memory descriptor for safe cleanup
    std::unique_ptr<nixlGusliMemReq> md(static_cast<nixlGusliMemReq *>(_md));

    __LOG_DBG("unregister dev[0x%lx].ram_lba[%p].len=0x%lx, type=%u",
              md->devId,
              (void *)md->ioBufs[0].ptr,
              md->ioBufs[0].byte_len,
              md->memType);
    if (md->memType == BLK_SEG) {
        // Nothing to do
    } else {
        const gusli::connect_rv rv = lib_->close_bufs_unregist(md->bdev, md->ioBufs);
        if (rv != gusli::connect_rv::C_OK)
            __LOG_RETERR(conErrConv(rv),
                         "unregister buf rv=%d, [%p,0x%lx]",
                         (int)rv,
                         (void *)md->ioBufs[0].ptr,
                         md->ioBufs[0].byte_len);
    }
    return NIXL_SUCCESS;
}

/********************************** IO ***************************************/
#define __LOG_IO(o, fmt, ...) __LOG_TRC("IO[%c%p]" fmt, (o)->op, (o), ##__VA_ARGS__)

class nixlGusliBackendReqHbase : public nixlBackendReqH {
public:
    enum gusli::io_error_codes
        pollableAsyncRV; // NIXL actively polls on rv instead of waiting for completion.
    enum gusli::io_type op; // USed for prints
    [[nodiscard]] virtual nixl_status_t
    exec(void) = 0;
    [[nodiscard]] virtual nixl_status_t
    pollStatus(void) = 0;

    nixlGusliBackendReqHbase(const nixl_xfer_op_t _op)
        : op((_op == NIXL_WRITE) ? gusli::G_WRITE : gusli::G_READ) {
        __LOG_IO(this, "_prep");
    }

    virtual ~nixlGusliBackendReqHbase() {
        __LOG_IO(this, "_free");
    }

protected:
    [[nodiscard]] nixl_status_t
    getCompStatus(void) const {
        const enum gusli::io_error_codes rv = pollableAsyncRV;
        switch (rv) {
        case gusli::io_error_codes::E_OK:
            return NIXL_SUCCESS;
        case gusli::io_error_codes::E_IN_TRANSFER:
            return NIXL_IN_PROG;
        case gusli::io_error_codes::E_INVAL_PARAMS:
            return NIXL_ERR_INVALID_PARAM;
        case gusli::io_error_codes::E_THROTTLE_RETRY_LATER:
            return NIXL_ERR_NOT_ALLOWED;
        default:
            __LOG_RETERR(NIXL_ERR_BACKEND, "IO[%c%p], io exec error rv=%d", op, this, (int)rv);
        }
    }
};

class nixlGusliBackendReqHSingleBdev : public nixlGusliBackendReqHbase {
public:
    nixlGusliBackendReqHSingleBdev(const nixl_xfer_op_t nixlOp,
                                   int32_t gid,
                                   const nixlMetaDesc &local,
                                   const nixlMetaDesc &remote)
        : nixlGusliBackendReqHbase(nixlOp) {
        initCommon();

        io.params.init_1_rng(
            op, gid, (uint64_t)remote.addr, (uint64_t)local.len, (void *)local.addr);

        __LOG_IO(this,
                 ".RNG1: dev=%d, %p, 0x%zx[b], lba=0x%lx, gid=%d",
                 remote.devId,
                 (void *)local.addr,
                 local.len,
                 remote.addr,
                 gid);
    }

    nixlGusliBackendReqHSingleBdev(const nixl_xfer_op_t nixl_op,
                                   int32_t gid,
                                   const nixl_meta_dlist_t &local,
                                   const nixl_meta_dlist_t &remote)
        : nixlGusliBackendReqHbase(nixl_op) {
        initCommon();
        const int num_ranges = remote.descCount();
        gusli::io_multi_map_t *mio =
            (gusli::io_multi_map_t *)local[0].addr; // Allocate scatter gather in the first entry

        mio->init_num_entries(num_ranges - 1); // First entry is the scatter gather
        if (mio->my_size() > local[0].len) {
            __LOG_ERR("mmap of sg=0x%lx[b] > is too short=0x%lx[b], Enlarge mapping or use "
                      "shorter transfer list",
                      mio->my_size(),
                      local[0].len);
            throw std::runtime_error("Failed to initialize io, See log");
        }

        __LOG_IO(this,
                 ".SGL: dev=%d, %p, 0x%zx[b], lba=0x%lx, gid=%d",
                 remote[0].devId,
                 mio,
                 local[0].len,
                 remote[0].addr,
                 gid);

        for (int i = 1; i < num_ranges; i++) { // Skip first range
            mio->entries[i - 1].init((void *)local[i].addr, local[i].len, remote[i].addr);
            __LOG_IO(this,
                     ".RNG: dev=%d, %p, 0x%zx[b], lba=0x%lx, idx=%u",
                     remote[i].devId,
                     (void *)local[i].addr,
                     local[i].len,
                     remote[i].addr,
                     i);
        }

        io.params.init_multi(op, gid, *mio);
    }

    ~nixlGusliBackendReqHSingleBdev() override {
        io.done();
        // If io was completed - meaningless, otherwise if io is in air, Gusli will auto cancel it
    }

    [[nodiscard]] nixl_status_t
    exec(void) override {
        pollableAsyncRV = gusli::io_error_codes::E_IN_TRANSFER;
        __LOG_IO(this,
                 "start, #ranges=%u, size=%lu[KB]",
                 io.params.num_ranges(),
                 ((long)io.params.buf_size() >> 10));
        io.submit_io();
        return NIXL_IN_PROG;
    }

    [[nodiscard]] nixl_status_t
    pollStatus(void) override {
        pollableAsyncRV = io.get_error();
        return getCompStatus();
    }

private:
    gusli::io_request io; // gusli executor of 1 io

    void
    initCommon(void) {
        io.params.set(op).set_priority(100).set_async_pollable();
    }
};

class nixlGusliBackendReqHCompound : public nixlGusliBackendReqHbase {
public:
    nixlGusliBackendReqHCompound(const nixl_xfer_op_t nixl_op,
                                 unsigned nSubIOs,
                                 bool hasSglMem,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 std::function<int32_t(uint64_t)> convertIdFunc)
        : nixlGusliBackendReqHbase(nixl_op) {
        child.reserve(nSubIOs);
        const unsigned num_ranges = remote.descCount();
        unsigned i = (hasSglMem ? 1 : 0); // If supplied sgl, can't use it for now, just ignore it
        __LOG_IO(this, "_Compound IO, has_sgl=%d, nSubIOs=%u", hasSglMem, (num_ranges - i));
        for (; i < num_ranges; i++)
            child.emplace_back(nixl_op, convertIdFunc(remote[i].devId), local[i], remote[i]);
    }

    ~nixlGusliBackendReqHCompound() override = default;

    [[nodiscard]] nixl_status_t
    exec(void) override {
        pollableAsyncRV = gusli::io_error_codes::E_IN_TRANSFER;
        __LOG_IO(this, "start, nSubIOs=%zu", child.size());
        for (auto &sub : child)
            (void)sub.exec(); // We know that return value is in progress
        return NIXL_IN_PROG;
    }

    [[nodiscard]] nixl_status_t
    pollStatus(void) override {
        if (pollableAsyncRV != gusli::io_error_codes::E_IN_TRANSFER)
            return getCompStatus(); // All sub ios returned and already updated this compound op
        for (auto &sub : child) {
            if (sub.pollStatus() == NIXL_IN_PROG)
                return NIXL_IN_PROG; // At least 1 sub-io is in air, still wait
        }
        for (auto &sub : child) { // All sub-ios completed find out if at least 1 failed
            if (sub.pollStatus() != NIXL_SUCCESS) {
                __LOG_IO(this,
                         "_done_all_sub, inherit_sub_io[%zu].rv=%d",
                         (&sub - &child[0]),
                         sub.pollableAsyncRV);
                pollableAsyncRV = sub.pollableAsyncRV; // Propagate error up the tree
                return getCompStatus(); // Dont care about success/failure of the rest of children
            }
        }
        __LOG_IO(this, "_done_all_sub, success");
        pollableAsyncRV = gusli::io_error_codes::E_OK;
        return getCompStatus();
    }

private:
    std::vector<nixlGusliBackendReqHSingleBdev> child;
};

nixl_status_t
nixlGusliEngine::prepXfer(const nixl_xfer_op_t &op,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    handle = nullptr;
    nixl_status_t verifyRv = verifyRequestParams(op, local, remote);
    if (verifyRv != NIXL_SUCCESS) return verifyRv;

    const int32_t gid = getGidOfBDev(remote[0].devId); // First bdev for IO
    const unsigned num_ranges = remote.descCount();
    const bool is_single_range_io = (num_ranges == 1);
    const bool has_sgl_mem =
        (opt_args && (opt_args->customParam.find("-sgl") != std::string::npos));
    const bool entire_io_1_bdev = isEntireIOto1Bdev(remote);
    const bool can_use_multi_range_optimization = (entire_io_1_bdev && has_sgl_mem);
    std::unique_ptr<nixlGusliBackendReqHbase> req;
    try {
        if (is_single_range_io) {
            req = std::make_unique<nixlGusliBackendReqHSingleBdev>(op, gid, local[0], remote[0]);
        } else if (can_use_multi_range_optimization) {
            req = std::make_unique<nixlGusliBackendReqHSingleBdev>(op, gid, local, remote);
        } else {
            req = std::make_unique<nixlGusliBackendReqHCompound>(
                op, num_ranges, has_sgl_mem, local, remote, [this](uint64_t devId) {
                    return this->getGidOfBDev(devId);
                });
        }
        handle = (nixlBackendReqH *)req.release();
    }
    catch (const std::exception &e) {
        __LOG_RETERR(NIXL_ERR_INVALID_PARAM,
                     "missing SGL, or SGL too small 0x%lx[b], info=%s",
                     local[0].len,
                     e.what());
    }
    __LOG_IO((nixlGusliBackendReqHbase *)handle,
             "HDR: 1-gio=%d, 1-bdev=%d, has_sgl=%d, vec_size=%u, opt=%p cust=%s",
             (is_single_range_io || can_use_multi_range_optimization),
             entire_io_1_bdev,
             has_sgl_mem,
             num_ranges,
             opt_args,
             opt_args->customParam.c_str());
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGusliEngine::postXfer(const nixl_xfer_op_t &operation,
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
nixlGusliEngine::checkXfer(nixlBackendReqH *handle) const {
    nixlGusliBackendReqHbase *req = (nixlGusliBackendReqHbase *)handle;
    return req->pollStatus();
}

nixl_status_t
nixlGusliEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete ((nixlGusliBackendReqHbase *)handle);
    return NIXL_SUCCESS;
}
