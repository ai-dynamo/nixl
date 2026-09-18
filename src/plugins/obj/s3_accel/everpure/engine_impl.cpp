/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "client.h"
#include "rdma_interface.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <atomic>
#include <memory>
#include <vector>
#include <algorithm>

#include "obj_engine_registry.h"

#if defined(HAVE_CUOBJ_CLIENT)

namespace {

objAccelEngineRegistrar reg_everpure(
    "everpure",
    [](const nixlBackendInitParams *p) { return std::make_unique<S3EverpureObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3EverpureObjEngineImpl>(p, std::move(s3));
    });

// RDMA GET responses arrive with an empty HTTP body and no checksum
// headers, so the SDK's response-checksum validation has nothing to check
// against. Default resp_checksum to required, unless the caller already set it.
//
// RDMA PUT requests also carry an empty HTTP body, so a request checksum
// computed over it would never match the real object. Default req_checksum
// to its more conservative setting (WHEN_REQUIRED), unless already set.
void
applyEverpureDefaults(const nixlBackendInitParams *init_params,
                      nixl_b_params_t *&params_to_use,
                      nixl_b_params_t &owned_fallback) {
    if (init_params->customParams) {
        init_params->customParams->emplace("resp_checksum", "required");
        init_params->customParams->emplace("req_checksum", "required");
        params_to_use = init_params->customParams;
        return;
    }
    owned_fallback["resp_checksum"] = "required";
    owned_fallback["req_checksum"] = "required";
    params_to_use = &owned_fallback;
}

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Invalid operation type: %d", operation);
        return false;
    }
    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat("Remote agent (%s) does not match requesting agent (%s)",
                                     remote_agent,
                                     local_agent);
    }
    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat("Local memory must be DRAM_SEG or VRAM_SEG, got %d",
                                      local.getType());
        return false;
    }
    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Remote memory must be OBJ_SEG, got %d", remote.getType());
        return false;
    }
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat("Descriptor count mismatch: %d local vs %d remote",
                                      local.descCount(),
                                      remote.descCount());
        return false;
    }
    return true;
}

/// Backend metadata for a registered object key (OBJ_SEG side).
class nixlEverpureObjKeyMD : public nixlBackendMD {
public:
    explicit nixlEverpureObjKeyMD(std::string obj_key)
        : nixlBackendMD(true), objKey(std::move(obj_key)) {}

    std::string objKey;
};

/// Backend metadata for a registered local memory region (DRAM_SEG/VRAM_SEG side).
class nixlEverpureMemRegMD : public nixlBackendMD {
public:
    explicit nixlEverpureMemRegMD(uintptr_t addr) : nixlBackendMD(true), addr(addr) {}

    uintptr_t addr;
};

/// Everything one RDMA transfer needs to hand off to the S3 client, plus the
/// token to release via cuMemObjPutRDMAToken once the handle is released.
struct everpureTransferUnit {
    uintptr_t addr;
    size_t size;
    size_t offset;
    std::string objKey;
    std::string rdmaDesc;
    char *rdmaToken = nullptr;

    everpureTransferUnit(uintptr_t addr_, size_t size_, size_t offset_, std::string obj_key)
        : addr(addr_),
          size(size_),
          offset(offset_),
          objKey(std::move(obj_key)) {}
};

/**
 * Request handle for a batch of RDMA transfers.
 *
 * Completion is tracked with a single outstanding-count and a sticky
 * failure flag, both updated directly from the S3 client's async
 * callbacks, so checkXfer() is a plain load rather than a walk over
 * per-unit state.
 */
class nixlEverpureObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<everpureTransferUnit> units;
    std::atomic<size_t> outstanding{0};
    std::atomic<bool> anyFailed{false};

    void
    armCompletionTracking() {
        outstanding.store(units.size(), std::memory_order_release);
    }

    void
    onUnitDone(bool success) {
        if (!success) {
            anyFailed.store(true, std::memory_order_relaxed);
        }
        outstanding.fetch_sub(1, std::memory_order_acq_rel);
    }

    nixl_status_t
    poll() const {
        if (outstanding.load(std::memory_order_acquire) != 0) {
            return NIXL_IN_PROG;
        }
        return anyFailed.load(std::memory_order_relaxed) ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
    }
};

// cuObjClient's constructor requires an IO callback table; every transfer
// resolves its descriptor via cuMemObjGetRDMAToken instead, so these never run.
ssize_t
unusedGetCallback(const void *, char *, size_t, loff_t, const cufileRDMAInfo_t *) {
    return -EINVAL;
}

ssize_t
unusedPutCallback(const void *, const char *, size_t, loff_t, const cufileRDMAInfo_t *) {
    return -EINVAL;
}

// Dynamically Connected verbs transport for the cuObject client.
CUObjIOOps everpureIoOps = {.get = unusedGetCallback, .put = unusedPutCallback};

} // namespace

S3EverpureObjEngineImpl::S3EverpureObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params) {
    nixl_b_params_t *params_to_use = nullptr;
    nixl_b_params_t owned_fallback;
    applyEverpureDefaults(init_params, params_to_use, owned_fallback);

    s3Client_ = std::make_shared<awsS3EverpureClient>(params_to_use, executor_);
    cuClient_ = std::make_shared<cuObjClient>(everpureIoOps, CUOBJ_PROTO_RDMA_DC_V1);
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "S3EverpureObjEngineImpl: cuObjClient failed to connect";
        return;
    }
    NIXL_INFO << "S3EverpureObjEngineImpl: ready (S3-RDMA)";
}

S3EverpureObjEngineImpl::S3EverpureObjEngineImpl(const nixlBackendInitParams *init_params,
                                                 std::shared_ptr<iS3Client> s3_client)
    : S3AccelObjEngineImpl(init_params, s3_client) {
    if (s3_client) {
        s3Client_ = std::move(s3_client);
    } else {
        nixl_b_params_t *params_to_use = nullptr;
        nixl_b_params_t owned_fallback;
        applyEverpureDefaults(init_params, params_to_use, owned_fallback);
        s3Client_ = std::make_shared<awsS3EverpureClient>(params_to_use, executor_);
    }

    cuClient_ = std::make_shared<cuObjClient>(everpureIoOps, CUOBJ_PROTO_RDMA_DC_V1);
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "S3EverpureObjEngineImpl: cuObjClient failed to connect";
        return;
    }
    NIXL_INFO << "S3EverpureObjEngineImpl: ready (S3-RDMA, injected S3 client)";
}

nixl_status_t
S3EverpureObjEngineImpl::requireCuObjReady() const {
    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "S3EverpureObjEngineImpl: cuObjClient is not connected";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
S3EverpureObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                     const nixl_mem_t &nixl_mem,
                                     nixlBackendMD *&out) {
    if (auto status = requireCuObjReady(); status != NIXL_SUCCESS) {
        return status;
    }

    switch (nixl_mem) {
    case OBJ_SEG: {
        std::string obj_key = mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo;
        out = new nixlEverpureObjKeyMD(std::move(obj_key));
        return NIXL_SUCCESS;
    }
    case DRAM_SEG:
    case VRAM_SEG: {
        if (mem.len > CUOBJ_MAX_MEMORY_REG_SIZE) {
            NIXL_ERROR << "registerMem: region of " << mem.len
                       << " bytes exceeds CUOBJ_MAX_MEMORY_REG_SIZE";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        cuObjErr_t status = cuClient_->cuMemObjGetDescriptor(
            reinterpret_cast<void *>(mem.addr), mem.len);
        if (status != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "registerMem: cuMemObjGetDescriptor failed, status=" << status;
            return NIXL_ERR_BACKEND;
        }
        out = new nixlEverpureMemRegMD(mem.addr);
        return NIXL_SUCCESS;
    }
    default:
        return NIXL_ERR_NOT_SUPPORTED;
    }
}

nixl_status_t
S3EverpureObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    if (auto *obj_md = dynamic_cast<nixlEverpureObjKeyMD *>(meta)) {
        delete obj_md;
        return NIXL_SUCCESS;
    }

    if (auto *mem_md = dynamic_cast<nixlEverpureMemRegMD *>(meta)) {
        cuObjErr_t status = cuClient_->cuMemObjPutDescriptor(reinterpret_cast<void *>(mem_md->addr));
        if (status != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "deregisterMem: cuMemObjPutDescriptor failed, status=" << status;
            // Leave *mem_md alive so the caller can retry the deregistration.
            return NIXL_ERR_BACKEND;
        }
        delete mem_md;
        return NIXL_SUCCESS;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
S3EverpureObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  const std::string &local_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    if (auto status = requireCuObjReady(); status != NIXL_SUCCESS) {
        return status;
    }
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlEverpureObjBackendReqH>();
    req_h->units.reserve(local.descCount());

    const cuObjOpType_t cu_op = (operation == NIXL_WRITE) ? CUOBJ_PUT : CUOBJ_GET;

    for (int i = 0; i < local.descCount(); ++i) {
        auto *obj_md = dynamic_cast<nixlEverpureObjKeyMD *>(remote[i].metadataP);
        if (!obj_md) {
            NIXL_ERROR << "prepXfer: object segment devId " << remote[i].devId
                       << " was never registered";
            return NIXL_ERR_INVALID_PARAM;
        }

        auto *mem_md = dynamic_cast<nixlEverpureMemRegMD *>(local[i].metadataP);
        if (!mem_md) {
            NIXL_ERROR << "prepXfer: local segment devId " << local[i].devId
                       << " was never registered";
            return NIXL_ERR_INVALID_PARAM;
        }

        everpureTransferUnit unit(local[i].addr, local[i].len, remote[i].addr, obj_md->objKey);

        // cuMemObjGetRDMAToken requires the exact base pointer that was passed
        // to cuMemObjGetDescriptor at registration time (the whole scratch
        // pool), plus the byte offset of this transfer's region within it --
        // it does not accept an already-offset pointer with buffer_offset=0.
        char *desc_str = nullptr;
        const size_t buffer_offset = static_cast<size_t>(unit.addr - mem_md->addr);
        cuObjErr_t status = cuClient_->cuMemObjGetRDMAToken(
            reinterpret_cast<void *>(mem_md->addr), unit.size, buffer_offset, cu_op, &desc_str);
        if (status != CU_OBJ_SUCCESS || desc_str == nullptr) {
            NIXL_ERROR << "prepXfer: cuMemObjGetRDMAToken failed, status=" << status;
            for (const auto &acquired : req_h->units) {
                cuClient_->cuMemObjPutRDMAToken(acquired.rdmaToken);
            }
            return NIXL_ERR_BACKEND;
        }
        unit.rdmaDesc = desc_str;
        unit.rdmaToken = desc_str;

        req_h->units.push_back(std::move(unit));
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
S3EverpureObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "postXfer: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req_h = static_cast<nixlEverpureObjBackendReqH *>(handle);

    auto *rdma_client = dynamic_cast<iEverpureS3RdmaClient *>(s3Client_.get());
    if (!rdma_client) {
        NIXL_ERROR << "postXfer: S3 client does not implement iEverpureS3RdmaClient";
        return NIXL_ERR_BACKEND;
    }

    req_h->armCompletionTracking();
    for (const auto &unit : req_h->units) {
        auto on_done = [req_h](bool success) { req_h->onUnitDone(success); };
        if (operation == NIXL_WRITE) {
            rdma_client->putObjectRdmaAsync(
                unit.objKey, unit.addr, unit.size, unit.offset, unit.rdmaDesc, on_done);
        } else {
            rdma_client->getObjectRdmaAsync(
                unit.objKey, unit.addr, unit.size, unit.offset, unit.rdmaDesc, on_done);
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
S3EverpureObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "checkXfer: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    return static_cast<nixlEverpureObjBackendReqH *>(handle)->poll();
}

nixl_status_t
S3EverpureObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "releaseReqH: transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *req_h = static_cast<nixlEverpureObjBackendReqH *>(handle);
    for (const auto &unit : req_h->units) {
        cuClient_->cuMemObjPutRDMAToken(unit.rdmaToken);
    }
    delete req_h;
    return NIXL_SUCCESS;
}

iS3Client *
S3EverpureObjEngineImpl::getClient() const {
    return s3Client_.get();
}

#endif // HAVE_CUOBJ_CLIENT
