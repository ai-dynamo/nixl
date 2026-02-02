/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <memory>
#include <future>
#include <optional>
#include <vector>
#include <chrono>
#include <algorithm>


namespace {

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent)
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);

    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat("Error: Local memory type must be VRAM_SEG or DRAM_SEG, got %d",
                                      local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    return true;
}

class obsObjTransferRequestH {
    public:
        uintptr_t addr;
        size_t size;
        size_t offset;
        std::string rdma_desc;
        std::string obj_key;

        obsObjTransferRequestH() {
            addr = 0;
            size = 0;
            offset = 0;
            rdma_desc = "";
            obj_key = "";
        }
        obsObjTransferRequestH(uintptr_t a, size_t s, size_t offset) {
            addr = a;
            size = s;
            this->offset = offset;
        }
        ~obsObjTransferRequestH() = default;
};

class nixlObsObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<obsObjTransferRequestH> reqs_;
    nixlObsObjBackendReqH() = default;
    ~nixlObsObjBackendReqH() = default;

    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixl_status_t
    getOverallStatus() {
        while (!statusFutures_.empty()) {
            if (statusFutures_.back().wait_for(std::chrono::seconds(0)) ==
                std::future_status::ready) {
                auto current_status = statusFutures_.back().get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                statusFutures_.pop_back();
            } else {
                return NIXL_IN_PROG;
            }
        }
        return NIXL_SUCCESS;
    }
};

class nixlObsObjMetadata : public nixlBackendMD {
public:
    nixlObsObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key) {}
    nixlObsObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          localAddr(addr) {}
    ~nixlObsObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

typedef struct rdma_ctx {
    std::string rdma_desc;
} rdma_ctx_t;

static ssize_t objectGet(const void *handle, char* buf, size_t size, loff_t offset, const cufileRDMAInfo_t *infop) {
    void *ctx = cuObjClient::getCtx(handle);
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

static ssize_t objectPut(const void *handle, const char* buf, size_t size, loff_t offset, const cufileRDMAInfo_t *infop) {
    void *ctx = cuObjClient::getCtx(handle);
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

CUObjIOOps obs_ops = {
    .get  = objectGet,
    .put  = objectPut
};
}



S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params) {
    s3Client_ = std::make_shared<awsS3DellObsClient>(init_params->customParams, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Dell ObjectScale client";

    cuClient_ = std::make_shared<cuObjClient>(obs_ops, CUOBJ_PROTO_RDMA_DC_V1);
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iS3Client> s3_client,
                                           std::shared_ptr<iS3Client> s3_client_accel)
    : S3AccelObjEngineImpl(init_params, s3_client) {
    s3Client_ = std::make_shared<awsS3DellObsClient>(init_params->customParams, executor_);

    cuClient_ = std::make_shared<cuObjClient>(obs_ops, CUOBJ_PROTO_RDMA_DC_V1);
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
}

nixl_status_t
S3DellObsObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                           const nixl_mem_t &nixl_mem,
                           nixlBackendMD *&out) {
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end())
        return NIXL_ERR_NOT_SUPPORTED;

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlObsObjMetadata> obj_md = std::make_unique<nixlObsObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else if ((nixl_mem == DRAM_SEG) || (nixl_mem == VRAM_SEG)) {
        std::unique_ptr<nixlObsObjMetadata> mem_md = std::make_unique<nixlObsObjMetadata>(nixl_mem, mem.addr);
        cuClient_->cuMemObjGetDescriptor((void*)(mem.addr), mem.len);
        out = mem_md.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
S3DellObsObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    nixlObsObjMetadata *md = static_cast<nixlObsObjMetadata *>(meta);
    if (md) {
        if (md->nixlMem == OBJ_SEG) {
            std::unique_ptr<nixlObsObjMetadata> obj_md_ptr = std::unique_ptr<nixlObsObjMetadata>(md);
            devIdToObjKey_.erase(obj_md_ptr->devId);
        }
        else if ((md->nixlMem == DRAM_SEG) || (md->nixlMem == VRAM_SEG)) {
            std::unique_ptr<nixlObsObjMetadata> mem_md_ptr = std::unique_ptr<nixlObsObjMetadata>(md);
            cuClient_->cuMemObjPutDescriptor((void*)(mem_md_ptr->localAddr));
        }

    }
    return NIXL_SUCCESS;
}

nixl_status_t
S3DellObsObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        const std::string &local_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {

    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlObsObjBackendReqH* req_h = new nixlObsObjBackendReqH();

    for (int i = 0; i < local.descCount(); ++i) {
        obsObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);
        rdma_ctx ctx;

         if (operation == NIXL_WRITE) {
            cuClient_->cuObjPut(&ctx, (void *)req.addr, req.size, req.offset);
        }
        else if (operation == NIXL_READ) {
            cuClient_->cuObjGet(&ctx, (void *)req.addr, req.size, req.offset);
        }
        req.rdma_desc = ctx.rdma_desc;

        auto obj_key_search = devIdToObjKey_.find(remote[i].devId);
        if (obj_key_search == devIdToObjKey_.end()) {
            NIXL_ERROR << "The object segment key " << remote[i].devId
                       << " is not registered with the backend";
            delete req_h;
            return NIXL_ERR_INVALID_PARAM;
        }

        req.obj_key = obj_key_search->second;

        req_h->reqs_.push_back(req);
    }

    handle = req_h;

    return NIXL_SUCCESS;
}

nixl_status_t
S3DellObsObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);

    for (auto req : req_h->reqs_) {

        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        // S3 client interface signals completion via a callback, but NIXL API polls request handle
        // for the status code. Use future/promise pair to bridge the gap.
        if (operation == NIXL_WRITE) {
            s3Client_->putObjectRdmaAsync(
                req.obj_key, req.addr, req.size, req.offset, req.rdma_desc, [status_promise](bool success) {
                    status_promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                });
        }
        else {
            s3Client_->getObjectRdmaAsync(
                req.obj_key, req.addr, req.size, req.offset, req.rdma_desc, [status_promise](bool success) {
                    status_promise->set_value(success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                });
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
S3DellObsObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
S3DellObsObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    delete req_h;
    req_h = nullptr;
    return NIXL_SUCCESS;
}

iS3Client *
S3DellObsObjEngineImpl::getClient() const {
    return s3Client_.get();
}

