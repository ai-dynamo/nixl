/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "client.h"
#include "object/rdma/cuobj_client.h"
#include "obj_engine_registry.h"
#include "common/nixl_log.h"
#include <stdexcept>

namespace {
objAccelEngineRegistrar reg_dell(
    "dell",
    [](const nixlBackendInitParams *p) { return std::make_unique<S3DellObsObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3DellObsObjEngineImpl>(p, std::move(s3));
    });

std::shared_ptr<iS3Client>
makeDellClient(const nixlBackendInitParams *params,
               nixl_obj_rdma::SharedCuObjClient &rdma,
               std::shared_ptr<Aws::Utils::Threading::Executor> executor) {
    // RDMA GET has an empty HTTP body; preserve explicit checksum overrides.
    nixl_b_params_t custom = params->customParams ? *params->customParams : nixl_b_params_t{};
    custom.emplace("resp_checksum", "required");
    return std::make_shared<awsS3DellObsClient>(&custom, rdma, std::move(executor));
}
} // namespace

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3DellObsObjEngineImpl(init_params, nullptr) {}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                                               std::shared_ptr<iS3Client> s3_client,
                                               nixl_obj_rdma::SharedCuObjClient *rdma)
    : S3AccelObjEngineImpl(init_params, s3_client),
      rdma_(rdma ? rdma : nixl_obj_rdma::SharedCuObjClient::instance()) {
    if (!rdma_ || !rdma_->isConnected()) {
        throw std::runtime_error("Dell ObjectScale requires a connected cuObject RDMA fabric");
    }
    if (!s3_client) {
        s3Client_ = makeDellClient(init_params, *rdma_, executor_);
    }
}

nixl_obj_rdma::SharedCuObjClient *
S3DellObsObjEngineImpl::rdmaClient() const {
    return rdma_;
}

nixl_status_t
S3DellObsObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                    const nixl_mem_t &nixl_mem,
                                    nixlBackendMD *&out) {
    if (nixl_mem == DRAM_SEG || nixl_mem == VRAM_SEG) {
        if (mem.len > CUOBJ_MAX_MEMORY_REG_SIZE) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (!mem.addr || !mem.len) {
            return NIXL_ERR_BACKEND;
        }
    }
    return S3AccelObjEngineImpl::registerMem(mem, nixl_mem, out);
}

nixl_status_t
S3DellObsObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &remote_agent,
                                 const std::string &local_agent,
                                 nixlBackendReqH *&handle,
                                 const nixl_opt_b_args_t *opt_args) const {
    if (local.descCount() != remote.descCount()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    for (int i = 0; i < remote.descCount(); ++i) {
        if (devIdToObjKey_.find(remote[i].devId) == devIdToObjKey_.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    // Tokens are minted at post time, including when a prepared request is reused.
    return DefaultObjEngineImpl::prepXfer(
        operation, local, remote, remote_agent, local_agent, handle, opt_args);
}

nixl_status_t
S3DellObsObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &remote_agent,
                                 nixlBackendReqH *&handle,
                                 const nixl_opt_b_args_t *opt_args) const {
    if ((operation != NIXL_WRITE && operation != NIXL_READ) ||
        local.descCount() != remote.descCount() ||
        (local.getType() != DRAM_SEG && local.getType() != VRAM_SEG) ||
        remote.getType() != OBJ_SEG) {
        return NIXL_ERR_INVALID_PARAM;
    }
    // Validate the whole batch before launching any asynchronous request.
    for (int i = 0; i < remote.descCount(); ++i) {
        if ((operation == NIXL_WRITE && remote[i].addr != 0) ||
            devIdToObjKey_.find(remote[i].devId) == devIdToObjKey_.end()) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    return DefaultObjEngineImpl::postXfer(operation, local, remote, remote_agent, handle, opt_args);
}
