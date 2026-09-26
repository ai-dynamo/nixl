/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"

#include <algorithm>
#include <memory>
#include <string>

#include "object/rdma/rdma.h"
#include "obj_engine_registry.h"
#include "common/nixl_log.h"

namespace {

// Registered-memory metadata for the accel engine. It records what to clean up
// on deregister: an OBJ_SEG key mapping, or a DRAM/VRAM buffer pinned for RDMA.
class accelObjMetadata : public nixlBackendMD {
public:
    accelObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(std::move(obj_key)) {}

    accelObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          localAddr(addr),
          rdmaRegistered(true) {}

    ~accelObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId = 0;
    std::string objKey;
    uintptr_t localAddr = 0;
    bool rdmaRegistered = false;
};

// Register the standard-S3 engine under "s3"; obj_backend normalizes a missing
// `type` to "s3", so `accelerated=true` with no type resolves here too.
objAccelEngineRegistrar reg_s3_accel(
    "s3",
    [](const nixlBackendInitParams *p) { return std::make_unique<S3AccelObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3AccelObjEngineImpl>(p, std::move(s3));
    });

} // namespace

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params)
    : DefaultObjEngineImpl(init_params) {
    auto client = std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
    accelClient_ = client.get();
    s3Client_ = std::move(client);
    NIXL_INFO << "Object storage backend initialized with S3 Accel client";
}

S3AccelObjEngineImpl::S3AccelObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iS3Client> s3_client)
    : DefaultObjEngineImpl(init_params, s3_client, nullptr) {
    if (!s3_client) {
        auto client = std::make_shared<awsS3AccelClient>(init_params->customParams, executor_);
        accelClient_ = client.get();
        s3Client_ = std::move(client);
    } else {
        s3Client_ = s3_client;
        accelClient_ = dynamic_cast<awsS3AccelClient *>(s3Client_.get());
    }
    NIXL_INFO << "Object storage backend initialized with S3 Accel client (injected)";
}

iS3Client *
S3AccelObjEngineImpl::getClient() const {
    return s3Client_.get();
}

bool
S3AccelObjEngineImpl::rdmaReady() const {
    return accelClient_ != nullptr && accelClient_->supportsRdma();
}

nixl_mem_list_t
S3AccelObjEngineImpl::getSupportedMems() const {
    // VRAM_SEG (GPU-direct) is advertised only when the RDMA fast path is ready,
    // so a GPU pointer can never reach the HTTP path.
    if (rdmaReady()) {
        return {DRAM_SEG, OBJ_SEG, VRAM_SEG};
    }
    return {DRAM_SEG, OBJ_SEG};
}

nixl_status_t
S3AccelObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    const nixl_mem_list_t supported_mems = getSupportedMems();
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        auto obj_md = std::make_unique<accelObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
        return NIXL_SUCCESS;
    }

    // DRAM_SEG / VRAM_SEG: pin the buffer so the client can mint a cuObject token
    // for it. The accelerated path is RDMA-only (no HTTP fallback), so when the
    // fast path is ready a registration failure is a hard error rather than a
    // silent success that fails later at transfer time.
    if (rdmaReady()) {
        auto *rdma = nixl_obj_rdma::SharedCuObjClient::instance();
        if (!rdma || !rdma->registerBuffer(reinterpret_cast<void *>(mem.addr), mem.len)) {
            NIXL_ERROR << "RDMA buffer registration failed; the accelerated path has no HTTP "
                          "fallback";
            return NIXL_ERR_BACKEND;
        }
        out = std::make_unique<accelObjMetadata>(nixl_mem, mem.addr).release();
        return NIXL_SUCCESS;
    }

    // Not RDMA-ready (e.g. an injected non-accel client in tests): VRAM is never
    // advertised, so this is DRAM, which needs no pinning.
    out = nullptr;
    return NIXL_SUCCESS;
}

nixl_status_t
S3AccelObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    auto *obj_md = static_cast<accelObjMetadata *>(meta);
    if (obj_md) {
        std::unique_ptr<accelObjMetadata> obj_md_ptr(obj_md);
        if (obj_md->nixlMem == OBJ_SEG) {
            devIdToObjKey_.erase(obj_md->devId);
        } else if (obj_md->rdmaRegistered) {
            if (auto *rdma = nixl_obj_rdma::SharedCuObjClient::instance()) {
                rdma->deregisterBuffer(reinterpret_cast<void *>(obj_md->localAddr));
            }
        }
    }
    return NIXL_SUCCESS;
}
